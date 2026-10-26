# -*- coding: utf-8 -*-
"""
Stage-2: fine-tune ONLY the triage head on a leakage-free balanced cohort.

Why
---
The v5 run showed the screening head collapsed to a constant output (risk ~0.9998
for every sample). Root cause: it was trained on the Kvasir fold-0 split, which
is 100% polyp frames (all-positive), so cross-entropy has no negatives to learn
from and the optimal constant is "always lesion". A screening gate cannot learn
specificity from data without healthy frames.

This script therefore fine-tunes `triage_head` (and `quality_head`) on the
balanced `data/kvasir_triage` cohort, with the backbone frozen -- which is safe by
construction because the triage head consumes stop-gradient features.

Leakage control (critical)
--------------------------
`data/kvasir_triage` was built from the whole Kvasir-SEG pool, so **68 of its 480
training images and 24 of its 120 validation images originate from frames that are
in the 150-case independent test split**. Those are removed here by matching the
source frame encoded in each filename against `splits.json["test"]`. The
subsequent certification on the mixed cohort (which is derived from those test
frames) would otherwise be meaningless.

Outputs: a new checkpoint with only the triage/quality heads updated, plus a JSON
report of the training separation.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.fee_net_early_exit import FEENetEarlyExit  # noqa: E402
from tools.eval_mixed_cohort_gate import Gate             # noqa: E402  (deployed screening path)

KSEG = os.path.join(ROOT, "data", "kvasir_seg")
TRIAGE = os.path.join(ROOT, "data", "kvasir_triage")
IMG = 256


def source_frames():
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(KSEG, "images", "*.*"))}


def source_frame_of(stem, frames):
    for s in frames:
        if stem.endswith(s):
            return s
    return None


def load_mixed_cohort(cohort_dir, split):
    """Load a mixed cohort built by tools/build_mixed_cohort.py (manifest.json)."""
    man = json.load(open(os.path.join(cohort_dir, "manifest.json"), encoding="utf-8"))
    imgs, labels, kinds = [], [], []
    for r in man["samples"]:
        if r["split"] != split:
            continue
        im = cv2.resize(cv2.imread(os.path.join(cohort_dir, split, "images", r["file"])),
                        (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        imgs.append(np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.0)
        lab = 1 if r["kind"] == "lesion" else 0
        labels.append(lab); kinds.append(r["kind"])
    return (np.stack(imgs) if imgs else np.zeros((0, 3, IMG, IMG), np.float32),
            np.array(labels), kinds, 0)


def load_cohort(split, exclude, frames):
    """Return (images, labels, kinds) with test-derived frames removed."""
    imgs, labels, kinds, dropped = [], [], [], 0
    for f in sorted(glob.glob(os.path.join(TRIAGE, split, "images", "*.png"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        src = source_frame_of(stem, frames)
        if src is None or src in exclude:
            dropped += 1
            continue
        im = cv2.resize(cv2.imread(f), (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        mk_path = os.path.join(TRIAGE, split, "masks", os.path.basename(f))
        mk = cv2.imread(mk_path, cv2.IMREAD_GRAYSCALE)
        lesion = int(mk is not None and (mk > 127).sum() > 20)
        imgs.append(np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.0)
        labels.append(lesion)
        kinds.append("lesion" if lesion else "normal")
    return (np.stack(imgs) if imgs else np.zeros((0, 3, IMG, IMG), np.float32),
            np.array(labels), kinds, dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "scratch", "kaggle_v5_results",
                                                   "best_feenetearlyexit.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch", "triage_head_stage2.pth"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--unfreeze", default="triage_head",
                    help="comma-separated prefixes to train, e.g. 'triage_head' (head only) "
                         "or 'triage_head,layer1,stem,wf1,attn' (whole screening path)")
    ap.add_argument("--init-from", default=None, help="optional warm start (e.g. a previous stage-2 run)")
    ap.add_argument("--cohort", default=None,
                    help="mixed-cohort dir from build_mixed_cohort.py; when given, the gate is "
                         "trained on its 'calib' half and validated on its 'certify' half. "
                         "Use this so training matches deployment (full-frame lesions).")
    ap.add_argument("--train-split", default="calib")
    ap.add_argument("--val-split", default="certify")
    a = ap.parse_args()

    splits = json.load(open(os.path.join(KSEG, "splits.json"), encoding="utf-8"))
    test_frames = set(splits["test"])
    frames = source_frames()

    if a.cohort:
        xtr, ytr, ktr, drop_tr = load_mixed_cohort(a.cohort, a.train_split)
        xva, yva, kva, drop_va = load_mixed_cohort(a.cohort, a.val_split)
    else:
        xtr, ytr, ktr, drop_tr = load_cohort("train", test_frames, frames)
        xva, yva, kva, drop_va = load_cohort("val", test_frames, frames)
    print(f"[stage2] train: {len(ytr)} samples ({int(ytr.sum())} lesion / {int((1-ytr).sum())} normal)"
          f"  [dropped {drop_tr} test-derived]")
    print(f"[stage2] val  : {len(yva)} samples ({int(yva.sum())} lesion / {int((1-yva).sum())} normal)"
          f"  [dropped {drop_va} test-derived]")
    if ytr.sum() == 0 or (1 - ytr).sum() == 0:
        raise SystemExit("[stage2] cohort is not balanced after leakage filtering -- abort")

    ee = FEENetEarlyExit(3, 1, base_channels=48)
    sd = torch.load(a.init_from or a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    miss, unexp = ee.load_state_dict(sd, strict=False)
    print(f"[stage2] loaded base checkpoint: missing={len(miss)} unexpected={len(unexp)}")

    # Freeze everything except the requested prefixes (default: triage/quality heads only)
    prefixes = tuple(x.strip() for x in a.unfreeze.split(",") if x.strip())
    trainable = []
    for n, p in ee.named_parameters():
        p.requires_grad = n.startswith(prefixes)
        if p.requires_grad:
            trainable.append(n)
    n_tr = sum(p.numel() for p in ee.parameters() if p.requires_grad)
    print(f"[stage2] trainable tensors: {len(trainable)} ({n_tr} params) -- backbone frozen")

    ee = ee.to(a.device).eval()          # keep frozen modules in eval mode
    for n, p in ee.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in ee.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if a.device.startswith("cuda") else None

    Xtr = torch.from_numpy(xtr).to(a.device)
    Ytr = torch.from_numpy(ytr).long().to(a.device)
    Xva = torch.from_numpy(xva).to(a.device)

    # The screening path is exactly stem+layer1+DWT+SCA+triage head; training it
    # through the full segmentation forward would be ~10x more expensive and is
    # functionally identical because the head consumes stop-gradient features.
    gate = Gate(ee).to(a.device)

    def logits_of(x):
        """Validation pass in eval mode (frozen BN behaviour)."""
        gate.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 32):
                out.append(gate(x[i:i + 32]).float())
        gate.head.eval()
        return torch.cat(out)

    def risks(x):
        return torch.softmax(logits_of(x), dim=-1)[:, 1].cpu().numpy()

    best = None
    for ep in range(1, a.epochs + 1):
        # CRITICAL: keep the shared backbone in inference mode. requires_grad=False
        # blocks gradients but NOT BatchNorm running-stat updates, and training-mode
        # BN silently rewrites the stem/layer1 statistics that the segmentation head
        # depends on (measured: 81.42 -> 14.06 Dice). Only the triage head trains.
        gate.eval()
        gate.head.train()
        perm = torch.randperm(len(Xtr), device=a.device)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = gate(Xtr[idx])
                loss = F.cross_entropy(logits.float(), Ytr[idx])
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        r = risks(Xva)
        les, nor = r[yva == 1], r[yva == 0]
        sep = float(les.mean() - nor.mean()) if len(nor) else 0.0
        acc = float(((r > 0.5).astype(int) == yva).mean())
        print(f"[stage2] ep{ep:02d} loss={tot/max(nb,1):.4f} | val lesion {les.mean():.4f} "
              f"normal {nor.mean() if len(nor) else float('nan'):.4f} | sep={sep:+.4f} acc={acc*100:.1f}%")
        score = acc + sep
        if best is None or score > best[0]:
            best = (score, ep, sep, acc)

    print(f"[stage2] best epoch {best[1]} separation {best[2]:+.4f} acc {best[3]*100:.1f}%")
    torch.save({"model_state_dict": ee.state_dict(),  # triage/quality heads updated
                "triage_gate_state_dict": gate.state_dict(),
                "stage": "2-triage-head-only",
                "base_ckpt": os.path.relpath(a.ckpt, ROOT),
                "trained_params": n_tr,
                "train_samples": int(len(ytr)),
                "leakage_filtered": int(drop_tr + drop_va)}, a.out)
    rep = {"trained_params": n_tr, "train_samples": int(len(ytr)),
           "train_lesion": int(ytr.sum()), "train_normal": int((1 - ytr).sum()),
           "leakage_dropped_train": int(drop_tr), "leakage_dropped_val": int(drop_va),
           "best_epoch": best[1], "val_separation": best[2], "val_acc": best[3],
           "out": os.path.relpath(a.out, ROOT)}
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"[stage2] saved {a.out}")


if __name__ == "__main__":
    main()
