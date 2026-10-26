# -*- coding: utf-8 -*-
"""
Local GPU probe harness -- fast configuration screening on the repository split.

Purpose: the authoritative benchmark runs on Kaggle (~80 min/run). This harness
trains the *repository* models on the same fold-0 data (680 train / 170 val) with
GPU-cached tensors so that loss-schedule questions can be answered in minutes.

It mirrors the Kaggle pipeline (Exp-030): repository models, repository
CompoundLoss, repository MetricTracker, explicit from_logits=True.

Examples
--------
    python tools/local_probe.py --model feenet --epochs 15
    python tools/local_probe.py --model unet   --epochs 15
    python tools/local_probe.py --model feenet --epochs 15 --no-edge --no-deep --no-boundary
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.unet import UNet                                    # noqa: E402
from models.fee_net import FEENet                                # noqa: E402
from models.fee_net_early_exit import FEENetEarlyExit            # noqa: E402
from utils.losses import CompoundLoss                            # noqa: E402
from utils.metrics import MetricTracker                          # noqa: E402

DATA = os.path.join(ROOT, "data", "kvasir_seg")
OUT = os.path.join(ROOT, "assets", "probe_results")
IMG_SIZE = 256
BATCH = 16
TRIAGE_PIXEL_THRESHOLD = 10


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(stems, img_dir, mask_dir):
    n = len(stems)
    imgs = np.empty((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    msks = np.empty((n, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, s in enumerate(stems):
        img = cv2.imread(os.path.join(img_dir, f"{s}.jpg"))
        if img is None:
            img = cv2.imread(os.path.join(img_dir, f"{s}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(mask_dir, f"{s}.jpg"), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = cv2.imread(os.path.join(mask_dir, f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        imgs[i] = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        msks[i, 0] = (msk > 127).astype(np.float32)
    return torch.from_numpy(imgs).cuda(), torch.from_numpy(msks).cuda()


def build(name):
    if name == "unet":
        return UNet(3, 1)
    if name == "feenet":
        return FEENet(3, 1, use_dwt=True, use_egm=True, base_channels=48)
    if name == "feenet_ee":
        return FEENetEarlyExit(3, 1, base_channels=48)
    raise ValueError(name)


def run(cfg):
    set_seed(cfg["seed"])
    splits = json.load(open(os.path.join(DATA, "splits.json"), encoding="utf-8"))
    tr = load_split(splits["fold_0"]["train"], os.path.join(DATA, "images"), os.path.join(DATA, "masks"))
    va = load_split(splits["fold_0"]["val"], os.path.join(DATA, "images"), os.path.join(DATA, "masks"))
    model = build(cfg["model"]).cuda()
    n_par = sum(p.numel() for p in model.parameters())

    crit = CompoundLoss(lambda_bce=1.0, lambda_dice=1.0,
                        lambda_boundary=cfg["w_bound"] if cfg["use_bound"] else 0.0).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    is_ee = cfg["model"] == "feenet_ee"
    is_fee = cfg["model"].startswith("feenet")

    traj = []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(tr[0]), device="cuda")
        tot, nb = torch.zeros(1, device="cuda"), 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            x, y = tr[0][idx], tr[1][idx]
            if random.random() > 0.5:
                x, y = torch.flip(x, [3]), torch.flip(y, [3])
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                if is_ee or is_fee:
                    out = model(x, return_aux=True)
                    seg, edge = out[0], out[1]
                    aux = out[-1]
                    loss, _ = crit(seg, edge if cfg["use_edge"] else None, y)
                    if cfg["use_deep"] and aux:
                        a = sum(crit(F.interpolate(z.float(), size=y.shape[2:], mode="bilinear",
                                                   align_corners=True), None, y)[0] for z in aux) / len(aux)
                        loss = loss + cfg["w_deep"] * a
                    if is_ee and cfg["use_triage"]:
                        tgt = (y.sum(dim=(1, 2, 3)) > TRIAGE_PIXEL_THRESHOLD).long()
                        loss = loss + 0.5 * F.cross_entropy(out[2], tgt)
                else:
                    loss, _ = crit(model(x), None, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.detach(); nb += 1
        sched.step()

        model.eval()
        tk = MetricTracker()
        with torch.no_grad():
            for i in range(0, len(va[0]), BATCH):
                sl = slice(i, min(i + BATCH, len(va[0])))
                out = model(va[0][sl], force_full=True, return_raw_logits=True) if is_ee else model(va[0][sl])
                tk.update(out, va[1][sl], from_logits=True)
        r = tk.get_results()
        traj.append({"epoch": ep, "loss": round(float(tot / max(nb, 1)), 4),
                     "val_dice": round(r["dice"] * 100, 2), "val_iou": round(r["iou"] * 100, 2),
                     "val_hd95": round(r["hd95"], 2)})
        print(f"[{cfg['tag']}] ep{ep:02d}/{cfg['epochs']} loss={traj[-1]['loss']:.4f} "
              f"val_dice={traj[-1]['val_dice']:.2f}% val_iou={traj[-1]['val_iou']:.2f}% "
              f"hd95={traj[-1]['val_hd95']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    best = max(traj, key=lambda r: r["val_dice"])
    res = {"cfg": {k: v for k, v in cfg.items()}, "params_m": round(n_par / 1e6, 3),
           "best": best, "last": traj[-1], "traj": traj,
           "minutes": round((time.time() - t0) / 60, 1)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{cfg['tag']}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[{cfg['tag']}] BEST val_dice={best['val_dice']:.2f}% @ep{best['epoch']} "
          f"| {res['minutes']} min | params={res['params_m']}M", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="feenet", choices=["unet", "feenet", "feenet_ee"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w-bound", type=float, default=0.1)
    ap.add_argument("--w-deep", type=float, default=0.3)
    ap.add_argument("--no-boundary", dest="use_bound", action="store_false")
    ap.add_argument("--no-edge", dest="use_edge", action="store_false")
    ap.add_argument("--no-deep", dest="use_deep", action="store_false")
    ap.add_argument("--no-triage", dest="use_triage", action="store_false")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = dict(model=a.model, epochs=a.epochs, lr=a.lr, seed=a.seed,
               w_bound=a.w_bound, w_deep=a.w_deep,
               use_bound=a.use_bound, use_edge=a.use_edge, use_deep=a.use_deep,
               use_triage=a.use_triage,
               tag=a.tag or f"{a.model}_e{a.epochs}_lr{a.lr:g}"
                        + ("" if a.use_bound else "_nb")
                        + ("" if a.use_edge else "_ne")
                        + ("" if a.use_deep else "_nd"))
    run(cfg)


if __name__ == "__main__":
    main()
