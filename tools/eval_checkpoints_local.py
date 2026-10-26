# -*- coding: utf-8 -*-
"""
Evaluate saved checkpoints on the repository 150-case independent test set.

Why this exists
---------------
The v4 cloud run trained and saved all seven checkpoints but crashed in a later
cell (missing `import subprocess`), so its benchmark CSV was never written. This
script recovers the table locally with the repository loader and the repository
metrics, and -- importantly -- re-verifies the FEENetEarlyExit row that the v3
double-sigmoid bug had corrupted.

It also applies the same sanity envelope as tools/ingest_kaggle_results.py, so a
broken row is reported as rejected rather than silently returned.

Usage
-----
    python tools/eval_checkpoints_local.py --ckpt_dir scratch/kaggle_output_v4/outputs
    python tools/eval_checkpoints_local.py --ckpt_dir checkpoints --tag local_old
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.unet import UNet                        # noqa: E402
from models.attention_unet import AttentionUNet      # noqa: E402
from models.unet_plus_plus import UNetPlusPlus      # noqa: E402
from models.fee_net import FEENet                   # noqa: E402
from models.fee_net_early_exit import FEENetEarlyExit  # noqa: E402
from utils.metrics import MetricTracker                                  # noqa: E402

DATA = os.path.join(ROOT, "data", "kvasir_seg")
SPECS = {
    "UNet": (UNet, {}),
    "AttentionUNet": (AttentionUNet, {}),
    "UNetPlusPlus": (UNetPlusPlus, {}),
    "FEENet": (FEENet, dict(use_dwt=True, use_egm=True, base_channels=48)),
    "FEENetEarlyExit": (FEENetEarlyExit, dict(base_channels=48)),
    "FEENet_noDWT": (FEENet, dict(use_dwt=False, use_egm=True, base_channels=48)),
    "FEENet_noEGM": (FEENet, dict(use_dwt=True, use_egm=False, base_channels=48)),
}
CKPT_NAMES = {
    "UNet": "best_unet.pth", "AttentionUNet": "best_attentionunet.pth",
    "UNetPlusPlus": "best_unetplusplus.pth", "FEENet": "best_feenet.pth",
    "FEENetEarlyExit": "best_feenetearlyexit.pth", "FEENet_noDWT": "best_feenet_nodwt.pth",
    "FEENet_noEGM": "best_feenet_noegm.pth",
}
BOUNDS = {"Test Dice (%)": (40, 99), "Test mIoU (%)": (30, 99), "HD95 (px)": (1, 200),
          "Precision (%)": (40, 100), "Recall (%)": (40, 100)}


def load_test():
    splits = json.load(open(os.path.join(DATA, "splits.json"), encoding="utf-8"))
    stems = list(splits["test"])
    n = len(stems)
    imgs = np.empty((n, 3, 256, 256), np.float32)
    msks = np.empty((n, 1, 256, 256), np.float32)
    for i, s in enumerate(stems):
        ip = glob.glob(os.path.join(DATA, "images", s + ".*"))[0]
        mp = glob.glob(os.path.join(DATA, "masks", s + ".*"))[0]
        im = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        imgs[i] = np.transpose(cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR), (2, 0, 1)) / 255.0
        msks[i, 0] = (cv2.resize(mk, (256, 256), interpolation=cv2.INTER_NEAREST) > 127)
    return torch.from_numpy(imgs), torch.from_numpy(msks), stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    x, y, stems = load_test()
    print(f"[eval] test cases: {len(stems)} (from repository splits.json)")
    x, y = x.to(a.device), y.to(a.device)

    rows, rejected = [], []
    for name, (cls, kw) in SPECS.items():
        path = os.path.join(a.ckpt_dir, CKPT_NAMES[name])
        if not os.path.exists(path):
            print(f"[eval] skip {name}: no checkpoint at {path}")
            continue
        model = cls(3, 1, **kw).to(a.device).eval()
        sd = torch.load(path, map_location=a.device, weights_only=False)
        sd = sd.get("model_state_dict", sd)
        res = model.load_state_dict(sd, strict=False)
        if len(res.missing_keys) or len(res.unexpected_keys):
            print(f"[eval] {name}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
                  f"  <-- architecture drift")
        tk = MetricTracker()
        with torch.no_grad():
            for i in range(0, len(x), 8):
                sl = slice(i, min(i + 8, len(x)))
                # Repository API: EE returns probabilities unless raw logits are requested
                out = (model(x[sl], force_full=True, return_raw_logits=True)
                       if name == "FEENetEarlyExit" else model(x[sl]))
                tk.update(out, y[sl], from_logits=True)
        r = tk.get_results()
        row = {"Model": name,
               "Test Dice (%)": round(r["dice"] * 100, 2),
               "Test mIoU (%)": round(r["iou"] * 100, 2),
               "HD95 (px)": round(r["hd95"], 2),
               "Precision (%)": round(r["precision"] * 100, 2),
               "Recall (%)": round(r["recall"] * 100, 2),
               "Params (M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 3)}
        bad = [f"{k}={row[k]} out of [{lo},{hi}]" for k, (lo, hi) in BOUNDS.items()
               if not (lo <= row[k] <= hi)]
        if row["Test Dice (%)"] - row["Test mIoU (%)"] > 12:
            bad.append(f"Dice-mIoU gap {row['Test Dice (%)']-row['Test mIoU (%)']:.1f}")
        (rejected.append((name, bad)) if bad else rows.append(row))
        del model
        torch.cuda.empty_cache()

    hdr = list(rows[0].keys()) if rows else []
    print("\n=== validated rows ===")
    if rows:
        w = [max(len(str(h)), *(len(str(r[h])) for r in rows)) for h in hdr]
        print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
        for r in rows:
            print("  ".join(str(r[h]).ljust(w[i]) for i, h in enumerate(hdr)))
    print(f"\n[eval] valid={len(rows)} rejected={len(rejected)}")
    for n, b in rejected:
        print(f"  REJECT {n}: {'; '.join(b)}")

    out = a.out or os.path.join(ROOT, "scratch", f"local_eval_{a.tag or 'run'}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=hdr)
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
