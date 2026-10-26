# -*- coding: utf-8 -*-
"""Regenerate the qualitative panel so that it supports the paper's actual claim.

Columns: endoscopy, expert mask, U-Net, UNet++, FEE-Net (full), FEE-Net w/o EGM.
Each column header carries the mean test Dice of that configuration; each panel
carries the per-case Dice, so the figure and Table II cannot disagree.

Rows are the 10th, 50th and 90th percentile of lesion area over the 150 held-out
test frames -- a deterministic, stated selection rule.
"""
import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.unet import UNet                                    # noqa: E402
from models.attention_unet import AttentionUNet                  # noqa: E402
from models.unet_plus_plus import UNetPlusPlus                  # noqa: E402
from models.fee_net import FEENet                               # noqa: E402

DATA = os.path.join(ROOT, "data", "kvasir_seg")
CKPT = os.path.join(ROOT, "scratch", "kaggle_v5_results")
OUT = os.path.join(ROOT, "05_Paper_Draft", "figure_qualitative_v6.png")

COLS = [
    ("U-Net", UNet, {}, "best_unet.pth", 84.14),
    ("UNet++", UNetPlusPlus, {}, "best_unetplusplus.pth", 84.07),
    ("FEE-Net", FEENet, dict(use_dwt=True, use_egm=True, base_channels=48),
     "best_feenet.pth", 79.47),
    ("FEE-Net w/o EGM", FEENet, dict(use_dwt=True, use_egm=False, base_channels=48),
     "best_feenet_noegm.pth", 83.08),
]


def load_test():
    stems = list(json.load(open(os.path.join(DATA, "splits.json"), encoding="utf-8"))["test"])
    imgs = np.empty((len(stems), 3, 256, 256), np.float32)
    msks = np.empty((len(stems), 1, 256, 256), np.float32)
    for i, s in enumerate(stems):
        ip = glob.glob(os.path.join(DATA, "images", s + ".*"))[0]
        mp = glob.glob(os.path.join(DATA, "masks", s + ".*"))[0]
        im = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        imgs[i] = np.transpose(cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR),
                               (2, 0, 1)) / 255.0
        msks[i, 0] = (cv2.resize(mk, (256, 256), interpolation=cv2.INTER_NEAREST) > 127)
    return torch.from_numpy(imgs), torch.from_numpy(msks), stems


def dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else 2.0 * (p & g).sum() / denom


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, stems = load_test()

    # deterministic row selection: 10th / 50th / 90th percentile of lesion area
    areas = y[:, 0].sum(dim=(1, 2)).numpy()
    order = np.argsort(areas)
    picks = [int(order[int(q * (len(order) - 1))]) for q in (0.10, 0.50, 0.90)]
    print("[fig] selected cases:", [(stems[i], int(areas[i])) for i in picks])

    preds = {}
    for name, cls, kw, ckpt, mean_dice in COLS:
        m = cls(3, 1, **kw).to(dev).eval()
        sd = torch.load(os.path.join(CKPT, ckpt), map_location=dev, weights_only=False)
        m.load_state_dict(sd.get("model_state_dict", sd), strict=False)
        chunks = []
        with torch.no_grad():
            for i in range(0, len(x), 8):
                sl = slice(i, min(i + 8, len(x)))
                chunks.append(torch.sigmoid(m(x[sl].to(dev))).cpu().numpy()[:, 0])
            torch.cuda.empty_cache()
        p = np.concatenate(chunks) > 0.5
        preds[name] = p
        got = np.mean([dice(p[i], y[i, 0].numpy()) for i in range(len(stems))]) * 100
        print(f"[fig] {name:18s} recomputed mean Dice = {got:.2f} (table: {mean_dice})")

    gt = y[:, 0].numpy() > 0.5
    fig, axes = plt.subplots(3, 2 + len(COLS), figsize=(15.6, 8.2), dpi=300)
    heads = ["内镜图像", "专家标注"] + [f"{c[0]}\n（均值 {c[4]:.2f}%）" for c in COLS]

    for r, idx in enumerate(picks):
        panels = [np.transpose(x[idx].numpy(), (1, 2, 0)),
                  np.repeat(gt[idx][..., None], 3, axis=2).astype(float)]
        panels += [np.repeat(preds[c[0]][idx][..., None], 3, axis=2).astype(float) for c in COLS]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1), cmap=None if c == 0 else "gray")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color("#888888")
            if r == 0:
                ax.set_title(heads[c], fontsize=9.2, pad=6)
            if c >= 2:
                d = dice(preds[COLS[c - 2][0]][idx], gt[idx]) * 100
                ax.set_xlabel(f"{d:.1f}%", fontsize=8.4, labelpad=2)
    fig.text(0.5, 0.012, "每一格下方为该例 Dice；列标题中的均值为 150 例测试集上的均值 Dice",
             ha="center", fontsize=8.2, color="#333333")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, facecolor="white")
    print("[fig] wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
