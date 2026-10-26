# -*- coding: utf-8 -*-
"""Redraw the accuracy-computation Pareto plane (Fig. 2).

Reads the authoritative benchmark CSV written by the Kaggle v5 run and plots
Dice against FLOPs; marker area scales inversely with HD95.  Labels follow the
same vocabulary as Tables 2/3 in the manuscript ("w/o EGM", "w/o DWT"),
so the figure never drifts from the tables.

Usage:
    python tools/plot_pareto_v5.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CSV = os.path.join(PROJECT_ROOT, "assets", "kaggle_v5_final",
                   "benchmark_metrics_independent_test.csv")

# 与表2/表3 完全一致的术语，避免图内出现"去 EGM"这类半中半英写法
LABELS = {
    "UNet":           ("U-Net",                   "o", "#4C78A8", (0, 26),   "center"),
    "AttentionUNet":  ("Attention U-Net",         "s", "#5AA469", (10, 16),  "left"),
    "UNetPlusPlus":   ("UNet++",                  "^", "#D9534F", (-10, 18), "right"),
    "FEENet":         ("FEE-Net（完整）",          "D", "#6C5B7B", (14, -16), "left"),
    "FEENetEarlyExit": ("FEE-Net-EE",             "v", "#8C6D46", (14, 6),   "left"),
    "FEENet_noDWT":   ("FEE-Net w/o DWT",         "P", "#C0504D", (14, -16), "left"),
    "FEENet_noEGM":   ("FEE-Net w/o EGM（推荐）",  "X", "#C9A227", (16, -20), "left"),
}


def load():
    with open(CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row.get("") or "").strip()
            if key in LABELS:
                yield (key,
                       float(row["Test Dice (%)"]),
                       float(row["HD95 (px)"]),
                       float(row["FLOPs (G)"]))


def main():
    rows = list(load())
    if not rows:
        raise SystemExit("no rows parsed from %s" % CSV)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.2, 4.7), dpi=220)

    hd = [r[2] for r in rows]
    lo, hi = min(hd), max(hd)
    for key, dice, hd95, fl in rows:
        name, marker, color, off, ha = LABELS[key]
        size = 60 + 320 * (hi - hd95) / (hi - lo) if hi > lo else 140
        ax.scatter(fl, dice, s=size, marker=marker, color=color,
                   edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(f"{name}\n{dice:.2f}% / {fl:.2f}G", (fl, dice),
                    textcoords="offset points", xytext=off,
                    fontsize=7.6, color="#333333",
                    ha=ha)

    ax.set_xlabel("计算量（GFLOPs，256×256，批大小 1）", fontsize=10)
    ax.set_ylabel("测试集 Dice (%)", fontsize=10)
    ax.set_title("150 例留出测试划分上的精度—计算平面", fontsize=11, pad=10)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#BBBBBB", zorder=0)
    ax.set_xlim(8, 37)
    ax.set_ylim(76, 86)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    out = os.path.join(PROJECT_ROOT, "05_Paper_Draft", "figure_pareto_v5_zh.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
