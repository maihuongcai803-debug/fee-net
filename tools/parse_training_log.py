# -*- coding: utf-8 -*-
"""
Parse Kaggle Training Log and Plot High-Resolution Convergence Curves
"""

import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

log_path = r"d:\machine_learning\notebook19aa62670b.log"
with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

pattern = re.compile(
    r"Epoch \[(\d+)/60\] \| Train: ([\d\.]+) \| Val Dice: ([\d\.]+)%, IoU: ([\d\.]+)%, HD95: ([\d\.]+)"
)

folds_data = {}
current_fold = -1

for line in text.splitlines():
    if ">>> Starting Fold" in line:
        m_fold = re.search(r"Starting Fold (\d+)", line)
        if m_fold:
            current_fold = int(m_fold.group(1))
            folds_data[current_fold] = {
                "epochs": [],
                "train_loss": [],
                "val_dice": [],
                "val_iou": [],
                "val_hd95": []
            }
    m = pattern.search(line)
    if m and current_fold != -1:
        ep = int(m.group(1))
        tr_loss = float(m.group(2))
        dice = float(m.group(3))
        iou = float(m.group(4))
        hd = float(m.group(5))
        folds_data[current_fold]["epochs"].append(ep)
        folds_data[current_fold]["train_loss"].append(tr_loss)
        folds_data[current_fold]["val_dice"].append(dice)
        folds_data[current_fold]["val_iou"].append(iou)
        folds_data[current_fold]["val_hd95"].append(hd)

print("Parsed Folds:", list(folds_data.keys()))
for f, d in folds_data.items():
    best_idx = int(np.argmax(d["val_dice"]))
    best_ep = d["epochs"][best_idx]
    best_dice = d["val_dice"][best_idx]
    min_loss = min(d["train_loss"])
    print(f"Fold {f}: {len(d['epochs'])} checkpoints recorded | Min Loss: {min_loss:.4f} | Peak Dice: {best_dice:.2f}% at Epoch {best_ep}")

out_json = r"d:\machine_learning\assets\fee_net_5fold_training_log.json"
with open(out_json, "w", encoding="utf-8") as fp:
    json.dump(folds_data, fp, indent=2)
print(f"Structured metrics JSON written to: {out_json}")

# Plot publication-grade convergence figures
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Subplot 1: Training Loss Convergence
ax1 = axes[0]
for f in range(5):
    d = folds_data[f]
    ax1.plot(d["epochs"], d["train_loss"], label=f"Fold {f}", color=colors[f], lw=2, alpha=0.85)

ax1.axvline(x=25, color="gray", linestyle="--", alpha=0.7, label="25-Epoch Benchmark (Steep)")
ax1.axvspan(45, 60, color="green", alpha=0.1, label="Flat Convergence Basin (Epochs 45-60)")
ax1.set_title("Training Loss Convergence Curve (All 5 Folds)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Compound Loss (BCE + Dice + Boundary-IoU)", fontsize=11)
ax1.set_xlim(1, 60)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=9)

# Subplot 2: Validation Dice (%) Convergence
ax2 = axes[1]
for f in range(5):
    d = folds_data[f]
    ax2.plot(d["epochs"], d["val_dice"], label=f"Fold {f}", color=colors[f], lw=2, alpha=0.85)
    best_idx = int(np.argmax(d["val_dice"]))
    ax2.scatter(d["epochs"][best_idx], d["val_dice"][best_idx], color=colors[f], s=60, zorder=5)

ax2.axvline(x=25, color="gray", linestyle="--", alpha=0.7, label="25-Epoch Level (~72-77%)")
ax2.axvspan(45, 60, color="green", alpha=0.1, label="Optimal Plateau (80.2% - 83.3%)")
ax2.set_title("Validation Dice (%) Convergence Curve (All 5 Folds)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Validation Dice Similarity Coefficient (%)", fontsize=11)
ax2.set_xlim(1, 60)
ax2.set_ylim(0, 90)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", fontsize=9)

plt.tight_layout()
fig_out = r"d:\machine_learning\assets\fee_net_convergence_curve.png"
plt.savefig(fig_out)
plt.close()
print(f"Convergence plot saved to: {fig_out}")
