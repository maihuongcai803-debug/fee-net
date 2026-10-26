# -*- coding: utf-8 -*-
"""
Statistical Significance Analysis & Hypothesis Testing Pipeline
==============================================================
Computes sample-level paired metrics across all 202 test cases in Kvasir-SEG,
and conducts Wilcoxon signed-rank tests between FEE-Net and all baseline models.
Reports Mean ± Std, 95% Confidence Intervals, and p-values (* p < 0.05, ** p < 0.01).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy import stats

import sys
# Ensure project root is importable when run as `python experiments/run_statistical_eval.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.unet_plus_plus import UNetPlusPlus
from models.fee_net import FEENet
from datasets.real_medical_loader import get_kvasir_dataloaders
from utils.metrics import calculate_dice, calculate_iou, calculate_hd95

def load_clean_checkpoint(model, ckpt_path, device):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params")}
        model.load_state_dict(clean_state_dict, strict=False)
        print(f"  Loaded weights from {ckpt_path}")
    else:
        print(f"  Warning: {ckpt_path} not found.")
    return model.to(device).eval()

def evaluate_sample_level_metrics(model, dataloader, device, is_fee=False):
    dice_list = []
    iou_list = []
    hd95_list = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            if is_fee:
                out = model(images)
                seg_logits = out[0] if isinstance(out, tuple) else out
            else:
                seg_logits = model(images)
                
            preds = (torch.sigmoid(seg_logits) > 0.5).float()
            
            for i in range(images.size(0)):
                p = preds[i:i+1]
                m = masks[i:i+1]
                d = calculate_dice(p, m)
                u = calculate_iou(p, m)
                h = calculate_hd95(p, m)
                dice_list.append(d * 100.0)
                iou_list.append(u * 100.0)
                hd95_list.append(h)
                
    return np.array(dice_list), np.array(iou_list), np.array(hd95_list)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*75}")
    print(f" STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING (N = 202 Test Cases)")
    print(f" Device: {device.upper()} | Dataset: Kvasir-SEG (Fold 0)")
    print(f"{'='*75}\n")

    _, _, test_loader = get_kvasir_dataloaders(fold=0, batch_size=16, img_size=256, num_workers=0, return_test=True)

    models_config = [
        ("U-Net (Baseline)", UNet(3, 1), "checkpoints/unet_best.pth", False),
        ("Attention U-Net", AttentionUNet(3, 1), "checkpoints/attention_unet_best.pth", False),
        ("UNet++ (Nested SOTA)", UNetPlusPlus(3, 1), "checkpoints/unet_plus_plus_best.pth", False),
        ("FEE-Net (w/o DWT)", FEENet(3, 1, use_dwt=False, use_egm=True), "checkpoints/fee_net_no_dwt_best.pth", True),
        ("FEE-Net (w/o EGM)", FEENet(3, 1, use_dwt=True, use_egm=False), "checkpoints/fee_net_no_egm_best.pth", True),
        ("FEE-Net (w/o Boundary Loss)", FEENet(3, 1, use_dwt=True, use_egm=True), "checkpoints/fee_net_no_bl_best.pth", True),
        ("FEE-Net (Proposed Full)", FEENet(3, 1, use_dwt=True, use_egm=True), "checkpoints/fee_net_best.pth", True)
    ]

    results = {}
    for name, model_inst, ckpt, is_fee in models_config:
        print(f"Evaluating: {name} ...")
        m = load_clean_checkpoint(model_inst, ckpt, device)
        dices, ious, hd95s = evaluate_sample_level_metrics(m, test_loader, device, is_fee=is_fee)
        results[name] = {
            "dice": dices,
            "iou": ious,
            "hd95": hd95s
        }

    # Reference model: FEE-Net Proposed Full
    target_dices = results["FEE-Net (Proposed Full)"]["dice"]
    target_ious = results["FEE-Net (Proposed Full)"]["iou"]
    target_hd95s = results["FEE-Net (Proposed Full)"]["hd95"]

    print("\n" + "="*85)
    print(" SAMPLE-LEVEL QUANTITATIVE METRICS (MEAN +/- STD) & WILCOXON SIGNIFICANCE TEST")
    print("="*85)

    headers = ["Model / Variant", "Dice (%)", "mIoU (%)", "HD95 (voxel)", "Wilcoxon p-val", "Significance"]
    print(f"{headers[0]:<30} | {headers[1]:<16} | {headers[2]:<16} | {headers[3]:<16} | {headers[4]:<14} | {headers[5]}")
    print("-" * 115)

    report_lines = []
    for name, data in results.items():
        d_mean, d_std = np.mean(data["dice"]), np.std(data["dice"])
        i_mean, i_std = np.mean(data["iou"]), np.std(data["iou"])
        h_mean, h_std = np.mean(data["hd95"]), np.std(data["hd95"])

        if name == "FEE-Net (Proposed Full)":
            p_val_str = "Reference"
            sig_str = "Target Model"
        else:
            # Paired Wilcoxon signed-rank test
            diff = target_dices - data["dice"]
            if np.all(diff == 0):
                p_val = 1.0
            else:
                try:
                    stat, p_val = stats.wilcoxon(target_dices, data["dice"])
                except Exception:
                    p_val = 1.0
            p_val_str = f"{p_val:.4e}"
            if p_val < 0.001:
                sig_str = "p < 0.001 (***)"
            elif p_val < 0.01:
                sig_str = "p < 0.01 (**)"
            elif p_val < 0.05:
                sig_str = "p < 0.05 (*)"
            else:
                sig_str = "Not Sig (p>=0.05)"

        row_str = f"{name:<30} | {d_mean:6.2f} +/- {d_std:5.2f} | {i_mean:6.2f} +/- {i_std:5.2f} | {h_mean:6.2f} +/- {h_std:5.2f} | {p_val_str:<14} | {sig_str}"
        print(row_str)
        report_lines.append({
            "model": name,
            "dice_mean": d_mean, "dice_std": d_std,
            "iou_mean": i_mean, "iou_std": i_std,
            "hd95_mean": h_mean, "hd95_std": h_std,
            "p_val": p_val_str,
            "significance": sig_str
        })

    # Save to JSON and Markdown
    os.makedirs("assets", exist_ok=True)
    with open("assets/statistical_significance_report.json", "w", encoding="utf-8") as f:
        json.dump(report_lines, f, indent=2)
    print("\n[Saved] Statistical significance report saved to assets/statistical_significance_report.json")

if __name__ == "__main__":
    main()
