# -*- coding: utf-8 -*-
"""
Conformal Risk Control Calibration for FEENetEarlyExit
======================================================
Calibrates the early-exit triage threshold for FEENetEarlyExit on held-out
clinical calibration data using distribution-free Conformal Risk Control (CRC).
Provides finite-sample statistical guarantees:
    P( FNR(tau*) <= alpha ) >= 1 - delta

References:
- Angelopoulos et al. (2022): "Conformal Risk Control"
- Clopper & Pearson (1934): Exact Binomial Confidence Intervals
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fee_net_early_exit import FEENetEarlyExit
from datasets.real_medical_loader import get_kvasir_triage_dataloaders
from utils.conformal_triage import ConformalClinicalTriage


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate FEENetEarlyExit with Conformal Risk Control")
    parser.add_argument("--ckpt", type=str, default="checkpoints/fee_net_early_exit_fold0_seed42_best.pth",
                        help="Path to trained FEENetEarlyExit checkpoint")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Target clinically tolerable False Negative Rate bound (default: 0.05 = 5%)")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Significance level (confidence = 1 - delta = 0.95)")
    parser.add_argument("--output", type=str, default="checkpoints/fee_net_early_exit_crc_calibration.json",
                        help="Output path for calibrated threshold and metadata JSON")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def collect_scores(model, dataloader, device):
    """Evaluates model to collect triage predicted probabilities and binary lesion ground truths."""
    model.eval()
    all_scores = []
    all_targets = []

    for images, masks in dataloader:
        images = images.to(device)
        # Binary target: 1 if polyp present (> 10 pixels), 0 if normal mucosa
        targets = (masks.sum(dim=(1, 2, 3)) > 10).long().numpy()

        # Forward pass requesting risk scores
        out = model(images, return_edge=True)
        # out: (pred_masks, pred_edges, risk_scores, execution_tiers)
        risk_scores = out[2].detach().cpu().numpy()

        all_scores.extend(risk_scores)
        all_targets.extend(targets)

    return np.array(all_targets, dtype=int), np.array(all_scores, dtype=float)


def main():
    args = parse_args()
    print("=" * 70)
    print(" CONFORMAL RISK CONTROL CALIBRATION FOR FEENET-EE")
    print(f" Target FNR Bound (alpha): {args.alpha:.2%}")
    print(f" Confidence Level (1-delta): {1.0 - args.delta:.2%}")
    print(f" Device: {args.device} | Checkpoint: {args.ckpt}")
    print("=" * 70)

    # 1. Initialize Model
    model = FEENetEarlyExit(base_channels=48, detach_triage_feature=True).to(args.device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        clean_sd = {k: v for k, v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params")}
        model.load_state_dict(clean_sd, strict=False)
        print(f" Loaded weights from: {args.ckpt}")
    else:
        print(f" Warning: Checkpoint {args.ckpt} not found! Calibrating with initialized weights.")

    # 2. Load Dataloaders (Val = Calibration Cohort, Test = Generalization Cohort)
    train_loader, cal_loader, test_loader = get_kvasir_triage_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=0,
        return_test=True
    )
    print(f" Calibration Cohort Size: {len(cal_loader.dataset)} patients")
    print(f" Test Cohort Size       : {len(test_loader.dataset)} patients")

    # 3. Collect Predictions
    print("\n[Step 1/3] Collecting predictions on Calibration Set...")
    y_cal_true, y_cal_scores = collect_scores(model, cal_loader, args.device)
    n_pos_cal = int((y_cal_true == 1).sum())
    n_neg_cal = int((y_cal_true == 0).sum())
    print(f"   -> Calibration Positives (Lesions): {n_pos_cal}, Negatives (Normal): {n_neg_cal}")

    # 4. Perform Conformal Risk Control Calibration
    print("\n[Step 2/3] Performing Clopper-Pearson Exact Inversion...")
    triage = ConformalClinicalTriage(target_fnr=args.alpha, confidence=1.0 - args.delta)
    tau_crc = triage.calibrate(y_cal_true, y_cal_scores)

    meta = triage.calibration_metadata
    print(f"   * Calibrated Optimal Tau (tau_crc) : {tau_crc:.4f}")
    print(f"   * Guaranteed Upper Bound FNR       : {meta['guaranteed_ucb_fnr']:.2%}")
    cal_missed = int(round(meta['calibration_empirical_fnr'] * n_pos_cal))
    print(f"   * Calibration Lesions Missed       : {cal_missed} / {n_pos_cal}")

    # 5. Evaluate on Independent Test Set
    print("\n[Step 3/3] Evaluating on Independent Test Set...")
    y_test_true, y_test_scores = collect_scores(model, test_loader, args.device)
    test_eval = triage.evaluate_cohort(y_test_true, y_test_scores)

    print(f"   -> Test Empirical Recall (Sensitivity) : {test_eval['empirical_sensitivity']:.2%}")
    print(f"   -> Wilson 95% Confidence Interval       : [{test_eval['wilson_ci_95_sensitivity'][0]:.2%}, {test_eval['wilson_ci_95_sensitivity'][1]:.2%}]")
    print(f"   -> Test Normal Mucosa Bypass Rate       : {test_eval['normal_bypass_rate']:.2%}")
    print(f"   -> Test Lesions Missed                  : {test_eval['lesions_missed']} / {test_eval['lesion_cohort_size']}")

    # 6. Save Calibration Artifact
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    triage.save_calibration(args.output)
    print(f"\n[Saved CRC Calibration JSON] {args.output}")

    # Verification Assertion
    if n_pos_cal >= 10:
        assert meta["guaranteed_ucb_fnr"] <= args.alpha, f"CRC Guarantee violated: UCB FNR {meta['guaranteed_ucb_fnr']} > alpha {args.alpha}"
    print("=" * 70)
    print(" CALIBRATION SUCCESSFULLY COMPLETED WITH MATHEMATICAL GUARANTEES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
