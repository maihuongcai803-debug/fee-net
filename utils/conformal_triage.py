# -*- coding: utf-8 -*-
"""
Conformal Risk Control for Clinical Triage
=========================================
Grounded in Finite-Sample Distribution-Free Uncertainty Quantification:
- Angelopoulos & Bates (2021): "A Gentle Introduction to Conformal Prediction and Distribution-Free UQ"
- Angelopoulos et al. (2022): "Conformal Risk Control", Harvard/UC Berkeley.
- Clopper & Pearson (1934): Exact Binomial Confidence Intervals.

Clinical Motivation:
In high-throughput endoscopic screening, empirical heuristic thresholds (e.g., manually tuning
tau until zero lesions are missed on a small test cohort) suffer from severe test-set overfitting
and lack finite-sample safety guarantees.

This module replaces heuristic tuning with Conformal Risk Control (CRC):
1. User specifies a clinically tolerable False Negative Rate bound alpha (e.g., alpha = 0.05 or 0.01)
   and a statistical confidence level 1 - delta (e.g., 95% or 99%).
2. On a held-out calibration cohort, we compute the finite-sample Upper Confidence Bound (UCB)
   of the False Negative Rate via the Clopper-Pearson exact binomial inversion.
3. We identify the optimal operating threshold tau_crc that maximizes normal mucosa bypass
   (saving computational FLOPs) while mathematically guaranteeing:
       P( FNR(tau_crc) <= alpha ) >= 1 - delta.
"""

import math
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import scipy.stats as stats
import torch


def compute_wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson Score confidence interval for a binomial proportion.
    Robust for small sample sizes and proportions near 0 or 1.
    """
    if total <= 0:
        return 0.0, 1.0

    p_hat = successes / total
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z ** 2
    n = total

    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    half_width = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n ** 2)))

    ci_lower = max(0.0, center - half_width)
    ci_upper = min(1.0, center + half_width)
    return float(ci_lower), float(ci_upper)


def compute_clopper_pearson_ucb(k_failures: int, n_trials: int, confidence: float = 0.95) -> float:
    """
    Computes the exact Clopper-Pearson Upper Confidence Bound for binomial failure rate.
    Uses the Beta distribution quantile:
        UCB_{1-delta}(k, n) = Beta^{-1}(1 - delta; k + 1, n - k)
    """
    if n_trials <= 0:
        return 1.0
    if k_failures >= n_trials:
        return 1.0
    if k_failures < 0:
        return 0.0

    return float(stats.beta.ppf(confidence, k_failures + 1, n_trials - k_failures))


class ConformalClinicalTriage:
    """
    Conformal Risk Control Manager for Clinical Video Triage.
    Provides mathematically sound, finite-sample statistical guarantees for clinical gating.
    """
    def __init__(self, target_fnr: float = 0.05, confidence: float = 0.95):
        """
        Args:
            target_fnr: Maximum acceptable False Negative Rate (alpha), default: 0.05 (>=95% Recall).
            confidence: Statistical guarantee level (1 - delta), default: 0.95 (95% confidence).
        """
        self.target_fnr = target_fnr
        self.confidence = confidence
        self.delta = 1.0 - confidence
        self.calibrated_tau: float = 0.50
        self.calibration_metadata: Dict[str, Any] = {}

    def calibrate(
        self,
        y_true: Union[np.ndarray, List[int], torch.Tensor],
        y_scores: Union[np.ndarray, List[float], torch.Tensor],
        num_grid_points: int = 2001
    ) -> float:
        """
        Calibrates the operating threshold tau using Conformal Risk Control on a held-out dataset.

        Args:
            y_true: Ground truth binary labels (0 = Normal, 1 = Lesion).
            y_scores: Predicted lesion risk probabilities in [0, 1].
            num_grid_points: Number of threshold evaluation points.

        Returns:
            calibrated_tau: The optimal threshold satisfying the finite-sample risk guarantee.
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_scores, torch.Tensor):
            y_scores = y_scores.detach().cpu().numpy()

        y_true = np.asarray(y_true).astype(int)
        y_scores = np.asarray(y_scores).astype(float)

        pos_mask = (y_true == 1)
        pos_scores = y_scores[pos_mask]
        n_pos = len(pos_scores)

        if n_pos == 0:
            raise ValueError("Calibration set must contain at least one positive (lesion) case!")

        thresholds = np.linspace(0.0, 1.0, num_grid_points)
        best_tau = 0.0
        best_ucb_fnr = 1.0
        best_emp_fnr = 0.0

        for tau in thresholds:
            k_misses = int(np.sum(pos_scores < tau))
            ucb_fnr = compute_clopper_pearson_ucb(k_misses, n_pos, confidence=self.confidence)

            if ucb_fnr <= self.target_fnr:
                best_tau = float(tau)
                best_ucb_fnr = float(ucb_fnr)
                best_emp_fnr = float(k_misses / n_pos)

        self.calibrated_tau = best_tau
        self.calibration_metadata = {
            "calibrated_tau": round(best_tau, 4),
            "target_fnr_alpha": self.target_fnr,
            "statistical_confidence_1_minus_delta": self.confidence,
            "calibration_lesion_count": n_pos,
            "calibration_total_count": len(y_true),
            "guaranteed_ucb_fnr": round(best_ucb_fnr, 4),
            "calibration_empirical_fnr": round(best_emp_fnr, 4),
            "guaranteed_sensitivity_lower_bound": round(1.0 - best_ucb_fnr, 4)
        }
        return self.calibrated_tau

    def evaluate_cohort(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_scores: Union[np.ndarray, List[float]],
        tau: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates triage performance on a test cohort with rigorous Wilson Score intervals.
        """
        if tau is None:
            tau = self.calibrated_tau

        y_true = np.asarray(y_true).astype(int)
        y_scores = np.asarray(y_scores).astype(float)

        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        n_total = len(y_true)
        n_pos = int(np.sum(pos_mask))
        n_neg = int(np.sum(neg_mask))

        # Predictions at threshold tau: 1 if score >= tau (flagged for full segmentation)
        flagged = (y_scores >= tau)

        # Lesion metrics
        if n_pos > 0:
            pos_detected = int(np.sum(flagged[pos_mask]))
            pos_missed = n_pos - pos_detected
            empirical_sensitivity = float(pos_detected / n_pos)
            ci_low, ci_high = compute_wilson_score_interval(pos_detected, n_pos, confidence=self.confidence)
        else:
            pos_detected = 0
            pos_missed = 0
            empirical_sensitivity = 1.0
            ci_low, ci_high = 0.0, 1.0

        # Normal metrics (Bypass)
        if n_neg > 0:
            neg_bypassed = int(np.sum(~flagged[neg_mask]))
            bypass_rate = float(neg_bypassed / n_neg)
            b_ci_low, b_ci_high = compute_wilson_score_interval(neg_bypassed, n_neg, confidence=self.confidence)
        else:
            neg_bypassed = 0
            bypass_rate = 0.0
            b_ci_low, b_ci_high = 0.0, 1.0

        return {
            "evaluated_tau": round(float(tau), 4),
            "total_cohort_size": n_total,
            "lesion_cohort_size": n_pos,
            "normal_cohort_size": n_neg,
            "lesions_detected": pos_detected,
            "lesions_missed": pos_missed,
            "empirical_sensitivity": round(empirical_sensitivity, 4),
            "wilson_ci_95_sensitivity": [round(ci_low, 4), round(ci_high, 4)],
            "normal_bypassed": neg_bypassed,
            "normal_bypass_rate": round(bypass_rate, 4),
            "wilson_ci_95_bypass_rate": [round(b_ci_low, 4), round(b_ci_high, 4)]
        }

    def save_calibration(self, output_path: str):
        """Saves calibration parameters and guarantees to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.calibration_metadata, f, indent=2, ensure_ascii=False)

    def load_calibration(self, input_path: str) -> float:
        """Loads calibration parameters and guarantees from JSON."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.calibration_metadata = data
        self.calibrated_tau = float(data.get("calibrated_tau", 0.50))
        self.target_fnr = float(data.get("target_fnr_alpha", 0.05))
        self.confidence = float(data.get("statistical_confidence_1_minus_delta", 0.95))
        return self.calibrated_tau


if __name__ == "__main__":
    print("=" * 65)
    print("Testing Conformal Risk Control for Clinical Triage")
    print("=" * 65)

    np.random.seed(42)
    # Simulated calibration set: 100 lesions, 400 normal mucosa
    n_cal_pos = 100
    n_cal_neg = 400
    y_cal_true = np.array([1] * n_cal_pos + [0] * n_cal_neg)
    y_cal_scores = np.concatenate([
        np.random.uniform(0.60, 0.99, n_cal_pos),   # Clear lesion scores
        np.random.uniform(0.01, 0.45, n_cal_neg)    # Normal mucosa scores
    ])

    # Instantiate Conformal Triage with alpha = 0.05 (95% Sensitivity bound) and 95% Confidence
    triage = ConformalClinicalTriage(target_fnr=0.05, confidence=0.95)
    tau = triage.calibrate(y_cal_true, y_cal_scores)

    print(f"1. Conformal Calibration Results:")
    print(f"   - Target False Negative Rate (alpha): {triage.target_fnr:.2%}")
    print(f"   - Statistical Confidence (1 - delta) : {triage.confidence:.2%}")
    print(f"   - Calibrated Operating Threshold tau : {tau:.4f}")
    print(f"   - Guaranteed Upper Bound FNR         : {triage.calibration_metadata['guaranteed_ucb_fnr']:.2%}")
    print(f"   - Guaranteed Sensitivity Lower Bound : {triage.calibration_metadata['guaranteed_sensitivity_lower_bound']:.2%}")

    # Simulated test cohort: 40 lesions, 160 normal
    n_test_pos = 40
    n_test_neg = 160
    y_test_true = np.array([1] * n_test_pos + [0] * n_test_neg)
    y_test_scores = np.concatenate([
        np.random.uniform(0.58, 0.99, n_test_pos),
        np.random.uniform(0.02, 0.42, n_test_neg)
    ])

    eval_results = triage.evaluate_cohort(y_test_true, y_test_scores)
    print(f"\n2. Test Cohort Rigorous Evaluation:")
    print(f"   - Cohort Size        : {eval_results['total_cohort_size']} (Lesions: {eval_results['lesion_cohort_size']})")
    print(f"   - Missed Lesions     : {eval_results['lesions_missed']} / {eval_results['lesion_cohort_size']}")
    print(f"   - Empirical Recall   : {eval_results['empirical_sensitivity']:.2%}")
    print(f"   - Wilson 95% CI      : [{eval_results['wilson_ci_95_sensitivity'][0]:.2%}, {eval_results['wilson_ci_95_sensitivity'][1]:.2%}]")
    print(f"   - Normal Bypass Rate : {eval_results['normal_bypass_rate']:.2%}")

    # Verify assertions
    assert tau > 0.0, "Calibrated tau must be positive"
    assert triage.calibration_metadata['guaranteed_ucb_fnr'] <= triage.target_fnr, "Guaranteed UCB FNR must be <= alpha"
    print("\n" + "=" * 65)
    print("CONFORMAL RISK CONTROL VERIFICATION PASSED PERFECTLY!")
    print("=" * 65)
