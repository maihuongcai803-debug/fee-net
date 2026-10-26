import unittest
import numpy as np
import tempfile
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.conformal_triage import (
    ConformalClinicalTriage,
    compute_wilson_score_interval,
    compute_clopper_pearson_ucb
)


class TestConformalTriage(unittest.TestCase):
    def test_wilson_score_interval(self):
        """Test Wilson score interval computation on various proportions."""
        # 100% successes on 2 cases: Wilson CI must not be [1.0, 1.0]
        low, high = compute_wilson_score_interval(2, 2, confidence=0.95)
        self.assertLess(low, 0.50, "Wilson CI lower bound for 2/2 should be ~0.34")
        self.assertEqual(high, 1.0)

        # 0 successes
        low_zero, high_zero = compute_wilson_score_interval(0, 10, confidence=0.95)
        self.assertEqual(low_zero, 0.0)
        self.assertGreater(high_zero, 0.0)

        # 50% successes
        low_half, high_half = compute_wilson_score_interval(50, 100, confidence=0.95)
        self.assertAlmostEqual((low_half + high_half) / 2.0, 0.50, places=2)

    def test_clopper_pearson_ucb(self):
        """Test Clopper-Pearson exact upper confidence bound."""
        # 0 misses out of 100 positive cases at 95% confidence
        # Upper bound on miss rate should be ~2.95% (rule of three: ~3/n)
        ucb = compute_clopper_pearson_ucb(0, 100, confidence=0.95)
        self.assertGreater(ucb, 0.0)
        self.assertLess(ucb, 0.035)

        # 10 misses out of 100 cases
        ucb_10 = compute_clopper_pearson_ucb(10, 100, confidence=0.95)
        self.assertGreater(ucb_10, 0.10)
        self.assertLess(ucb_10, 0.18)

    def test_conformal_calibration_guarantee(self):
        """Test Conformal Clinical Triage calibrates threshold satisfying UCB FNR <= alpha."""
        np.random.seed(42)
        n_pos = 100
        n_neg = 300
        y_true = np.array([1] * n_pos + [0] * n_neg)
        y_scores = np.concatenate([
            np.random.uniform(0.65, 0.99, n_pos),
            np.random.uniform(0.01, 0.40, n_neg)
        ])

        triage = ConformalClinicalTriage(target_fnr=0.05, confidence=0.95)
        tau = triage.calibrate(y_true, y_scores)

        self.assertGreater(tau, 0.0, "Calibrated threshold must be positive")
        self.assertLessEqual(
            triage.calibration_metadata["guaranteed_ucb_fnr"],
            0.05,
            "Guaranteed UCB FNR must be <= alpha"
        )
        self.assertGreaterEqual(
            triage.calibration_metadata["guaranteed_sensitivity_lower_bound"],
            0.95,
            "Guaranteed sensitivity lower bound must be >= 1 - alpha"
        )

    def test_cohort_evaluation_and_serialization(self):
        """Test test-cohort evaluation and JSON calibration serialization."""
        triage = ConformalClinicalTriage(target_fnr=0.05, confidence=0.95)
        y_cal_true = np.array([1] * 100 + [0] * 100)
        y_cal_scores = np.concatenate([np.ones(100) * 0.9, np.zeros(100)])
        triage.calibrate(y_cal_true, y_cal_scores)

        # Evaluate cohort
        eval_res = triage.evaluate_cohort([1, 1, 0, 0], [0.95, 0.92, 0.01, 0.05])
        self.assertEqual(eval_res["lesions_detected"], 2)
        self.assertEqual(eval_res["lesions_missed"], 0)
        self.assertEqual(eval_res["normal_bypassed"], 2)

        # Test Save & Load
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            triage.save_calibration(tmp_path)
            triage_new = ConformalClinicalTriage()
            loaded_tau = triage_new.load_calibration(tmp_path)
            self.assertEqual(loaded_tau, triage.calibrated_tau)
            self.assertEqual(triage_new.target_fnr, 0.05)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
