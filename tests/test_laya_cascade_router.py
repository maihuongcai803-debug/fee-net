# -*- coding: utf-8 -*-
"""
Unit tests for LayaCascadeRouter (System 1 + System 2 3-tier cascade routing engine).
Tests vectorized masked batching, shallow frequency boundary probe, and edge cases.
"""

import unittest
import torch
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from laya.modules.laya_cascade_router import LayaCascadeRouter, create_synthetic_test_batch


class TestLayaCascadeRouter(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.router = LayaCascadeRouter(
            triage_ckpt="checkpoints/laya_triage_best.pth",
            feenet_ckpt="checkpoints/fee_net_best.pth",
            feenet_base_channels=48,
            boundary_noise_threshold=0.60,
            verbose=False
        ).to(self.device)

    def test_initialization_and_complexity(self):
        """Test instantiation, parameter counts and complexity constants."""
        self.assertTrue(hasattr(self.router, "triage_net"))
        self.assertTrue(hasattr(self.router, "fee_net"))
        self.assertTrue(hasattr(self.router, "feenet"))
        self.assertEqual(self.router.fee_shallow_gflops, 1.8518)
        self.assertEqual(self.router.fee_full_gflops, 12.4427)

    def test_3tier_cascade_routing(self):
        """Test 3-tier routing produces expected execution tiers and structured dictionary."""
        batch, labels = create_synthetic_test_batch()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            output = self.router(batch, tau=0.5414, ambiguous_margin=0.15)
            
        # Dictionary structure verification
        expected_keys = ["pred_masks", "edge_masks", "triage_decisions", "execution_tiers", "cohort_gflops", "avg_latency_ms"]
        for key in expected_keys:
            self.assertIn(key, output, f"Missing key '{key}' in cascade router output")
            
        # Tensor shapes and sizes
        self.assertEqual(output["pred_masks"].shape, (3, 1, 256, 256))
        self.assertEqual(output["edge_masks"].shape, (3, 1, 256, 256))
        self.assertEqual(len(output["triage_decisions"]), 3)
        self.assertEqual(len(output["execution_tiers"]), 3)
        
        # Verify 3 distinct tiers
        self.assertEqual(output["execution_tiers"][0], "TIER1_BYPASS")
        self.assertEqual(output["execution_tiers"][1], "TIER2_LIGHT_CLEAR")
        self.assertEqual(output["execution_tiers"][2], "TIER3_FULL_SEG")
        
        # Verify zero mask for bypass and light clear
        self.assertEqual(output["pred_masks"][0].sum().item(), 0.0)
        self.assertEqual(output["pred_masks"][1].sum().item(), 0.0)
        
        # Verify FLOP savings
        monolithic_gflops = 3 * self.router.fee_full_gflops
        self.assertLess(output["cohort_gflops"], monolithic_gflops)

    def test_shallow_boundary_probe(self):
        """Test shallow frequency boundary probe on single sample and batch."""
        batch, _ = create_synthetic_test_batch()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            # Batched call
            b_resp, max_resp = self.router.extract_shallow_boundary(batch)
            self.assertEqual(b_resp.shape, (3, 1, 256, 256))
            self.assertTrue(torch.is_tensor(max_resp))
            self.assertEqual(max_resp.shape, (3,))
            self.assertTrue((b_resp >= 0.0).all() and (b_resp <= 1.0).all())
            
            # Single sample call
            b_resp_s, max_resp_s = self.router.extract_shallow_boundary(batch[1:2])
            self.assertEqual(b_resp_s.shape, (1, 1, 256, 256))
            self.assertIsInstance(max_resp_s, float)
            self.assertAlmostEqual(max_resp[1].item(), max_resp_s, places=5)

    def test_vectorized_batch_equivalence(self):
        """Test that vectorized batched forward pass matches individual evaluations."""
        batch, _ = create_synthetic_test_batch()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            batched_out = self.router(batch, tau=0.5414, ambiguous_margin=0.15)
            
            single_out_0 = self.router(batch[0:1], tau=0.5414, ambiguous_margin=0.15)
            single_out_1 = self.router(batch[1:2], tau=0.5414, ambiguous_margin=0.15)
            single_out_2 = self.router(batch[2:3], tau=0.5414, ambiguous_margin=0.15)
            
        self.assertEqual(batched_out["execution_tiers"][0], single_out_0["execution_tiers"][0])
        self.assertEqual(batched_out["execution_tiers"][1], single_out_1["execution_tiers"][0])
        self.assertEqual(batched_out["execution_tiers"][2], single_out_2["execution_tiers"][0])
        
        # Segmented mask for lesion sample should match
        diff = torch.abs(batched_out["pred_masks"][2] - single_out_2["pred_masks"][0]).max().item()
        self.assertLess(diff, 1e-4, f"Vectorized batch output differs from single sample by {diff}")

    def test_tier2_promotion_to_tier3(self):
        """Test that ambiguous samples with high boundary response promote to Tier 3."""
        batch, _ = create_synthetic_test_batch()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            out = self.router(batch, tau=0.5414, ambiguous_margin=0.15, boundary_noise_threshold=0.10)
            
        self.assertEqual(out["execution_tiers"][0], "TIER1_BYPASS")
        self.assertEqual(out["execution_tiers"][1], "TIER3_FULL_SEG")  # Promoted!
        self.assertEqual(out["execution_tiers"][2], "TIER3_FULL_SEG")

    def test_graceful_checkpoint_fallback(self):
        """Test router initializes and forwards correctly even with nonexistent checkpoints."""
        fallback_router = LayaCascadeRouter(
            triage_ckpt="invalid_path.pth",
            feenet_ckpt="invalid_path.pth",
            verbose=False
        ).to(self.device)
        
        x = torch.randn(2, 3, 256, 256, device=self.device)
        with torch.no_grad():
            out = fallback_router(x)
            
        self.assertEqual(out["pred_masks"].shape, (2, 1, 256, 256))
        self.assertEqual(len(out["execution_tiers"]), 2)
        self.assertGreater(out["cohort_gflops"], 0.0)


if __name__ == '__main__':
    unittest.main()
