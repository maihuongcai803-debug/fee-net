import unittest
import torch
import torch.nn as nn
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.fee_net_early_exit import FEENetEarlyExit, EarlyExitTriageHead


class TestFEENetEarlyExit(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.in_channels = 3
        self.num_classes = 1
        self.h = 256
        self.w = 256
        self.dummy_input = torch.randn(self.batch_size, self.in_channels, self.h, self.w, device=self.device)

    def test_parameter_count_and_head(self):
        """Test parameter count (~8.52M) and submodules."""
        model = FEENetEarlyExit(base_channels=48).to(self.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(total_params, 8_400_000)
        self.assertLess(total_params, 8_600_000)
        self.assertTrue(hasattr(model, 'triage_head'))
        self.assertTrue(isinstance(model.triage_head, EarlyExitTriageHead))

    def test_training_mode_multi_task(self):
        """Test training mode returns 5-tuple with return_aux=True and gradients flow to all heads."""
        model = FEENetEarlyExit(base_channels=48).to(self.device)
        model.train()
        
        seg, edge, triage, quality, aux = model(self.dummy_input, return_aux=True)
        self.assertEqual(seg.shape, (self.batch_size, self.num_classes, self.h, self.w))
        self.assertEqual(edge.shape, (self.batch_size, self.num_classes, self.h, self.w))
        self.assertEqual(triage.shape, (self.batch_size, 2))
        self.assertEqual(quality.shape, (self.batch_size, 2))
        self.assertEqual(len(aux), 3)

        loss = seg.sum() + edge.sum() + triage.sum() + quality.sum() + sum(a.sum() for a in aux)
        loss.backward()

        self.assertIsNotNone(model.stem[0].weight.grad)
        self.assertIsNotNone(model.triage_head.mlp[0].weight.grad)
        self.assertIsNotNone(model.deep_sup3.weight.grad)
        bot_conv = model.bot[0] if isinstance(model.bot, nn.Sequential) else model.bot.conv1
        self.assertIsNotNone(bot_conv.weight.grad)
        self.assertFalse(torch.isnan(model.triage_head.mlp[0].weight.grad).any())

    def test_eval_mode_early_exit(self):
        """Test eval mode properly routes between early exit and full segmentation."""
        model = FEENetEarlyExit(base_channels=48).to(self.device)
        model.eval()

        with torch.no_grad():
            # Force early exit with high tau (1.0)
            masks_exit, edges_exit, risks_exit, tiers_exit = model(self.dummy_input, tau=1.0, return_edge=True)
            self.assertEqual(masks_exit.shape, (self.batch_size, 1, self.h, self.w))
            self.assertTrue(all(t == "TIER1_EARLY_EXIT" for t in tiers_exit))
            self.assertEqual(masks_exit.sum().item(), 0.0)

            # Force full segmentation with low tau (0.0)
            masks_full, edges_full, risks_full, tiers_full = model(self.dummy_input, tau=0.0, return_edge=True)
            self.assertEqual(masks_full.shape, (self.batch_size, 1, self.h, self.w))
            self.assertTrue(all(t == "TIER2_FULL_SEG" for t in tiers_full))

    def test_checkpoint_backward_compatibility(self):
        """Test loading backbone weights from fee_net_v2_fold0_best.pth."""
        model = FEENetEarlyExit(base_channels=48).to(self.device)
        ckpt_path = "checkpoints/fee_net_v2_fold0_best.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            sd = ckpt.get("model_state_dict", ckpt)
            # Filter compatible keys
            model_sd = model.state_dict()
            compatible = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
            res = model.load_state_dict(compatible, strict=False)
            self.assertEqual(len(res.unexpected_keys), 0)

    def test_batch_size_one_layernorm(self):
        """Test LayerNorm allows batch size B=1 in training and eval without crashing."""
        model = FEENetEarlyExit(base_channels=48).to(self.device)
        single_input = torch.randn(1, self.in_channels, self.h, self.w, device=self.device)
        
        # Train mode B=1
        model.train()
        seg, edge, triage, quality, aux = model(single_input, return_aux=True)
        self.assertEqual(triage.shape, (1, 2))
        loss = triage.sum()
        loss.backward()
        self.assertIsNotNone(model.triage_head.mlp[0].weight.grad)

        # Eval mode B=1
        model.eval()
        with torch.no_grad():
            out_mask = model(single_input, tau=0.5)
            self.assertEqual(out_mask.shape, (1, 1, self.h, self.w))

    def test_gradient_decoupling(self):
        """Test stop-gradient feature probing isolates backbone when detach_triage_feature=True."""
        # 1. With decoupling enabled (default)
        model_decoupled = FEENetEarlyExit(base_channels=48, detach_triage_feature=True).to(self.device)
        model_decoupled.train()
        seg, edge, triage, quality, aux = model_decoupled(self.dummy_input, return_aux=True)
        
        # Backward ONLY triage loss
        triage_loss = triage.sum()
        triage_loss.backward()

        # Triage head must have gradients
        self.assertIsNotNone(model_decoupled.triage_head.mlp[0].weight.grad)
        # Shared stem & layer1 MUST NOT have gradients from triage loss
        self.assertIsNone(model_decoupled.stem[0].weight.grad)
        self.assertIsNone(model_decoupled.layer1.conv1.weight.grad)

        # Now backward seg loss -> backbone must receive gradients
        seg_loss = seg.sum()
        seg_loss.backward()
        self.assertIsNotNone(model_decoupled.stem[0].weight.grad)
        self.assertIsNotNone(model_decoupled.layer1.conv1.weight.grad)

        # 2. With decoupling disabled -> triage gradients flow to backbone
        model_coupled = FEENetEarlyExit(base_channels=48, detach_triage_feature=False).to(self.device)
        model_coupled.train()
        seg_c, edge_c, triage_c, quality_c, aux_c = model_coupled(self.dummy_input, return_aux=True)
        triage_c.sum().backward()
        self.assertIsNotNone(model_coupled.stem[0].weight.grad)
        self.assertIsNotNone(model_coupled.layer1.conv1.weight.grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
