import unittest
import torch
import torch.nn as nn
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.fee_net import FEENet
from models.modules.egm_module import EdgeGatedModule
from utils.losses import CompoundLoss


class TestFEENetRefinement(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.in_channels = 3
        self.num_classes = 1
        self.h = 256
        self.w = 256
        self.dummy_input = torch.randn(self.batch_size, self.in_channels, self.h, self.w, device=self.device)
        self.dummy_mask = torch.randint(0, 2, (self.batch_size, 1, self.h, self.w), dtype=torch.float32, device=self.device)

    def test_egm_module_gamma(self):
        """Test EdgeGatedModule has learnable gamma initialized to 0.1 and receives gradients."""
        egm = EdgeGatedModule(dec_channels=64, freq_channels=64).to(self.device)
        
        self.assertTrue(hasattr(egm, 'gamma'), "EdgeGatedModule must have attribute gamma")
        self.assertTrue(isinstance(egm.gamma, nn.Parameter), "gamma must be an nn.Parameter")
        self.assertTrue(egm.gamma.requires_grad, "gamma must have requires_grad=True")
        self.assertAlmostEqual(egm.gamma.item(), 0.1, places=5, msg="Initial gamma must be 0.1")
        
        f_dec = torch.randn(2, 64, 32, 32, device=self.device, requires_grad=True)
        f_freq = torch.randn(2, 64, 32, 32, device=self.device, requires_grad=True)
        out, attn = egm(f_dec, f_freq)
        
        self.assertEqual(out.shape, f_dec.shape)
        self.assertEqual(attn.shape, (2, 1, 32, 32))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(egm.gamma.grad, "gamma must receive non-None gradient")
        self.assertFalse(torch.isnan(egm.gamma.grad).any(), "gamma gradient must not contain NaNs")
        self.assertNotEqual(egm.gamma.grad.item(), 0.0, "gamma gradient should be non-zero")

    def test_feenet_parameter_count(self):
        """Test FEENet parameter count (~8.52M with base channels 48 and 512 dilated bottleneck)."""
        model = FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=True, base_channels=48).to(self.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Ensure ~8.52M parameters matching best_model_fold0.pth checkpoint
        self.assertGreater(total_params, 8_400_000)
        self.assertLess(total_params, 8_600_000)
        self.assertAlmostEqual(model.egm1.gamma.item(), 0.1, places=5)
        self.assertAlmostEqual(model.egm2.gamma.item(), 0.1, places=5)

    def test_feenet_training_mode(self):
        """Test FEENet in training mode returns (seg_logits, edge_logits) by default and 3-tuple with return_aux."""
        model = FEENet(in_channels=3, num_classes=1).to(self.device)
        model.train()
        
        out = model(self.dummy_input)
        self.assertIsInstance(out, tuple, "Training mode forward must return a tuple")
        self.assertEqual(len(out), 2, "Training mode tuple must contain exactly 2 elements (seg, edge)")

        out_aux = model(self.dummy_input, return_aux=True)
        self.assertEqual(len(out_aux), 3, "Training mode with return_aux=True must return (seg, edge, aux_preds)")
        
        seg, edge = out
        self.assertEqual(seg.shape, (self.batch_size, self.num_classes, self.h, self.w))
        self.assertEqual(edge.shape, (self.batch_size, self.num_classes, self.h, self.w))
        
        # Test backward pass with CompoundLoss
        criterion = CompoundLoss().to(self.device)
        loss, _ = criterion(seg, edge, self.dummy_mask)
        loss.backward()
        
        self.assertIsNotNone(model.egm1.gamma.grad, "EGM1 gamma must receive gradient in FEENet backward")
        self.assertIsNotNone(model.egm2.gamma.grad, "EGM2 gamma must receive gradient in FEENet backward")
        self.assertIsNotNone(model.egm1.conv_dir_h[0].weight.grad, "EGM1 DWF 1x3 conv must receive gradient")
        self.assertIsNotNone(model.egm1.conv_dir_v[0].weight.grad, "EGM1 DWF 3x1 conv must receive gradient")
        self.assertIsNotNone(model.egm1.conv_dir_diag[0].weight.grad, "EGM1 DWF 3x3 diagonal conv must receive gradient")
        self.assertIsNotNone(model.wf1_attn.fc[0].weight.grad, "wf1_attn spectral attention must receive gradient")
        self.assertIsNotNone(model.egm1.semantic_gate[0].weight.grad, "EGM1 semantic gate must receive gradient")
        self.assertFalse(torch.isnan(model.egm1.gamma.grad).any())
        self.assertFalse(torch.isnan(model.egm2.gamma.grad).any())

    def test_feenet_eval_mode_default(self):
        """Test FEENet in eval mode (return_edge=False) returns single seg_logits Tensor."""
        model = FEENet(in_channels=3, num_classes=1).to(self.device)
        model.eval()
        
        with torch.no_grad():
            out = model(self.dummy_input)
            self.assertTrue(torch.is_tensor(out), "Eval mode with return_edge=False must return a Tensor directly")
            self.assertEqual(out.shape, (self.batch_size, self.num_classes, self.h, self.w))

    def test_feenet_eval_mode_return_edge(self):
        """Test FEENet in eval mode with return_edge=True returns (seg_logits, edge_logits)."""
        model = FEENet(in_channels=3, num_classes=1).to(self.device)
        model.eval()
        
        with torch.no_grad():
            # Keyword argument
            out_kw = model(self.dummy_input, return_edge=True)
            self.assertIsInstance(out_kw, tuple)
            self.assertEqual(len(out_kw), 2)
            seg, edge = out_kw
            self.assertEqual(seg.shape, (self.batch_size, self.num_classes, self.h, self.w))
            self.assertEqual(edge.shape, (self.batch_size, self.num_classes, self.h, self.w))
            
            # Positional boolean compatibility
            out_pos = model(self.dummy_input, True)
            self.assertIsInstance(out_pos, tuple)
            self.assertEqual(len(out_pos), 2)
            
            # Positional boolean False
            out_false = model(self.dummy_input, False)
            self.assertTrue(torch.is_tensor(out_false))

    def test_feenet_ablation_compatibility(self):
        """Test FEENet ablations (w/o EGM, w/o DWT) run smoothly in both train and eval modes."""
        for use_dwt, use_egm in [(True, False), (False, True), (False, False)]:
            model = FEENet(in_channels=3, num_classes=1, use_dwt=use_dwt, use_egm=use_egm).to(self.device)
            
            # Train
            model.train()
            out_train = model(self.dummy_input)
            self.assertIsInstance(out_train, tuple)
            self.assertEqual(len(out_train), 2)
            loss = out_train[0].sum() + out_train[1].sum()
            loss.backward()
            
            # Eval
            model.eval()
            with torch.no_grad():
                out_eval = model(self.dummy_input)
                self.assertTrue(torch.is_tensor(out_eval))
                
                out_eval_edge = model(self.dummy_input, return_edge=True)
                self.assertIsInstance(out_eval_edge, tuple)


if __name__ == '__main__':
    unittest.main(verbosity=2)
