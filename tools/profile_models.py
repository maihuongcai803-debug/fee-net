# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.unet_plus_plus import UNetPlusPlus
from models.fee_net import FEENet
from utils.metrics import compute_model_complexity

models = [
    ("UNet++ (Nested SOTA)", UNetPlusPlus(3, 1)),
    ("Attention U-Net", AttentionUNet(3, 1)),
    ("U-Net (Baseline)", UNet(3, 1)),
    ("FEE-Net v2.1 (Proposed SOTA)", FEENet(3, 1, base_channels=48))
]

print("="*65)
print(f"{'Model Architecture':<30} | {'Params (M)':<12} | {'FLOPs (G)':<12}")
print("-"*65)
for name, m in models:
    c = compute_model_complexity(m, input_size=(1, 3, 256, 256), device="cpu")
    print(f"{name:<30} | {c['params_m']:8.2f} M   | {c['flops_g']:8.2f} G")
print("="*65)
