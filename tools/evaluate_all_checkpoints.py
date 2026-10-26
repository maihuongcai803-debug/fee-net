# -*- coding: utf-8 -*-
"""
Evaluate All Models on the 100 Blind Held-Out Test Lesions:
1. Kaggle-trained UNet (results (1)/unet_best.pth)
2. Kaggle-trained Attention UNet (results (1)/attention_unet_best.pth)
3. Kaggle-trained UNet++ (results (1)/unet_plus_plus_best.pth)
4. Kaggle-trained FEENetEarlyExit (results (1)/feenet_early_exit_best.pth)
5. Fully Converged FEE-Net Backbone (checkpoints/best_model_fold0.pth)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from scipy.spatial.distance import directed_hausdorff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions Matching Kaggle Notebook ---
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetBaseline(nn.Module):
    def __init__(self, in_c=3, num_classes=1, base_c=32):
        super().__init__()
        c = base_c
        self.inc = DoubleConv(in_c, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*2, c*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*4, c*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*8, 512))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(512, c*8, 1, bias=False))
        self.dec1 = DoubleConv(c*16, c*8)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c*8, c*4, 1, bias=False))
        self.dec2 = DoubleConv(c*8, c*4)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c*4, c*2, 1, bias=False))
        self.dec3 = DoubleConv(c*4, c*2)
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c*2, c, 1, bias=False))
        self.dec4 = DoubleConv(c*2, c)
        self.outc = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d1 = self.dec1(torch.cat([self.up1(x5), x4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), x3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), x2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), x1], dim=1))
        return self.outc(d4)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

class AttentionUNetBaseline(nn.Module):
    def __init__(self, in_c=3, num_classes=1, base_c=32):
        super().__init__()
        c = base_c
        self.inc = DoubleConv(in_c, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*2, c*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*4, c*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*8, 512))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(512, c*8, 1, bias=False))
        self.ag1 = AttentionGate(c*8, c*8, c*4)
        self.dec1 = DoubleConv(c*16, c*8)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c*8, c*4, 1, bias=False))
        self.ag2 = AttentionGate(c*4, c*4, c*2)
        self.dec2 = DoubleConv(c*8, c*4)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c*4, c*2, 1, bias=False))
        self.ag3 = AttentionGate(c*2, c*2, c)
        self.dec3 = DoubleConv(c*4, c*2)
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c*2, c, 1, bias=False))
        self.ag4 = AttentionGate(c, c, max(c//2, 8))
        self.dec4 = DoubleConv(c*2, c)
        self.outc = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.up1(x5)
        d1 = self.dec1(torch.cat([u1, self.ag1(u1, x4)], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, self.ag2(u2, x3)], dim=1))
        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([u3, self.ag3(u3, x2)], dim=1))
        u4 = self.up4(d3)
        d4 = self.dec4(torch.cat([u4, self.ag4(u4, x1)], dim=1))
        return self.outc(d4)

class UNetPlusPlusBaseline(nn.Module):
    def __init__(self, in_c=3, num_classes=1, base_c=32):
        super().__init__()
        c = [base_c, base_c*2, base_c*4, base_c*8, 512]
        self.conv0_0 = DoubleConv(in_c, c[0])
        self.conv1_0 = DoubleConv(c[0], c[1])
        self.conv2_0 = DoubleConv(c[1], c[2])
        self.conv3_0 = DoubleConv(c[2], c[3])
        self.conv4_0 = DoubleConv(c[3], c[4])
        self.conv0_1 = DoubleConv(c[0] + c[1], c[0])
        self.conv1_1 = DoubleConv(c[1] + c[2], c[1])
        self.conv2_1 = DoubleConv(c[2] + c[3], c[2])
        self.conv3_1 = DoubleConv(c[3] + c[4], c[3])
        self.conv0_2 = DoubleConv(c[0]*2 + c[1], c[0])
        self.conv1_2 = DoubleConv(c[1]*2 + c[2], c[1])
        self.conv2_2 = DoubleConv(c[2]*2 + c[3], c[2])
        self.conv0_3 = DoubleConv(c[0]*3 + c[1], c[0])
        self.conv1_3 = DoubleConv(c[1]*3 + c[2], c[1])
        self.conv0_4 = DoubleConv(c[0]*4 + c[1], c[0])
        self.final = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, size=x3_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], 1))
        return self.final(x0_4)

class DWT_2D(nn.Module):
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        return x1+x2+x3+x4, -x1+x2-x3+x4, -x1-x2+x3+x4, x1-x2-x3+x4

class EdgeGatedModule(nn.Module):
    def __init__(self, dec_c, freq_c):
        super().__init__()
        self.conv_dir_h = nn.Sequential(nn.Conv2d(freq_c, freq_c, (1, 3), padding=(0, 1), groups=freq_c, bias=False), nn.BatchNorm2d(freq_c))
        self.conv_dir_v = nn.Sequential(nn.Conv2d(freq_c, freq_c, (3, 1), padding=(1, 0), groups=freq_c, bias=False), nn.BatchNorm2d(freq_c))
        self.relu_dir = nn.ReLU(inplace=True)
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(dec_c, max(dec_c // 4, 16), 1, bias=False), nn.BatchNorm2d(max(dec_c // 4, 16)), nn.ReLU(inplace=True),
            nn.Conv2d(max(dec_c // 4, 16), 1, 1), nn.Sigmoid()
        )
        self.conv_q = nn.Conv2d(dec_c, dec_c, 1, bias=False)
        self.conv_k = nn.Conv2d(freq_c, dec_c, 1, bias=False)
        self.conv_attn = nn.Conv2d(dec_c, 1, 1)
        self.conv_proj = nn.Conv2d(freq_c, dec_c, 1, bias=False)
        self.bn_proj = nn.BatchNorm2d(dec_c)
        self.gamma = nn.Parameter(torch.zeros(1) + 0.1)

    def forward(self, f_dec, f_freq, coarse_pred=None):
        if f_freq.shape[2:] != f_dec.shape[2:]:
            f_freq = F.interpolate(f_freq, size=f_dec.shape[2:], mode='bilinear', align_corners=False)
        f_dir = self.relu_dir(self.conv_dir_h(f_freq) + self.conv_dir_v(f_freq))
        m_sem = self.semantic_gate(f_dec)
        if coarse_pred is not None:
            c_up = F.interpolate(torch.sigmoid(coarse_pred), size=f_dec.shape[2:], mode='bilinear', align_corners=False)
            m_sem = m_sem * c_up
        f_pure = f_dir * m_sem
        attn = torch.sigmoid(self.conv_attn(F.relu(self.conv_q(f_dec) + self.conv_k(f_pure))))
        f_proj = self.bn_proj(self.conv_proj(f_pure))
        return f_dec + self.gamma * (attn * f_proj), attn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c)) if (stride!=1 or in_c!=out_c) else None
    def forward(self, x):
        res = self.down(x) if self.down else x
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + res)

class EarlyExitTriageHead(nn.Module):
    def __init__(self, in_channels=48, hidden_dim=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, e1, hf_1):
        feat = torch.cat([self.pool(e1).flatten(1), self.pool(hf_1).flatten(1)], dim=1)
        return self.mlp(feat)

class FEENetEarlyExitKaggle(nn.Module):
    def __init__(self, in_c=3, num_classes=1, base_c=48, detach_triage=True):
        super().__init__()
        self.detach_triage = detach_triage
        c = base_c
        self.stem = nn.Sequential(nn.Conv2d(in_c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.layer1 = ResidualBlock(c, c, stride=2)
        self.dwt = DWT_2D()
        self.wf1 = nn.Conv2d(c * 3, c, 1, bias=False)
        self.wf2 = nn.Conv2d(c * 3, c, 1, bias=False)
        self.triage_head = EarlyExitTriageHead(in_channels=c)
        
        self.layer2 = ResidualBlock(c, c*2, stride=2)
        self.layer3 = ResidualBlock(c*2, c*4, stride=2)
        self.layer4 = ResidualBlock(c*4, c*8, stride=2)
        self.bot = nn.Sequential(
            nn.Conv2d(c*8, 512, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(512, c*4, 1, bias=False), nn.BatchNorm2d(c*4), nn.ReLU(inplace=True))
        self.dec4 = ResidualBlock(c*8, c*4)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c*4, c*2, 1, bias=False), nn.BatchNorm2d(c*2), nn.ReLU(inplace=True))
        self.dec3 = ResidualBlock(c*4, c*2)
        
        self.coarse_head = nn.Conv2d(c*2, num_classes, 1)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c*2, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.egm2 = EdgeGatedModule(c, c)
        self.dec2 = ResidualBlock(c*2, c)
        
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.egm1 = EdgeGatedModule(c, c)
        self.dec1 = ResidualBlock(c*2, c)
        
        self.head_seg = nn.Conv2d(c, num_classes, 1)
        self.head_edge = nn.Conv2d(c, num_classes, 1)
        self.deep_sup4 = nn.Conv2d(c*4, num_classes, 1)
        self.deep_sup3 = nn.Conv2d(c*2, num_classes, 1)
        self.deep_sup2 = nn.Conv2d(c, num_classes, 1)

    def forward(self, x, tau=0.5, force_full=True, return_edge=False):
        b_sz, _, h_in, w_in = x.shape
        x_stem = self.stem(x)
        e1 = self.layer1(x_stem)
        _, lh_s, hl_s, hh_s = self.dwt(x_stem)
        hf_1 = self.wf1(torch.cat([lh_s, hl_s, hh_s], dim=1))
        
        feat_e1 = e1.detach() if (self.detach_triage and self.training) else e1
        feat_hf1 = hf_1.detach() if (self.detach_triage and self.training) else hf_1
        triage_logits = self.triage_head(feat_e1, feat_hf1)
        risk_scores = F.softmax(triage_logits, dim=-1)[:, 1]
        
        exit_mask = (risk_scores < tau) if not force_full else torch.zeros(b_sz, dtype=torch.bool, device=x.device)
        pred_masks = torch.zeros((b_sz, 1, h_in, w_in), device=x.device, dtype=x.dtype)
        idx_full = torch.where(~exit_mask)[0]
        
        if len(idx_full) > 0:
            sub_e1 = e1[idx_full]
            sub_x_stem = x_stem[idx_full]
            sub_hf_1 = hf_1[idx_full]
            _, lh_1, hl_1, hh_1 = self.dwt(sub_e1)
            sub_hf_2 = self.wf2(torch.cat([lh_1, hl_1, hh_1], dim=1))
            sub_e2 = self.layer2(sub_e1)
            sub_e3 = self.layer3(sub_e2)
            sub_e4 = self.layer4(sub_e3)
            sub_b = self.bot(sub_e4)
            sub_d4 = self.dec4(torch.cat([self.up4(sub_b), sub_e3], dim=1))
            sub_d3 = self.dec3(torch.cat([self.up3(sub_d4), sub_e2], dim=1))
            sub_c = self.coarse_head(sub_d3)
            sub_d2 = self.up2(sub_d3)
            sub_d2, _ = self.egm2(sub_d2, sub_hf_2, coarse_pred=sub_c)
            sub_d2 = self.dec2(torch.cat([sub_d2, sub_e1], dim=1))
            sub_d1 = self.up1(sub_d2)
            sub_hf1_up = F.interpolate(sub_hf_1, size=sub_d1.shape[2:], mode='bilinear', align_corners=False)
            sub_d1, _ = self.egm1(sub_d1, sub_hf1_up, coarse_pred=sub_c)
            sub_d1 = self.dec1(torch.cat([sub_d1, sub_x_stem], dim=1))
            sub_seg = self.head_seg(sub_d1)
            pred_masks[idx_full] = torch.sigmoid(sub_seg)
            
        return pred_masks

from models.fee_net import FEENet

def compute_hd95(pred_mask, gt_mask):
    p_b = (pred_mask > 0.5).astype(np.uint8)
    g_b = (gt_mask > 0.5).astype(np.uint8)
    if p_b.sum() == 0 or g_b.sum() == 0:
        return 100.0
    p_pts = np.argwhere(cv2.Canny(p_b * 255, 0, 1) > 0)
    g_pts = np.argwhere(cv2.Canny(g_b * 255, 0, 1) > 0)
    if len(p_pts) == 0 or len(g_pts) == 0:
        return 100.0
    d1 = directed_hausdorff(p_pts, g_pts)[0]
    d2 = directed_hausdorff(g_pts, p_pts)[0]
    return float(max(d1, d2))

test_img_paths = sorted(glob.glob("results (1)/data/kvasir_triage_v2/test/images/lesion_*.png"))
if len(test_img_paths) == 0:
    test_img_paths = sorted(glob.glob("data/kvasir_triage_v2/test/images/lesion_*.png"))
print(f"Total blind test lesion images: {len(test_img_paths)}")

def eval_model(model, name, is_already_sigmoid=False):
    model.eval()
    model.to(device)
    dices, ious, hd95s = [], [], []
    
    with torch.no_grad():
        for p in test_img_paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            
            mp = p.replace("images", "masks")
            msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            msk = cv2.resize(msk, (256, 256), interpolation=cv2.INTER_NEAREST)
            g_bin = (msk > 127).astype(np.float32)
            
            t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            t_img = t_img.unsqueeze(0).to(device)
            
            out = model(t_img)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if is_already_sigmoid:
                pred = out[0, 0].cpu().numpy()
            else:
                pred = torch.sigmoid(out)[0, 0].cpu().numpy()
            p_bin = (pred > 0.5).astype(np.float32)
            
            inter = (p_bin * g_bin).sum()
            union = p_bin.sum() + g_bin.sum() - inter
            d = (2.0 * inter + 1e-5) / (p_bin.sum() + g_bin.sum() + 1e-5)
            iou = (inter + 1e-5) / (union + 1e-5)
            hd = compute_hd95(p_bin, g_bin)
            
            dices.append(d)
            ious.append(iou)
            hd95s.append(hd)
            
    mean_d = np.mean(dices) * 100
    std_d  = np.std(dices) * 100
    mean_i = np.mean(ious) * 100
    mean_h = np.mean(hd95s)
    print(f"| {name:<35} | {mean_d:6.2f} ± {std_d:5.2f}% | {mean_i:6.2f}% | {mean_h:6.2f} |")
    return mean_d, mean_i, mean_h

print("=" * 80)
print(f"| {'Architecture / Checkpoint':<35} | {'Dice (%)':<16} | {'IoU (%)':<10} | {'HD95':<7} |")
print("=" * 80)

# 1. Kaggle Run from Scratch (40 Epochs)
m_unet = UNetBaseline(base_c=32)
m_unet.load_state_dict(torch.load("results (1)/unet_best.pth", map_location="cpu"))
eval_model(m_unet, "UNet (Kaggle 40 Epochs)")

m_att = AttentionUNetBaseline(base_c=32)
m_att.load_state_dict(torch.load("results (1)/attention_unet_best.pth", map_location="cpu"))
eval_model(m_att, "Attention UNet (Kaggle 40 Epochs)")

m_upp = UNetPlusPlusBaseline(base_c=32)
m_upp.load_state_dict(torch.load("results (1)/unet_plus_plus_best.pth", map_location="cpu"))
eval_model(m_upp, "UNet++ (Kaggle 40 Epochs)")

m_ee = FEENetEarlyExitKaggle(base_c=48)
m_ee.load_state_dict(torch.load("results (1)/feenet_early_exit_best.pth", map_location="cpu"), strict=False)
eval_model(m_ee, "FEENet-EE (Kaggle 40 Epochs)", is_already_sigmoid=True)

# 2. Fully Converged FEE-Net (best_model_fold0.pth)
m_feenet_best = FEENet(base_channels=48)
m_feenet_best.load_state_dict(torch.load("checkpoints/best_model_fold0.pth", map_location="cpu"))
eval_model(m_feenet_best, "FEE-Net (Converged Backbone)")

print("=" * 80)
