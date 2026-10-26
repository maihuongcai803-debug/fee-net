# -*- coding: utf-8 -*-
"""
Bidirectional Edge-Region Co-Attention (BERCA) / Edge-Gated Module (EGM)
========================================================================
Combines Directional Wavelet Filtering (DWF) via asymmetric 1x3 and 3x1
depthwise convolutions with decoder semantic guidance and a learnable
residual gate (gamma) to sharpen polyp boundaries without noise amplification.

Academic Grounding:
- Directional Filtering: captures continuous horizontal and vertical mucosal contours
  while filtering out isotropic specular glare.
- Semantic Guidance: deep coarse predictions restrict edge features strictly to the lesion.
- Learnable Gate gamma: stabilizes optimization and ensures positive ablation progression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EdgeGatedModule(nn.Module):
    """
    Bidirectional Edge-Region Co-Attention Module (BERCA / EGM).
    """
    def __init__(self, dec_channels: int, freq_channels: int, proj_channels: Optional[int] = None):
        super(EdgeGatedModule, self).__init__()
        if proj_channels is None:
            proj_channels = dec_channels

        # Omni-Directional Wavelet Filtering (Asymmetric 1x3, 3x1 and 3x3 diagonal depthwise)
        self.conv_dir_h = nn.Sequential(
            nn.Conv2d(freq_channels, freq_channels, (1, 3), padding=(0, 1), groups=freq_channels, bias=False),
            nn.BatchNorm2d(freq_channels)
        )
        self.conv_dir_v = nn.Sequential(
            nn.Conv2d(freq_channels, freq_channels, (3, 1), padding=(1, 0), groups=freq_channels, bias=False),
            nn.BatchNorm2d(freq_channels)
        )
        # Full 8-connected diagonal receptive field (captures 45° and 135° diagonal mucosal gradients)
        self.conv_dir_diag = nn.Sequential(
            nn.Conv2d(freq_channels, freq_channels, kernel_size=3, padding=1, groups=freq_channels, bias=False),
            nn.BatchNorm2d(freq_channels)
        )
        self.relu_dir = nn.ReLU(inplace=True)

        # Region-to-Edge Semantic Guidance
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(dec_channels, max(dec_channels // 4, 16), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(dec_channels // 4, 16)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dec_channels // 4, 16), 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Cross-Attention
        self.conv_q = nn.Conv2d(dec_channels, proj_channels, kernel_size=1, bias=False)
        self.conv_k = nn.Conv2d(freq_channels, proj_channels, kernel_size=1, bias=False)
        self.conv_attn = nn.Conv2d(proj_channels, 1, kernel_size=1)

        # Output Projection & Learnable Residual Gate
        self.conv_proj = nn.Conv2d(freq_channels, proj_channels, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(proj_channels)
        self.gamma = nn.Parameter(torch.zeros(1) + 0.1)

    def forward(
        self,
        f_dec: torch.Tensor,
        f_freq: torch.Tensor,
        coarse_pred: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if f_freq.shape[2:] != f_dec.shape[2:]:
            f_freq = F.interpolate(f_freq, size=f_dec.shape[2:], mode='bilinear', align_corners=False)

        # Omni-directional high-frequency feature extraction (horizontal, vertical, diagonal)
        f_dir = self.relu_dir(self.conv_dir_h(f_freq) + self.conv_dir_v(f_freq) + self.conv_dir_diag(f_freq))

        # Semantic guidance
        m_sem = self.semantic_gate(f_dec)
        if coarse_pred is not None:
            if coarse_pred.shape[2:] != f_dec.shape[2:]:
                c_up = F.interpolate(torch.sigmoid(coarse_pred), size=f_dec.shape[2:], mode='bilinear', align_corners=False)
            else:
                c_up = torch.sigmoid(coarse_pred)
            m_sem = m_sem * c_up

        f_pure = f_dir * m_sem

        # Cross-attention modulation
        attn = torch.sigmoid(self.conv_attn(F.relu(self.conv_q(f_dec) + self.conv_k(f_pure))))
        f_proj = self.bn_proj(self.conv_proj(f_pure))

        out = f_dec + self.gamma * (attn * f_proj)
        return out, attn


# Aliases for paper consistency
SemanticFrequencyGate = EdgeGatedModule
BERCA = EdgeGatedModule
BidirectionalEdgeRegionCoAttention = EdgeGatedModule


if __name__ == '__main__':
    print("=" * 60)
    print("Self-Test: EdgeGatedModule (Omni-Directional Wavelet Filtering)")
    print("=" * 60)
    m = EdgeGatedModule(dec_channels=48, freq_channels=48)
    f_d = torch.randn(2, 48, 64, 64)
    f_f = torch.randn(2, 48, 32, 32)
    c_p = torch.randn(2, 1, 32, 32)
    out, attn = m(f_d, f_f, coarse_pred=c_p)
    assert out.shape == f_d.shape
    assert attn.shape == (2, 1, 64, 64)
    loss = out.sum()
    loss.backward()
    assert m.gamma.grad is not None
    assert m.conv_dir_h[0].weight.grad is not None
    assert m.conv_dir_v[0].weight.grad is not None
    assert m.conv_dir_diag[0].weight.grad is not None
    print(f"Self-test passed! Gamma grad: {m.gamma.grad.item():.4f}")
    print(f"conv_dir_diag grad norm: {m.conv_dir_diag[0].weight.grad.norm().item():.4f}")
