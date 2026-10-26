"""
Attention U-Net for Medical Image Segmentation
Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
MIDL 2018 / arXiv:1804.03999. https://arxiv.org/abs/1804.03999

Channel specification (identical to U-Net baseline for rigorous fair comparison):
    Encoder:    [32, 64, 128, 256]
    Bottleneck: 512
    Decoder:    [256, 128, 64, 32]
    Output:     1x1 Conv -> 1 class (raw logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class DoubleConv(nn.Module):
    """
    Standard double convolution block:
    [Conv3x3 -> BatchNorm -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling stage: MaxPool2d(2) followed by DoubleConv block.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class AttentionGate(nn.Module):
    """
    Spatial Attention Gate (AG) Module (Oktay et al., MIDL 2018).
    
    Selectively suppresses irrelevant background regions while highlighting salient 
    lesion features passed through the skip connection.

    Formulation:
        alpha = Sigmoid( psi( ReLU( W_g(g) + W_x(x) ) ) )
        x_att = x * alpha

    Args:
        in_channels_g: Number of channels in the gating signal (from decoder).
        in_channels_x: Number of channels in the skip connection (from encoder).
        inter_channels: Number of intermediate projection channels. Default: in_channels_x // 2.
    """
    def __init__(self, in_channels_g: int, in_channels_x: int, inter_channels: Optional[int] = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(in_channels_x // 2, 1)

        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            g: Gating signal from decoder stage, shape (B, F_g, H_g, W_g)
            x: Skip feature map from encoder stage, shape (B, F_x, H_x, W_x)
            return_attention: If True, also returns the attention map alpha (B, 1, H_x, W_x)
        Returns:
            x_att: Gated skip connection of shape (B, F_x, H_x, W_x)
            alpha (optional): Attention coefficients
        """
        # Ensure gating signal matches the spatial resolution of skip connection
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        g1 = self.w_g(g)
        x1 = self.w_x(x)
        net = self.relu(g1 + x1)
        alpha = self.psi(net)

        x_att = x * alpha
        if return_attention:
            return x_att, alpha
        return x_att


class AttentionUpBlock(nn.Module):
    """
    Decoder stage with Attention Gate:
    Upsample -> AttentionGate(g, x_skip) -> Concat([g, x_att]) -> DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.bilinear = bilinear
        half_in = in_channels // 2

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.ag = AttentionGate(in_channels_g=in_channels, in_channels_x=half_in, inter_channels=half_in // 2)
            self.conv = DoubleConv(in_channels + half_in, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, half_in, kernel_size=2, stride=2)
            self.ag = AttentionGate(in_channels_g=half_in, in_channels_x=half_in, inter_channels=half_in // 2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, return_attention: bool = False):
        g = self.up(x)

        # Pad if odd spatial resolution difference occurs
        diff_y = skip.size(2) - g.size(2)
        diff_x = skip.size(3) - g.size(3)
        if diff_x != 0 or diff_y != 0:
            g = F.pad(g, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        if return_attention:
            skip_att, alpha = self.ag(g, skip, return_attention=True)
            out = self.conv(torch.cat([g, skip_att], dim=1))
            return out, alpha
        else:
            skip_att = self.ag(g, skip, return_attention=False)
            out = self.conv(torch.cat([g, skip_att], dim=1))
            return out


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture for medical image segmentation.

    Args:
        in_channels (int): Number of input image channels (default: 3).
        num_classes (int): Number of segmentation classes (default: 1).
        bilinear (bool): Whether to use bilinear upsampling instead of ConvTranspose2d.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, bilinear: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder (channels: 32 -> 64 -> 128 -> 256)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # Bottleneck (channels: 512)
        self.down4 = Down(256, 512)

        # Decoder with Attention Gates (channels: 256 -> 128 -> 64 -> 32)
        self.up1 = AttentionUpBlock(512, 256, bilinear=bilinear)
        self.up2 = AttentionUpBlock(256, 128, bilinear=bilinear)
        self.up3 = AttentionUpBlock(128, 64, bilinear=bilinear)
        self.up4 = AttentionUpBlock(64, 32, bilinear=bilinear)

        # Segmentation Head
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_attention_maps: bool = False):
        """
        Forward pass.
        Args:
            x: Input tensor (B, C, H, W)
            return_attention_maps: If True, returns (logits, [alpha1, alpha2, alpha3, alpha4])
        Returns:
            Logits of shape (B, num_classes, H, W)
        """
        # Encoder forward pass
        x1 = self.inc(x)        # (B, 32, H, W)
        x2 = self.down1(x1)     # (B, 64, H/2, W/2)
        x3 = self.down2(x2)     # (B, 128, H/4, W/4)
        x4 = self.down3(x3)     # (B, 256, H/8, W/8)

        # Bottleneck
        x5 = self.down4(x4)     # (B, 512, H/16, W/16)

        # Decoder forward pass with attention gating
        if return_attention_maps:
            d4, a4 = self.up1(x5, x4, return_attention=True)
            d3, a3 = self.up2(d4, x3, return_attention=True)
            d2, a2 = self.up3(d3, x2, return_attention=True)
            d1, a1 = self.up4(d2, x1, return_attention=True)
            logits = self.outc(d1)
            return logits, [a1, a2, a3, a4]
        else:
            d4 = self.up1(x5, x4)
            d3 = self.up2(d4, x3)
            d2 = self.up3(d3, x2)
            d1 = self.up4(d2, x1)
            logits = self.outc(d1)
            return logits

    def get_num_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 60)
    print("Self-test: Attention U-Net (Oktay et al., MIDL 2018)")
    print("=" * 60)

    # Initialize model
    model = AttentionUNet(in_channels=3, num_classes=1, bilinear=False)
    params = model.get_num_parameters()
    print(f"Total Trainable Parameters: {params:,} ({params / 1e6:.2f} M)")

    # Standard forward pass test
    dummy_input = torch.randn(2, 3, 256, 256)
    logits = model(dummy_input)

    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {logits.shape}")

    # Shape assertion
    assert logits.shape == (2, 1, 256, 256), f"Shape mismatch: expected (2, 1, 256, 256), got {logits.shape}"

    # Backward gradient flow test
    loss = logits.sum()
    loss.backward()
    print("Backward pass verified successfully: gradients populated.")

    # Attention maps extraction test
    logits, attn_maps = model(dummy_input, return_attention_maps=True)
    print(f"Extracted {len(attn_maps)} multi-scale Attention Maps:")
    for idx, att in enumerate(attn_maps):
        print(f"  Level {idx + 1} Attention Map Shape: {att.shape}, Min: {att.min().item():.3f}, Max: {att.max().item():.3f}")

    # Odd spatial dimension test
    dummy_odd = torch.randn(1, 3, 250, 250)
    logits_odd = model(dummy_odd)
    assert logits_odd.shape == (1, 1, 250, 250), f"Odd dimension mismatch: got {logits_odd.shape}"
    print(f"Odd dimension test (250x250) output shape: {logits_odd.shape}")

    print("=" * 60)
    print("ALL ATTENTION U-NET SELF-TESTS PASSED!")
    print("=" * 60)
