"""
Standard 4-Level U-Net for Medical Image Segmentation
Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015. https://arxiv.org/abs/1505.04597

Channel specification:
    Encoder:    [32, 64, 128, 256]
    Bottleneck: 512
    Decoder:    [256, 128, 64, 32]
    Output:     1x1 Conv -> 1 class (raw logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """
    Two-stage convolutional block:
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


class Up(nn.Module):
    """
    Upscaling stage: ConvTranspose2d (or Bilinear interpolation) followed by DoubleConv.
    Supports odd spatial dimension alignment via padding.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from previous decoder/bottleneck stage (to be upsampled).
            x2: Skip connection feature map from corresponding encoder stage.
        """
        x1 = self.up(x1)

        # Handle potential spatial dimension discrepancies due to odd input shapes
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard 4-level U-Net architecture for medical segmentation.

    Args:
        in_channels (int): Number of input image channels (e.g. 3 for RGB, 1 for Grayscale).
        num_classes (int): Number of target segmentation classes (default: 1 for binary mask).
        bilinear (bool): Whether to use bilinear interpolation instead of ConvTranspose2d.
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

        # Decoder (channels: 256 -> 128 -> 64 -> 32)
        self.up1 = Up(512, 256, bilinear=bilinear)
        self.up2 = Up(256, 128, bilinear=bilinear)
        self.up3 = Up(128, 64, bilinear=bilinear)
        self.up4 = Up(64, 32, bilinear=bilinear)

        # Final prediction head
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized logits.
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Logits of shape (B, num_classes, H, W)
        """
        # Encoder path with skip connections
        x1 = self.inc(x)        # (B, 32, H, W)
        x2 = self.down1(x1)     # (B, 64, H/2, W/2)
        x3 = self.down2(x2)     # (B, 128, H/4, W/4)
        x4 = self.down3(x3)     # (B, 256, H/8, W/8)

        # Bottleneck
        x5 = self.down4(x4)     # (B, 512, H/16, W/16)

        # Decoder path with skip concatenation
        d4 = self.up1(x5, x4)   # (B, 256, H/8, W/8)
        d3 = self.up2(d4, x3)   # (B, 128, H/4, W/4)
        d2 = self.up3(d3, x2)   # (B, 64, H/2, W/2)
        d1 = self.up4(d2, x1)   # (B, 32, H, W)

        logits = self.outc(d1)  # (B, num_classes, H, W)
        return logits

    def get_num_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 60)
    print("Self-test: Standard 4-Level U-Net")
    print("=" * 60)

    # Initialize model
    model = UNet(in_channels=3, num_classes=1, bilinear=False)
    params = model.get_num_parameters()
    print(f"Total Trainable Parameters: {params:,} ({params / 1e6:.2f} M)")

    # Test forward pass with batch of 2
    dummy_input = torch.randn(2, 3, 256, 256)
    logits = model(dummy_input)

    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {logits.shape}")

    # Shape assertion
    assert logits.shape == (2, 1, 256, 256), f"Shape mismatch: expected (2, 1, 256, 256), got {logits.shape}"

    # Test backward pass to verify gradients
    loss = logits.sum()
    loss.backward()
    print("Backward pass verified successfully: gradients populated.")

    # Test arbitrary spatial dimension handling
    dummy_odd = torch.randn(1, 3, 250, 250)
    logits_odd = model(dummy_odd)
    assert logits_odd.shape == (1, 1, 250, 250), f"Odd dimension mismatch: got {logits_odd.shape}"
    print(f"Odd dimension test (250x250) output shape: {logits_odd.shape}")

    print("=" * 60)
    print("ALL UNET SELF-TESTS PASSED!")
    print("=" * 60)
