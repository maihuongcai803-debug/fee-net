"""
UNet++ (Nested U-Net) for Medical Image Segmentation
Reference:
    Zhou et al., "UNet++: Redesigning Skip Connections to Exploit
    Multiscale Features in Image Segmentation", IEEE Transactions on
    Medical Imaging (IEEE TMI), 2020. https://arxiv.org/abs/1807.10165

Channel Specification:
    Base channels = 32
    Level 0: 32   (x_{0,0}, x_{0,1}, x_{0,2}, x_{0,3}, x_{0,4})
    Level 1: 64   (x_{1,0}, x_{1,1}, x_{1,2}, x_{1,3})
    Level 2: 128  (x_{2,0}, x_{2,1}, x_{2,2})
    Level 3: 256  (x_{3,0}, x_{3,1})
    Level 4: 512  (x_{4,0})
    Output:  1x1 Conv -> 1 class (raw logits)
    Deep supervision: 4 prediction heads at Level 0 (final1, final2, final3, final4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class DoubleConv(nn.Module):
    """
    Standard two-stage convolutional block:
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


class UNetPlusPlus(nn.Module):
    """
    Standard 4-level UNet++ (Nested U-Net) architecture with dense nested skip connections.

    Features:
        - Dense nested skip connections spanning 4 resolution levels (columns j=0..4).
        - Optional deep supervision with 4 prediction heads along level 0.
        - Bilinear interpolation with dynamic spatial alignment for odd/arbitrary input sizes.
        - Trainable parameter count ~9.16M (comparable to Attention U-Net at 7.85M and U-Net at 7.76M).

    Args:
        in_channels (int): Number of input image channels (default: 3).
        num_classes (int): Number of target segmentation classes (default: 1).
        base_channels (int): Number of base feature channels (default: 32).
        deep_supervision (bool): Whether to enable multi-head deep supervision (default: False).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        deep_supervision: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.deep_supervision = deep_supervision

        # Channel layout across 5 resolution stages: [32, 64, 128, 256, 512]
        nb_filter = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16
        ]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # -------------------------------------------------------------
        # Column j = 0: Backbone Encoder
        # -------------------------------------------------------------
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # -------------------------------------------------------------
        # Column j = 1: Dense Nested Skip Pathway 1
        # -------------------------------------------------------------
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        # -------------------------------------------------------------
        # Column j = 2: Dense Nested Skip Pathway 2
        # -------------------------------------------------------------
        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        # -------------------------------------------------------------
        # Column j = 3: Dense Nested Skip Pathway 3
        # -------------------------------------------------------------
        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        # -------------------------------------------------------------
        # Column j = 4: Dense Nested Skip Pathway 4
        # -------------------------------------------------------------
        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        # -------------------------------------------------------------
        # Prediction Heads (Level 0)
        # -------------------------------------------------------------
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _upsample_like(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Upsample src feature map to match target spatial dimensions."""
        if src.shape[2:] != target.shape[2:]:
            return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=True)
        return src

    def forward(
        self,
        x: torch.Tensor,
        return_all: Optional[bool] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for UNet++.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            return_all (bool, optional):
                - If True: returns list of all 4 multi-scale output heads [out1, out2, out3, out4].
                - If False: returns only the final head output out4.
                - If None (default): returns list if deep_supervision=True and model.training=True;
                  otherwise returns single logits tensor out4 (eval mode or deep_supervision=False).

        Returns:
            Single logits tensor of shape (B, num_classes, H, W),
            or list of 4 tensors when deep supervision is active in training mode.
        """
        # Column j = 0 (Backbone Encoder)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Column j = 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample_like(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample_like(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample_like(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample_like(x4_0, x3_0)], dim=1))

        # Column j = 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._upsample_like(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._upsample_like(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._upsample_like(x3_1, x2_0)], dim=1))

        # Column j = 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._upsample_like(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._upsample_like(x2_2, x1_0)], dim=1))

        # Column j = 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample_like(x1_3, x0_0)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)

            # In eval mode without explicit return_all=True, return final head logits (B, num_classes, H, W)
            if not self.training and not return_all:
                return out4

            if return_all is False:
                return out4

            return [out1, out2, out3, out4]
        else:
            return self.final(x0_4)

    def get_num_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Standard alias
NestedUNet = UNetPlusPlus


if __name__ == '__main__':
    print("=" * 65)
    print("Self-test: UNet++ (Nested U-Net, Zhou et al., IEEE TMI 2020)")
    print("=" * 65)

    # 1. Initialize standard UNet++ (deep_supervision=False)
    model = UNetPlusPlus(in_channels=3, num_classes=1, base_channels=32, deep_supervision=False)
    params = model.get_num_parameters()
    print(f"Total Trainable Parameters: {params:,} ({params / 1e6:.2f} M)")
    assert 9_000_000 <= params <= 9_300_000, f"Unexpected parameter count: {params}"

    # 2. Forward pass test in eval mode
    model.eval()
    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape (eval mode): {logits.shape}")
    assert logits.shape == (2, 1, 256, 256), f"Shape mismatch: expected (2, 1, 256, 256), got {logits.shape}"

    # 3. Backward pass test (gradient flow for deep_supervision=False)
    model.train()
    model.zero_grad()
    logits_train = model(dummy_input)
    loss = logits_train.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} did not receive a gradient!"
        assert p.grad.norm().item() > 0, f"Parameter {name} has zero gradient!"
    print(f"Backward pass verified: 100% of parameters in model received non-zero gradients.")

    # 4. Deep supervision test in training mode
    model_ds = UNetPlusPlus(in_channels=3, num_classes=1, base_channels=32, deep_supervision=True)
    params_ds = model_ds.get_num_parameters()
    print(f"Deep Supervision Model Parameters: {params_ds:,} ({params_ds / 1e6:.2f} M)")
    assert 9_000_000 <= params_ds <= 9_300_000, f"Unexpected parameter count: {params_ds}"

    model_ds.train()
    model_ds.zero_grad()
    outputs = model_ds(dummy_input)
    assert isinstance(outputs, list), f"Expected list output for deep supervision in training, got {type(outputs)}"
    assert len(outputs) == 4, f"Expected 4 multi-scale outputs, got {len(outputs)}"
    for idx, out in enumerate(outputs):
        assert out.shape == (2, 1, 256, 256), f"Output {idx} shape mismatch: {out.shape}"

    loss_ds = sum(out.sum() for out in outputs)
    loss_ds.backward()
    for name, p in model_ds.named_parameters():
        assert p.grad is not None, f"Parameter {name} did not receive gradient in deep supervision!"
        assert p.grad.norm().item() > 0, f"Parameter {name} has zero gradient in deep supervision!"
    print("Deep supervision verified in training mode: 4 heads returned and 100% gradients populated.")

    # 5. Deep supervision test in eval mode (should return single tensor)
    model_ds.eval()
    with torch.no_grad():
        out_eval = model_ds(dummy_input)
    assert isinstance(out_eval, torch.Tensor), f"Expected Tensor in eval mode, got {type(out_eval)}"
    assert out_eval.shape == (2, 1, 256, 256), f"Eval shape mismatch: {out_eval.shape}"
    print(f"Deep supervision verified in eval mode: returned single tensor {out_eval.shape}.")

    # 6. Arbitrary / odd spatial dimension test
    dummy_odd = torch.randn(1, 3, 250, 250)
    logits_odd = model(dummy_odd)
    assert logits_odd.shape == (1, 1, 250, 250), f"Odd dimension mismatch: got {logits_odd.shape}"
    print(f"Odd dimension test (250x250) verified: output shape {logits_odd.shape}.")

    print("=" * 65)
    print("ALL UNET++ SELF-TESTS PASSED!")
    print("=" * 65)
