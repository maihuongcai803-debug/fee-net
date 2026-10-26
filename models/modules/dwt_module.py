import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT_2D(nn.Module):
    def __init__(self):
        super(DWT_2D, self).__init__()
        # Haar wavelet basis
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        # (4, 1, 2, 2)
        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('weight', weight)

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.weight.repeat(C, 1, 1, 1).to(dtype=x.dtype, device=x.device) # (4C, 1, 2, 2)
        out = F.conv2d(x, weight, stride=2, groups=C) # (B, 4C, H/2, W/2)
        
        out = out.view(B, C, 4, H // 2, W // 2)
        LL = out[:, :, 0, :, :]
        LH = out[:, :, 1, :, :]
        HL = out[:, :, 2, :, :]
        HH = out[:, :, 3, :, :]
        
        return LL, LH, HL, HH

class I_DWT_2D(nn.Module):
    """
    Inverse 2D Haar Discrete Wavelet Transform (I-DWT).
    # Auxiliary / Inverse transform (retained for mathematical completeness)
    Reconstructs spatial domain representation from LL, LH, HL, HH subbands.
    """
    def __init__(self):
        super(I_DWT_2D, self).__init__()
        # Auxiliary / Inverse transform (retained for mathematical completeness)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('weight', weight)

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        x = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4*C, H, W)
        weight = self.weight.repeat(C, 1, 1, 1).to(dtype=x.dtype, device=x.device)
        out = F.conv_transpose2d(x, weight, stride=2, groups=C)

        return out

class SpectralChannelAttention(nn.Module):
    """
    Dynamically recalibrates high-frequency wavelet subband channels (LH, HL, HH).
    Suppresses isotropic specular reflection noise while amplifying structured mucosal boundary bands.
    Uses dual-pooling (GAP + GMP) to capture both subband energy and peak glare anomalies.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super(SpectralChannelAttention, self).__init__()
        mid_channels = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dual-pooling excitation captures both mean subband energy and peak specular glare
        w = self.fc(self.gap(x) + self.gmp(x))
        return x * w


class WaveletHighFrequencyFusion(nn.Module):
    """
    High-Frequency Wavelet Fusion with Active Spectral Channel Attention (SCA).
    Ingests concatenated [LH, HL, HH] high-frequency subbands, actively recalibrates
    channel weights to filter glare, and projects to output channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(WaveletHighFrequencyFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Spectral attention across concatenated high-frequency subbands [LH, HL, HH] (3 * in_channels)
        self.spectral_attn = SpectralChannelAttention(in_channels * 3)
        self.conv1 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
        x = torch.cat([LH, HL, HH], dim=1)
        x = self.spectral_attn(x)
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.gelu(self.bn2(self.conv2(x)))
        return x


if __name__ == '__main__':
    x = torch.randn(2, 16, 64, 64)
    dwt = DWT_2D()
    LL, LH, HL, HH = dwt(x)
    print(f"DWT Output Shapes: LL={LL.shape}, LH={LH.shape}, HL={HL.shape}, HH={HH.shape}")

    idwt = I_DWT_2D()
    recon = idwt(LL, LH, HL, HH)
    print(f"I-DWT Recon Shape: {recon.shape}")
    print(f"Max Recon Error: {torch.max(torch.abs(x - recon)).item()}")

    fusion = WaveletHighFrequencyFusion(16, 32)
    hf = fusion(LH, HL, HH)
    print(f"HF Fusion Shape: {hf.shape}")
    loss = hf.sum()
    loss.backward()
    assert fusion.spectral_attn.fc[0].weight.grad is not None, "Spectral attention must receive gradients"
    print("WaveletHighFrequencyFusion with SpectralChannelAttention verified successfully!")
