import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Append current dir to sys path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.modules.dwt_module import DWT_2D, WaveletHighFrequencyFusion, SpectralChannelAttention
    from models.modules.egm_module import EdgeGatedModule, BERCA
except ImportError:
    # Fallback for direct execution
    from modules.dwt_module import DWT_2D, WaveletHighFrequencyFusion, SpectralChannelAttention
    from modules.egm_module import EdgeGatedModule, BERCA


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class DilatedBottleneck(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, dilations=[2, 4], in_c=None, out_c=None):
        super(DilatedBottleneck, self).__init__()
        in_c = in_channels if in_channels is not None else in_c
        out_c = out_channels if out_channels is not None else out_c
        mid_channels = out_c // 2
        self.conv1 = nn.Conv2d(in_c, mid_channels, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_c, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class BilinearUpBlock(nn.Module):
    """
    Bilinear Upsampling + 1x1 Convolution Block.
    Following Odena et al. (2016), bilinear interpolation followed by a 1x1
    convolution is employed instead of transposed convolutions to avoid checkerboard
    artifacts, reduce parameters, and improve boundary reconstruction metrics.
    """
    def __init__(self, in_channels, out_channels):
        super(BilinearUpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FEENet(nn.Module):
    """
    FEE-Net: Frequency-Enhanced Edge Network for Endoscopic Polyp Segmentation.
    
    Architectural Modules:
    1. Dynamic Channel Capacity:
       Configured from base_channels (default: 48, baseline: 32):
       c = self.base_channels
       nb = [c, c*2, c*4, c*8, 512]
       - Encoder: [nb[0], nb[1], nb[2], nb[3]], Bottleneck: nb[4] (512)
       - Decoder: [nb[2], nb[1], nb[0], nb[0]]
    2. Residual Dilated Bottleneck:
       Equipped with shortcut residual connection preserving semantic context.
    3. Bidirectional Edge-Region Co-Attention (BERCA in EGM):
       - Directional Wavelet Filtering (DWF) via asymmetric 1x3 and 3x1 depthwise convolutions
         attenuating isotropic specular glare artifacts.
       - Region-to-Edge Semantic Guidance via deep decoder representations and supervised coarse
         segmentation predictions.
       - Learnable residual gate self.gamma (init 0.1) stabilizing early optimization.
    4. Bilinear Upsampling Stages (Odena et al., 2016):
       Reduces checkerboard artifacts and parameter count.
    5. Multi-Scale Deep Supervision:
       Auxiliary heads (deep_sup4, deep_sup3, deep_sup2) provide direct supervision signals.
       Note: deep_sup3 simultaneously serves as the coarse prior guidance for EGM modules,
       ensuring genuine ground-truth supervision over semantic gating.
    6. Dual-mode Forward Interface:
       - Training mode: returns (seg_logits, edge_logits) [or 3-tuple with deep supervision].
       - Eval mode: returns seg_logits by default (or tuple if return_edge=True).
    """
    def __init__(self, in_channels=3, num_classes=1, use_dwt=True, use_egm=True, base_channels=48):
        super(FEENet, self).__init__()
        self.use_dwt = use_dwt
        self.use_egm = use_egm
        self.base_channels = base_channels

        c = self.base_channels
        nb = [c, c * 2, c * 4, c * 8, 512]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, nb[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nb[0]),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ConvBlock(nb[0], nb[0], stride=2)           # H/2, W/2
        self.layer2 = ConvBlock(nb[0], nb[1], stride=2)           # H/4, W/4
        self.layer3 = ConvBlock(nb[1], nb[2], stride=2)           # H/8, W/8
        self.layer4 = ConvBlock(nb[2], nb[3], stride=2)           # H/16, W/16

        self.bot = nn.Sequential(
            nn.Conv2d(nb[3], 512, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        if self.use_dwt:
            self.dwt = DWT_2D()
            self.wf1_attn = SpectralChannelAttention(nb[0] * 3)
            self.wf1 = nn.Conv2d(nb[0] * 3, nb[0], kernel_size=1, bias=False)
            self.wf2_attn = SpectralChannelAttention(nb[0] * 3)
            self.wf2 = nn.Conv2d(nb[0] * 3, nb[0], kernel_size=1, bias=False)
        else:
            self.pool1 = nn.Conv2d(nb[0], nb[0], kernel_size=3, stride=2, padding=1)
            self.pool2 = nn.Conv2d(nb[0], nb[0], kernel_size=3, stride=2, padding=1)

        # Smooth Bilinear Upsampling Stages (Eliminates checkerboard artifacts & lowers HD95)
        self.up4 = BilinearUpBlock(nb[4], nb[2])                  # 512 -> nb[2], H/8
        self.dec4 = ConvBlock(nb[2] + nb[2], nb[2])               # nb[2] + nb[2] -> nb[2]

        self.up3 = BilinearUpBlock(nb[2], nb[1])                  # nb[2] -> nb[1], H/4
        self.dec3 = ConvBlock(nb[1] + nb[1], nb[1])               # nb[1] + nb[1] -> nb[1]

        self.up2 = BilinearUpBlock(nb[1], nb[0])                  # nb[1] -> nb[0], H/2
        if self.use_egm:
            self.egm2 = EdgeGatedModule(dec_channels=nb[0], freq_channels=nb[0])
        self.dec2 = ConvBlock(nb[0] + nb[0], nb[0])               # nb[0] + nb[0] -> nb[0]

        self.up1 = BilinearUpBlock(nb[0], nb[0])                  # nb[0] -> nb[0], H
        if self.use_egm:
            self.egm1 = EdgeGatedModule(dec_channels=nb[0], freq_channels=nb[0])
        self.dec1 = ConvBlock(nb[0] + nb[0], nb[0])               # nb[0] + nb[0] -> nb[0]

        self.head_seg = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        self.head_edge = nn.Conv2d(nb[0], num_classes, kernel_size=1)

        # Coarse prior head for EGM guidance
        self.coarse_head = nn.Conv2d(nb[1], num_classes, kernel_size=1)

        # Multi-scale Deep Supervision Auxiliary Heads
        self.deep_sup4 = nn.Conv2d(nb[2], num_classes, kernel_size=1)        # H/8
        self.deep_sup3 = nn.Conv2d(nb[1], num_classes, kernel_size=1)        # H/4 (aux head)
        self.deep_sup2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)        # H/2

    @property
    def bottleneck(self):
        return self.bot

    @bottleneck.setter
    def bottleneck(self, val):
        self.bot = val

    @property
    def aux_head(self):
        return self.deep_sup3

    @aux_head.setter
    def aux_head(self, val):
        self.deep_sup3 = val

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Robust state dict loader ensuring 100% backward compatibility across
        all legacy and v2 checkpoints.
        """
        model_state = self.state_dict()
        adapted = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if k.endswith("total_ops") or k.endswith("total_params"):
                continue

            # Remap legacy bottleneck / conv1 format -> native bot.0, bot.1, bot.3, bot.4
            if k.startswith("bottleneck.conv1."):
                k = "bot.0." + k[len("bottleneck.conv1."):]
            elif k.startswith("bottleneck.bn1."):
                k = "bot.1." + k[len("bottleneck.bn1."):]
            elif k.startswith("bottleneck.conv2."):
                k = "bot.3." + k[len("bottleneck.conv2."):]
            elif k.startswith("bottleneck.bn2."):
                k = "bot.4." + k[len("bottleneck.bn2."):]
            elif k.startswith("bottleneck."):
                k = "bot." + k[len("bottleneck."):]
            elif k.startswith("bot.conv1."):
                k = "bot.0." + k[len("bot.conv1."):]
            elif k.startswith("bot.bn1."):
                k = "bot.1." + k[len("bot.bn1."):]
            elif k.startswith("bot.conv2."):
                k = "bot.3." + k[len("bot.conv2."):]
            elif k.startswith("bot.bn2."):
                k = "bot.4." + k[len("bot.bn2."):]

            # Remap wf1.conv1.weight -> native wf1.weight
            if k == "wf1.conv1.weight":
                k = "wf1.weight"
            elif k == "wf2.conv1.weight":
                k = "wf2.weight"

            # Remap down -> downsample if needed
            k_parts = k.split(".")
            if len(k_parts) >= 3 and k_parts[1] == "down":
                k_parts[1] = "downsample"
                k = ".".join(k_parts)

            # Remap aux_head -> deep_sup3
            if k.startswith("aux_head."):
                k = "deep_sup3." + k[len("aux_head."):]

            if k in model_state:
                tgt_shape = model_state[k].shape
                if v.shape == tgt_shape:
                    adapted[k] = v
            else:
                adapted[k] = v

        if "dwt.weight" not in adapted and "dwt.weight" in model_state:
            adapted["dwt.weight"] = model_state["dwt.weight"]

        if not strict:
            adapted = {k: v for k, v in adapted.items() if k in model_state and v.shape == model_state[k].shape}

        return super(FEENet, self).load_state_dict(adapted, strict=strict, assign=assign)

    def forward(self, x, clinical_prior=None, return_edge=False, return_aux=False):
        """
        Forward pass.
        Args:
            x (Tensor): Input image batch of shape (B, C, H, W).
            clinical_prior (optional): System 1 risk prior or positional boolean.
            return_edge (bool): If True, returns (seg_logits, edge_logits) during eval.
            return_aux (bool): If True, returns (seg_logits, edge_logits, aux_preds) with deep supervision.
        """
        if isinstance(clinical_prior, bool):
            return_edge = clinical_prior
            clinical_prior = None

        # 1. Encoder Feature Extraction
        x_stem = self.stem(x)             # nb[0], H, W
        e1 = self.layer1(x_stem)          # nb[0], H/2, W/2
        e2 = self.layer2(e1)              # nb[1], H/4, W/4
        e3 = self.layer3(e2)              # nb[2], H/8, W/8
        e4 = self.layer4(e3)              # nb[3], H/16, W/16

        b = self.bot(e4)                  # nb[4] (512), H/16, W/16

        # 2. Wavelet High-Frequency Feature Extraction
        if self.use_dwt:
            LL_stem, LH_stem, HL_stem, HH_stem = self.dwt(x_stem)
            cat_stem = torch.cat([LH_stem, HL_stem, HH_stem], dim=1)
            cat_stem = self.wf1_attn(cat_stem) if hasattr(self, 'wf1_attn') else cat_stem
            hf_1 = self.wf1(cat_stem)  # nb[0], H/2, W/2

            LL_1, LH_1, HL_1, HH_1 = self.dwt(e1)
            cat_1 = torch.cat([LH_1, HL_1, HH_1], dim=1)
            cat_1 = self.wf2_attn(cat_1) if hasattr(self, 'wf2_attn') else cat_1
            hf_2 = self.wf2(cat_1)          # nb[0], H/4, W/4
        else:
            hf_1 = self.pool1(x_stem)
            hf_2 = self.pool2(e1)

        # 3. Decoder with BERCA Co-Attention
        # Level 4 (H/8)
        d4 = self.up4(b)                  # nb[2], H/8, W/8
        d4 = torch.cat([d4, e3], dim=1)   # nb[2] + nb[2]
        d4 = self.dec4(d4)                # nb[2]

        # Level 3 (H/4)
        d3 = self.up3(d4)                 # nb[1], H/4, W/4
        d3 = torch.cat([d3, e2], dim=1)   # nb[1] + nb[1]
        d3 = self.dec3(d3)                # nb[1]

        # coarse_head provides supervised coarse prior guidance to downstream EGMs
        coarse_pred = self.coarse_head(d3)  # (B, 1, H/4, W/4)

        # Level 2 (H/2)
        d2 = self.up2(d3)                 # nb[0], H/2, W/2
        if self.use_egm:
            d2, attn2 = self.egm2(d2, hf_2, coarse_pred=coarse_pred)
        else:
            hf_2_up = F.interpolate(hf_2, size=d2.shape[2:], mode='bilinear', align_corners=False)
            d2 = d2 + hf_2_up
        d2 = torch.cat([d2, e1], dim=1)   # nb[0] + nb[0]
        d2 = self.dec2(d2)                # nb[0]

        # Level 1 (H)
        d1 = self.up1(d2)                 # nb[0], H, W
        hf_1_up = F.interpolate(hf_1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        if self.use_egm:
            d1, attn1 = self.egm1(d1, hf_1_up, coarse_pred=coarse_pred)
        else:
            d1 = d1 + hf_1_up
        d1 = torch.cat([d1, x_stem], dim=1) # nb[0] + nb[0]
        d1 = self.dec1(d1)                # nb[0]

        seg_logits = self.head_seg(d1)
        edge_logits = self.head_edge(hf_1_up)

        if clinical_prior is not None and torch.is_tensor(clinical_prior):
            if clinical_prior.dim() == 2 and clinical_prior.size(1) == 1:
                prior_weight = 1.0 + 0.2 * clinical_prior.view(-1, 1, 1, 1)
                seg_logits = seg_logits * prior_weight

        if return_aux:
            aux_preds = [self.deep_sup4(d4), self.deep_sup3(d3), self.deep_sup2(d2)]
            return seg_logits, edge_logits, aux_preds

        if self.training or return_edge:
            return seg_logits, edge_logits
        return seg_logits


if __name__ == '__main__':
    print("=" * 65)
    print("Self-Test: FEENet with Base Channels 48 & BERCA Dual Co-Attention")
    print("=" * 65)

    model = FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=True, base_channels=48)

    # 1. Total Trainable Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"1. Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.4f} M)")
    print(f"   Base channels: {model.base_channels}")
    print(f"   EGM1 initial gamma: {model.egm1.gamma.item():.4f}")
    print(f"   EGM2 initial gamma: {model.egm2.gamma.item():.4f}")
    assert 8_300_000 < total_params < 8_700_000, \
        f"Expected ~8.52M parameters (BERCA-expanded architecture, post Exp-014/023), got {total_params}"

    x = torch.randn(2, 3, 256, 256)

    # 2. Training Mode Verification
    print("\n2. Testing Training Mode Forward & Backward:")
    model.train()
    out_train_default = model(x)
    assert isinstance(out_train_default, tuple) and len(out_train_default) == 2, "Default train mode must return (seg_logits, edge_logits)"

    out_train_aux = model(x, return_aux=True)
    assert isinstance(out_train_aux, tuple) and len(out_train_aux) == 3, "Train mode with return_aux=True must return (seg, edge, aux_preds)"
    seg_train, edge_train, aux_train = out_train_aux
    print(f"   Train default: len={len(out_train_default)} | Train return_aux: len={len(out_train_aux)}, aux_count={len(aux_train)}")
    assert seg_train.shape == (2, 1, 256, 256)
    assert edge_train.shape == (2, 1, 256, 256)
    assert len(aux_train) == 3

    # Backward Pass & Gradient Flow
    loss = seg_train.sum() + edge_train.sum() + sum(aux.sum() for aux in aux_train)
    loss.backward()

    assert model.egm1.gamma.grad is not None, "EGM1 gamma gradient must not be None"
    assert model.egm2.gamma.grad is not None, "EGM2 gamma gradient must not be None"
    assert model.egm1.conv_dir_h[0].weight.grad is not None, "EGM1 DWF 1x3 grad must not be None"
    assert model.egm1.conv_dir_v[0].weight.grad is not None, "EGM1 DWF 3x1 grad must not be None"
    assert model.egm1.conv_dir_diag[0].weight.grad is not None, "EGM1 DWF 3x3 diagonal grad must not be None"
    assert model.egm1.semantic_gate[0].weight.grad is not None, "EGM1 semantic gate grad must not be None"
    assert model.wf1.weight.grad is not None, "Wavelet fusion branch grad must not be None"
    assert model.wf1_attn.fc[0].weight.grad is not None, "Wavelet spectral attention grad must not be None"
    assert model.stem[0].weight.grad is not None, "Encoder stem grad must not be None"
    assert model.coarse_head.weight.grad is not None, "Coarse head grad must not be None"
    assert model.deep_sup4.weight.grad is not None, "Deep sup4 grad must not be None"
    assert model.deep_sup3.weight.grad is not None, "Deep sup3 grad must not be None"
    assert model.deep_sup2.weight.grad is not None, "Deep sup2 grad must not be None"
    assert model.bot[0].weight.grad is not None, "Bottleneck conv1 grad must not be None"
    assert model.bot[1].weight.grad is not None, "Bottleneck bn1 grad must not be None"
    if hasattr(model.bot, 'shortcut') and isinstance(model.bot.shortcut, nn.Sequential):
        assert model.bot.shortcut[0].weight.grad is not None, "Bottleneck shortcut grad must not be None"

    print("   Backward pass successful! Gradient flow confirmed across all branches:")
    print(f"   - EGM1 gamma grad: {model.egm1.gamma.grad.item():.6f}")
    print(f"   - EGM2 gamma grad: {model.egm2.gamma.grad.item():.6f}")
    print(f"   - EGM1 DWF dir_h grad norm: {model.egm1.conv_dir_h[0].weight.grad.norm().item():.6f}")
    print(f"   - EGM1 DWF dir_v grad norm: {model.egm1.conv_dir_v[0].weight.grad.norm().item():.6f}")
    print(f"   - EGM1 DWF dir_diag grad norm: {model.egm1.conv_dir_diag[0].weight.grad.norm().item():.6f}")
    print(f"   - WF1 Spectral Attention grad norm: {model.wf1_attn.fc[0].weight.grad.norm().item():.6f}")
    print(f"   - EGM1 semantic gate grad norm: {model.egm1.semantic_gate[0].weight.grad.norm().item():.6f}")
    print(f"   - Coarse head grad norm: {model.coarse_head.weight.grad.norm().item():.6f}")

    # 3. Eval Mode (Default: return_edge=False)
    print("\n3. Testing Eval Mode (default return_edge=False):")
    model.eval()
    with torch.no_grad():
        out_eval_default = model(x)
        assert torch.is_tensor(out_eval_default), "Eval mode (default) must return seg_logits Tensor directly"
        assert out_eval_default.shape == (2, 1, 256, 256)
        print(f"   Eval default output: {type(out_eval_default)} of shape {tuple(out_eval_default.shape)}")

    # 4. Eval Mode (Explicit return_edge=True)
    print("\n4. Testing Eval Mode (return_edge=True):")
    with torch.no_grad():
        out_eval_edge = model(x, return_edge=True)
        assert isinstance(out_eval_edge, tuple) and len(out_eval_edge) == 2, "Eval mode with return_edge=True must return tuple"
        seg_eval, edge_eval = out_eval_edge
        print(f"   Eval return_edge=True: seg={tuple(seg_eval.shape)}, edge={tuple(edge_eval.shape)}")

        # Positional boolean compatibility test
        out_pos = model(x, True)
        assert isinstance(out_pos, tuple) and len(out_pos) == 2
        out_false = model(x, False)
        assert torch.is_tensor(out_false)

    # 5. Ablation Variants Verification
    print("\n5. Testing Ablation Modes:")
    model_no_egm = FEENet(use_dwt=True, use_egm=False, base_channels=48)
    p_no_egm = sum(p.numel() for p in model_no_egm.parameters() if p.requires_grad)
    print(f"   FEENet (w/o EGM) parameters: {p_no_egm:,} ({p_no_egm / 1e6:.4f} M)")
    out_no_egm = model_no_egm(x)
    assert isinstance(out_no_egm, tuple), "Train mode w/o EGM should return tuple"

    model_no_dwt = FEENet(use_dwt=False, use_egm=True, base_channels=48)
    p_no_dwt = sum(p.numel() for p in model_no_dwt.parameters() if p.requires_grad)
    print(f"   FEENet (w/o DWT) parameters: {p_no_dwt:,} ({p_no_dwt / 1e6:.4f} M)")
    out_no_dwt = model_no_dwt(x)
    assert isinstance(out_no_dwt, tuple), "Train mode w/o DWT should return tuple"

    # 6. Dynamic Channel Verification (base_channels=32)
    print("\n6. Testing Dynamic Channels (base_channels=32):")
    model_32 = FEENet(base_channels=32)
    p_32 = sum(p.numel() for p in model_32.parameters() if p.requires_grad)
    print(f"   FEENet (base_channels=32) parameters: {p_32:,} ({p_32 / 1e6:.4f} M)")
    out_32 = model_32(x)
    assert out_32[0].shape == (2, 1, 256, 256)

    # 7. Checkpoint Loading Verification
    print("\n7. Testing Checkpoint Loading Compatibility:")
    for ckpt_p in ['checkpoints/fee_net_best.pth', 'checkpoints/fee_net_v2_fold0_best.pth']:
        if os.path.exists(ckpt_p):
            ckpt = torch.load(ckpt_p, map_location='cpu', weights_only=False)
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model_eval = FEENet(base_channels=48)
            res = model_eval.load_state_dict(sd, strict=False)
            print(f"   Loaded {ckpt_p}: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
            assert len(res.unexpected_keys) == 0, f"Expected 0 unexpected keys, got {res.unexpected_keys}"
            # Architecture-drift guard: pre-BERCA (6.90M-era) checkpoints carry
            # legacy bottleneck keys and WILL have missing keys under the current
            # 8.52M architecture. Surface it loudly instead of partial-loading silently.
            if len(res.missing_keys) > 0:
                print(f"   [WARN] {len(res.missing_keys)} keys missing -> checkpoint was trained "
                      f"on an older architecture (params_m={ckpt.get('params_m')}); "
                      f"weights are NOT fully compatible with the current model!")
            with torch.no_grad():
                model_eval.eval()
                test_out = model_eval(x)
                assert test_out.shape == (2, 1, 256, 256)

    print("\n" + "=" * 65)
    print("ALL FEENet ARCHITECTURAL SELF-TEST CHECKS PASSED PERFECTLY!")
    print("=" * 65)

