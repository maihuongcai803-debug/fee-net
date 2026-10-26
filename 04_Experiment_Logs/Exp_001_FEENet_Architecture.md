---
title: "Exp_001_FEENet_Architecture"
date: "2026-09-22"
tags: ["#architecture", "#wavelet", "#edge-aware"]
---

# Exp_001_FEENet_Architecture: Initial Model Construction

> [!CAUTION] ⛔ 【已废弃 / 仅保留早期概念原型归档】
> 本实验属于项目初期的可行性概念验证（PoC），使用了未加阻尼门控的初代 5.86M 架构，且后续在真实临床基准中已被重构升级为 **FEE-Net v2.1** ([[04_Experiment_Logs/Exp_016_FEE_Net_v2_1_Bilinear_Depthwise_and_DeepSupervision|Exp-016]]) 与 **FEENetEarlyExit** ([[04_Experiment_Logs/Exp_025_FEENet_EarlyExit_Pipeline_Integration_and_Gradient_Decoupling|Exp-025]])。详细归档见 [[04_Experiment_Logs/Exp_Archive_Legacy_Prototypes|历史实验归档]]。**本实验参数配置不可直接用于论文发表**。

## 实验日期与编号 (Experiment Date & ID)
- Date: 2026-09-22
- ID: Exp_001_FEENet_Architecture

## 实验目的与修改动机 (Objective & Motivation)
The objective is to implement the Core Architecture of FEE-Net (Frequency-guided Edge-aware Network) designed to improve segmentation performance through frequency analysis. High-frequency information corresponds strongly to object boundaries/edges. Utilizing the Haar 2D Discrete Wavelet Transform (DWT), we extract these edges explicitly and fuse them into the semantic decoding path using a novel EdgeGatedModule (EGM).

## 涉及文件与核心代码改动 (Involved Files & Core Changes)
- `models/modules/dwt_module.py`: Implemented `DWT_2D` and `I_DWT_2D` using fixed Haar convolution kernels (`F.conv2d`). Also created `WaveletHighFrequencyFusion` to aggregate `LH, HL, HH` bands.
- `models/modules/egm_module.py`: Implemented `EdgeGatedModule` which computes an attention map $A_e$ from decoder and frequency features, and modulates the features accordingly.
- `models/fee_net.py`: Constructed the lightweight `FEENet` (~6M params) combining 4 spatial encoder stages, a dilated bottleneck, multi-scale DWT branches, and a dual-head decoder.
- `utils/losses.py`: Added a `CompoundLoss` consisting of BCE, Soft Dice Loss, and Boundary Loss (via a `LaplacianEdgeExtractor`).

## 超参数与训练配置 (Hyperparameters & Training Config)
- Initial Model capacity targeted at 4-6M parameters.
- Soft Dice Loss with `smooth=1e-5`.
- Laplacian edge filter uses standard 3x3 kernel.
- Compound loss weights: $\lambda_{BCE} = 1.0, \lambda_{Dice} = 1.0, \lambda_{Boundary} = 1.0$.

## 结论与后续改进建议 (Conclusion & Future Work)
- The initial architecture passes dimensional sanity checks.
- Parameter count has been optimized to fit the 4M ~ 6M budget.
- Future work involves integrating this architecture into the training loop and performing hyperparameter sweeps on the loss weights.
