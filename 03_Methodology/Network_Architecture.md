---
title: FEE-Net 网络架构与张量流转图
date: 2026-09-21
tags:
  - Methodology
  - Architecture
  - Deep_Learning
---

# 🏗️ FEE-Net 详细网络架构与数据流

本笔记记录 FEE-Net 的模块划分、张量尺寸流转与连接拓扑。

---

## 1. 网络全局数据流图

```mermaid
flowchart TD
    subgraph Input_Stage["输入阶段"]
        Img["原始图像 (B, 3, 256, 256)"]
    end

    subgraph Spatial_Encoder["空间编码分支 (Spatial Encoder)"]
        E1["Enc Stage 1: ConvBlock (B, 32, 256, 256)"]
        P1["Downsample (Stride 2) -> (B, 64, 128, 128)"]
        E2["Enc Stage 2: ConvBlock (B, 64, 128, 128)"]
        P2["Downsample (Stride 2) -> (B, 128, 64, 64)"]
        E3["Enc Stage 3: ConvBlock (B, 128, 64, 64)"]
        P3["Downsample (Stride 2) -> (B, 256, 32, 32)"]
        Bottleneck["Bottleneck: Dilated Conv Block (B, 512, 16, 16)"]
    end

    subgraph Frequency_Branch["小波频域多尺度分支 (Wavelet DWT Branch)"]
        DWT1["DWT 1 -> LH, HL, HH (B, 9, 128, 128)"]
        Proj1["1x1 Conv + GELU -> Freq_1 (B, 64, 128, 128)"]
        DWT2["DWT 2 -> Sub-bands (B, 9, 64, 64)"]
        Proj2["1x1 Conv + GELU -> Freq_2 (B, 128, 64, 64)"]
    end

    subgraph Edge_Gated_Decoder["边缘门控解码分支 (Edge-Gated Decoder)"]
        Up3["Upsample (2x) -> (B, 256, 32, 32)"]
        Dec3["Dec Stage 3 (B, 128, 64, 64)"]
        EGM2["EGM Module 2 <br/> (融合 Dec3 与 Freq_2)"]
        Up2["Upsample (2x) -> (B, 128, 64, 64)"]
        Dec2["Dec Stage 2 (B, 64, 128, 128)"]
        EGM1["EGM Module 1 <br/> (融合 Dec2 与 Freq_1)"]
        Up1["Upsample (2x) -> (B, 64, 128, 128)"]
        Dec1["Dec Stage 1 (B, 32, 256, 256)"]
        HeadSeg["Segmentation Head (1x1 Conv) -> (B, 1, 256, 256)"]
        HeadEdge["Edge Auxiliary Head (1x1 Conv) -> (B, 1, 256, 256)"]
    end

    Img --> E1
    E1 --> P1 --> E2 --> P2 --> E3 --> P3 --> Bottleneck
    
    Img --> DWT1 --> Proj1
    Img --> DWT2 --> Proj2
    
    Bottleneck --> Up3 --> Dec3
    E3 -.Skip Connection.-> Dec3
    
    Dec3 --> EGM2
    Proj2 --> EGM2
    EGM2 --> Up2 --> Dec2
    
    E2 -.Skip Connection.-> Dec2
    Dec2 --> EGM1
    Proj1 --> EGM1
    EGM1 --> Up1 --> Dec1
    
    E1 -.Skip Connection.-> Dec1
    Dec1 --> HeadSeg
    Dec1 --> HeadEdge
```

---

## 2. 关键模块张量规范

| 阶段 | 模块输入 | 输出特征尺寸 (以 Batch=8 为例) | 关键层说明 |
| :--- | :--- | :--- | :--- |
| **Input** | Raw Image | `[8, 3, 256, 256]` | 归一化医学图像 |
| **Enc 1** | Input | `[8, 32, 256, 256]` | $3\times3$ Depthwise Separable Conv |
| **DWT 1** | Input | `[8, 9, 128, 128]` | Haar 小波 3 个高频子带（无参数卷积） |
| **Enc 2** | Enc 1 下采样 | `[8, 64, 128, 128]` | 残差卷积块 |
| **Enc 3** | Enc 2 下采样 | `[8, 128, 64, 64]` | 残差卷积块 |
| **Bottleneck** | Enc 3 下采样 | `[8, 256, 32, 32]` | 空洞卷积（膨胀率 2, 4）扩大感受野 |
| **EGM 2** | Dec 3 & Freq 2 | `[8, 128, 64, 64]` | 频域边缘先验门控注意力 |
| **EGM 1** | Dec 2 & Freq 1 | `[8, 64, 128, 128]` | 频域边缘先验门控注意力 |
| **Final Seg**| Dec 1 输出 | `[8, 1, 256, 256]` | Sigmoid 激活输出病灶二值分割掩码 |
| **Final Edge**| Dec 1 输出 | `[8, 1, 256, 256]` | 辅助监督：病灶边缘轮廓图 |
