---
title: IEEE 标准 EI/SCI 论文大纲与撰写规划 (Conformal Triage & Zero-Leakage Protocol)
date: 2026-10-03
tags:
  - Paper_Draft
  - IEEE
  - EI_Index
  - Conformal_Risk_Control
  - Safe_Screening
---

# 📑 FEE-Net: 基于保形认证分流筛查与零泄漏协议的轻量级内窥镜辅助系统

> **写作战略定位**：从传统的“刷榜新架构/超越SOTA”（经审计该路线在现有数据规模下无法立足）全面转型为**“轻量工程化分割 + 有限样本可证安全筛查系统 + 零泄漏可复现协议”**。
> **核心实验支撑**：Exp-027 (v2 Splits 150例盲测集)、Exp-031 (单一真源架构重跑与探针分析)、`assets/mixed_cohort_screening_table.md` (无泄漏混合队列保形认证实测)。

---

## 标题与摘要 (Title & Abstract)
- **Title**: *FEE-Net: A Lightweight Dual-Domain Architecture with Conformal-Certified Screening Triage and Zero-Leakage Protocol for Colorectal Endoscopy*
- **Abstract** (200–250 words):
  - **临床与算力背景**：结直肠癌筛查中 80%+ 视野为健康黏膜，全量逐像素致密分割造成计算严重冗余；同时微小息肉存在被早退门控漏诊的高危风险。
  - **核心创新与安全保障**：
    1. 提出首个在结肠镜筛查中实现有限样本安全认证的早退分流架构：基于 Clopper-Pearson 严格精确逆二项分布，在验证集标定阈值 $\tau^* = 0.96$，并在独立认证集上完成单次无泄漏评估；
    2. 解决传统全阳性测试集无法测度特异性的根本缺陷：从 150 例独立盲测帧构建无泄漏混合队列（150 病灶帧 + 267 正常黏膜帧，奇偶划分校准/认证集），实测达成 **FNR = 0.00%**（数学保证 $\le 3.92\%$，满足 $5\%$ 安全上界），同时实现 **64.66% 正常黏膜帧安全旁路**，端到端延迟降低至 7.45 ms ($1.45\times$ 吞吐加速)；
    3. 设计轻量频域-空间域双域网络 FEE-Net（8.54M 参数，12.44 GFLOPs），采用 2D Haar DWT + 频域通道注意力（SCA）抑制黏膜高光反光，结合全景八邻域滤波边缘门控模块（ODWF-EGM）保护病灶边缘。
  - **客观实验结论**：在 150 例独立零泄漏测试集上，与同预算严格复现的经典基线（U-Net, Attention U-Net, UNet++）进行客观对比，揭示辅助损失与小样本下的真实架构表现，并开源单一真源可复现流水线。

---

## I. INTRODUCTION (引言)
- **1.1 临床背景与高通量筛查痛点**：
  - 结直肠息肉筛查的临床意义与医师视觉疲劳；
  - 门诊与群体普查中的“阴性多数”现象：80%+ 帧为健康黏膜，高功耗持续运行在便携式/边缘内窥镜主机上的算力瓶颈。
- **1.2 现有技术的双重困境**：
  - 分割模型在下采样中高频细节衰减与黏膜镜面反光混叠；
  - 动态推理/早退网络缺乏统计安全界：一旦为了省算力而早退，极易漏诊微小扁平腺瘤，临床不可承受。
- **1.3 本文主要贡献 (Fourfold Contributions)**：
  1. **零泄漏可复现基准协议**：严格隔离 150 例分层独立测试集，建立单真源代码内联与溯源包流水线；
  2. **有限样本保形风险控制分流门（严格分离校准与认证）**：基于 Clopper-Pearson 精确置信上界，在混合队列实现可证安全（FNR $\le 3.92\%$）与 64.66% 正常帧旁路；
  3. **频域引导与全景滤波双域轻量架构 (FEE-Net)**：8.54M / 12.44 GFLOPs，以 2D-DWT + SCA 滤除反光，ODWF-EGM 消除方形卷积方向盲区；
  4. **诚实客观的基准实验与消融归因**：如实报告各组件在特定数据规模下的真实收益与局限性，修正文献中关于形态学边界损失的引文偏差。

---

## II. RELATED WORK (相关工作)
- **2.1 Polyp and Medical Image Segmentation**:
  - U-Net、Attention U-Net、UNet++ 的经典演进；
  - 现代息肉分割模型（PraNet 等）及参数规模与算力负担分析。
- **2.2 Frequency-Domain Learning in Colonoscopy**:
  - 小波变换（DWT）在医学图像中的应用脉络；
  - 梳理近年同类工作（WRANet 2023, AdaptUNet 2024），明确本文聚焦轻量工程化与安全分流系统，而非宣称跨文献新 SOTA。
- **2.3 Morphological & Boundary-Aware Losses**:
  - 正确溯源：El Jurdi et al. (MICCAI 2021) 膨胀-腐蚀周长损失、Karimi & Salcudean (MedIA 2022) Boundary IoU 与 (IEEE TMI 2019) 豪斯多夫距离损失，纠正业界常见误称。
- **2.4 Conformal Risk Control & Clinical Triage**:
  - 保形预测（Conformal Prediction）与有限样本风控（Bates et al., Angelopoulos et al.）；
  - 医学分割中的保形阈值控制（Conformal Lesion Segmentation, ICIPCA 2025 CRC-FDR）；
  - 强调校准集与测试集严格互斥的可交换性要求，以及应对漏诊损失不可认证风险的工程解法。

---

## III. METHODOLOGY (提出方法)
- **3.1 总体双阶段系统工作流 (Overall Architecture & System Pipeline)**
- **3.2 2D Haar DWT 与频域通道注意力 (SCA)**:
  - 正交子带解耦 ($LL, LH, HL, HH$)；
  - GAP+GMP 双通道压缩抑制镜面反光高光。
- **3.3 全景八邻域滤波边缘门控模块 (ODWF-EGM)**:
  - $1\times 3, 3\times 1, 3\times 3$ 深度可分离方向滤波组合；
  - 粗预测语义置信门与残差阻尼因子 $\gamma$。
- **3.4 双分支局部峰值早退头 (GAP+GMP: 4C) 与梯度解耦 (Stop-Gradient)**
- **3.5 分离式有限样本保形风险控制门禁 (Conformal Risk Control)**:
  - 校准集标定：$\tau^* = \inf \{ \tau \in [0, 1] : \widehat{R}_{cal}(\tau) + \dots \le \alpha \}$；
  - Clopper-Pearson 严格精确二项置信上界数学形式；
  - 认证集（Certify Set）单次推断与经验安全验证。

---

## IV. EXPERIMENTAL SETUP (实验设置)
- **4.1 Kvasir-SEG 基准与 v2 Splits 零泄漏协议**:
  - 1000 例按息肉面积五等分分层，严格前置抽样 150 例独立测试集；
  - 850 例分为 5 折，Fold 0 训练验证。
- **4.2 临床混合队列构建 (Mixed-Cohort Construction)**:
  - 解决 Kvasir-SEG 全阳性缺陷：从 150 例测试帧中无泄漏剪裁 267 例正常背景，按帧奇偶划分为校准组（75 病灶 + 134 正常）与认证组（75 病灶 + 133 正常）。
- **4.3 评测指标与硬件环境**:
  - 分割指标：Dice, mIoU, HD95 (px)；
  - 筛查指标：FNR (漏诊率), Normal Bypass Rate (正常旁路率), CP Upper Bound；
  - 硬件效率：Params (M), FLOPs (G), Latency (ms), Throughput (FPS)。

---

## V. RESULTS AND DISCUSSION (实验结果与讨论)
- **5.1 混合队列保形安全筛查实证 (Table II, 核心贡献)**:
  - 标定 $\tau^* = 0.96$ 的认证结果：FNR = 0.00%（理论上界 3.92% $\le 5\%$）；
  - 正常黏膜旁路率 64.66%，全队列旁路 41.35%，实测延迟从 10.77 ms 降至 7.45 ms ($1.45\times$ 加速)。
- **5.2 独立测试集分割基准对比 (Table I)**:
  - 报告同协议、同硬件、同随机种子下与 U-Net, Attention U-Net, UNet++ 的客观对比；
  - 客观讨论参数量与 FLOPs 优势，诚实呈现各架构在小样本医学数据下的权衡。
- **5.3 本地探针矩阵消融与归因分析**:
  - 欠训练假设排查（35 轮 vs 60 轮收敛分析）；
  - 辅助损失项（深监督、边缘监督、边界损失）在主干容量中的相互作用。
- **5.4 定性结果与边缘贴合度可视化**:
  - 反光区域与微小病灶分割效果对比。
- **5.5 局限性与开放挑战 (Limitations)**:
  - 跨中心数据泛化与多中心临床验证需求；
  - 频域组件在极低数据量下的归纳偏置特性；
  - 严格二项保形界在不同患病率群体中的灵敏度分析。

---

## VI. CONCLUSION (结论)
- 总结论文的方法学价值：以零泄漏可复现协议为基石，通过频域抗噪轻量网络与保形安全分流门，为高通量结直肠癌筛查提供具备有限样本数学保证的实用工程范式。
