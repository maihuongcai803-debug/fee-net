---
title: "Paper Summary: FEE-Net"
date: 2026-10-03
tags:
  - Paper_Draft
  - IEEE_Transactions
  - Polyp_Segmentation
  - Conformal_Risk_Control
  - Safe_Screening
---

# 📝 Paper Summary: FEE-Net

**Title**: FEE-Net: A Lightweight Dual-Domain Architecture with Conformal-Certified Screening Triage and Zero-Leakage Protocol for Colorectal Endoscopy  
**Target Venue**: IEEE Transactions on Biomedical Circuits and Systems / IEEE EMBC / ISBI / 计算机学报  
**Associated Manuscripts**: [[Paper_Manuscript_Full]] & `Draft_IEEE_FEE_Net.tex`  
**Strategic Framing**: 轻量工程化分割 + 有限样本可证安全筛查系统 + 零泄漏可复现协议（遵循 `06_Code_Docs/Novelty_Audit_2026-10-03.md`）

---

## 📌 1. Background & Clinical Motivation
- **Clinical Need**: 结直肠癌筛查中息肉的早期精准切除可降低 70% 死亡率，但人工检查面临 15%~28% 漏诊率。
- **Key Bottlenecks in Practice**:
  1. 空间连续下采样导致高频边界细节衰减；
  2. 肠道黏膜镜面反光与液体高光诱发虚假边缘响应；
  3. 微小病灶（$<5$\,mm）在动态推理中易被全局均值池化（GAP）均化而发生筛查早退漏诊；
  4. 群体普查中超过 80% 视野为健康黏膜，全量逐像素致密分割造成巨大的计算浪费。

---

## 🚀 2. Core Contributions
1. **严格零泄漏可复现基准协议 (v2 Splits Protocol)**：
   - 1000 例 Kvasir-SEG 前置隔离 150 例面积分层独立盲测集，代码级断言不相交；
   - 运行时内联仓库源码（`models/`, `utils/`），杜绝实现脱节。
2. **有限样本保形风险控制与混合队列认证 (Conformal Risk Control & Mixed Cohort)**：
   - 严格分离校准集（75 病灶 + 134 正常）与认证集（75 病灶 + 133 正常）；
   - 采用两阶段双重冻结训练（冻结主干权重与 BN 统计量，仅微调 12.7K 分流头），彻底攻克全阳性训练集导致的门控常数坍缩；
   - 基于 Clopper-Pearson 严格逆二项分布，在校准集标定 $\tau^* = 0.59$（上界 $3.92\% \le 5\%$）；
   - 在独立认证集单次验证实测 **FNR = 0.00%**，安全旁路 **99.25% 正常黏膜帧**，平均延迟从 11.30 ms 降至 5.93 ms（**1.91× 端到端硬件实测加速**，等效吞吐量 168.6 FPS）。
3. **轻量双域工程架构 (FEE-Net: 8.539M 参数量)**：
   - 2D Haar DWT + 频域通道注意力（SCA）抑制反光伪影；
   - 全景八邻域滤波边缘门控（ODWF-EGM）消除方形卷积方向盲区。
4. **学术诚信与引文修正**：
   - 正确溯源形态学边界损失（El Jurdi MICCAI 2021 周长损失、Karimi MedIA 2022 Boundary IoU、Karimi IEEE TMI 2019 HD 损失）；
   - 如实呈现基准对比现状与辅助损失在小样本下的相互作用（60 轮消融揭示深监督 +2.86%、边界 +1.03%、边缘 +0.63%，非可加性交互项 +1.57%）。

---

## 📊 3. 核心实验数据概览

### 3.1 混合队列保形筛查认证结果 (Table 2, 核心贡献)
- **校准集**：75 病灶 + 134 正常黏膜 $\to$ 标定 $\tau^* = 0.59$（CP 上界 3.92%）；
- **认证集**：75 病灶 + 133 正常黏膜 $\to$ 实测 FNR = 0.00%（保证成立）；
- **正常黏膜旁路率**：**99.25%**；全队列旁路率：63.5%；
- **实测端到端推理延迟**：空闲 GPU（RTX 4060 Laptop）精测门控 2.84 ms、全路径 11.30 ms，混合流平均延迟降至 **5.93 ms（1.91× 硬件实测加速，168.6 FPS）**；
- **分割精度无损保真**：81.42% Dice 逐位完全一致（阶段 2 训练零精度损失）。

### 3.2 独立测试集分割基准 (Table 1, Kaggle 双卡 T4 60 轮定稿实测)
| 架构 | Dice (%) | mIoU (%) | HD95 (px) | 参数量 (M) | FLOPs (G) | 延迟 (ms) | 吞吐量 (FPS) | 状态说明 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **U-Net** | **84.14** | 76.45 | 36.33 | 7.763 | 13.75 | **10.02** | **99.8** | 经典基线，同协议严谨对比 |
| **Attention U-Net** | 84.00 | 75.83 | 38.73 | 7.852 | 14.03 | 11.57 | 86.5 | 经典基线，同协议严谨对比 |
| **UNet++** | 84.07 | **76.57** | 33.38 | 9.160 | 34.62 | 16.92 | 59.1 | 经典基线，高算力代表 |
| **FEE-Net (Ours Full)** | 79.47 | 70.65 | 38.67 | 8.539 | **12.59** | 15.66 | 63.9 | 轻量双域架构，8.539M 紧凑参数 |
| **FEENetEarlyExit** | **81.42** | 73.07 | 36.41 | 8.552 | 12.59 (网关 0.94) | 17.43 | 57.4 | 早退特征探针多任务正则 (+1.95% Dice) |
| **FEE-Net w/o DWT** | 80.78 | 72.73 | 36.11 | 8.546 | 12.87 | 14.35 | 69.7 | 频域通道注意力消融 |
| **FEE-Net w/o EGM** | 83.08 | 75.59 | **31.71** | 8.522 | 11.83 | 11.94 | 83.8 | 全场最优 HD95 (31.71 px)，速度更快 |

> 注：全部指标来自 `scratch/kaggle_v5_results/benchmark_metrics_independent_test.csv` 单一真源入库产物包，150 例独立测试集盲测评定。
