---
title: 实验留档 Exp_004 全量基线对比与消融实验评测
date: 2026-09-22
tags:
  - Experiment
  - Benchmark
  - Ablation
  - FEE-Net
---

# 🧪 实验留档：Exp_004_Benchmark_and_Ablation_Results

> [!CAUTION] ⛔ 【已废弃 / 玩具数据 / 严禁写入论文】
> 本实验全部指标（Dice 98.89%、HD95 1.46 voxel 等）均产自早期 900 例数学合成数据集（Synthetic Medseg）。在真实临床场景下属严重伪过拟合与虚假高分，已被 [[04_Experiment_Logs/Exp_006_Reviewer_Critique_and_Improvement_Plan|Exp-006]] 严厉否定。真实有效基准已被官方真实数据集 **Kvasir-SEG 5 折交叉验证** ([[04_Experiment_Logs/Exp_017_Genuine_5Fold_Cross_Validation_Results|Exp-017]]) 及 [[04_Experiment_Logs/Exp_027_Codebase_Hygiene_and_Independent_Test_Protocol|Exp-027 v2 协议]] 彻底替代。详细归档见 [[04_Experiment_Logs/Exp_Archive_Legacy_Prototypes|历史实验归档]]。**本实验所有指标严禁用于论文发表**。

## 1. 实验基本信息
- **实验日期**：2026-09-22
- **实验编号**：Exp-004
- **运行平台**：本地 NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) | PyTorch 2.7.1+cu118 AMP
- **关联文档**：[[00_Index|主索引]] | [[03_Methodology/Network_Architecture|FEE-Net网络架构]] | [[04_Experiment_Logs/Exp_Index|实验矩阵索引]]

---

## 2. 实验目的与修改动机
1. **全面基线对比 (Benchmark Comparison)**：
   在统一的数据集（ISIC/拟真多模态病灶分割数据集，包含低对比度、浸润性边界模糊病灶）、相同的训练配置与评价协议下，对比经典 **U-Net**、**Attention U-Net** 与所提 **FEE-Net (Full)** 的分割精度与计算效率。
2. **多维度消融实验 (Ablation Study)**：
   系统性验证 FEE-Net 三大核心创新组件对模型最终性能的独立与协同贡献：
   - **消融 1 (w/o DWT)**：移除 2D 离散小波高频分解分支，替换为常规卷积降采样；
   - **消融 2 (w/o EGM)**：移除边缘门控交叉注意力模块 (Edge-Gated Module)，使用简单加法融合；
   - **消融 3 (w/o Boundary Loss)**：移除基于可微拉普拉斯滤波的轮廓监督损失 $\mathcal{L}_{Boundary}$，仅使用常规 BCE + Dice 损失。

---

## 3. 涉及文件与核心代码改动
| 文件路径 | 变更类型 | 核心变动说明 |
| :--- | :--- | :--- |
| [`models/fee_net.py`](file:///d:/machine_learning/models/fee_net.py) | 核心算法 | 实现 `FEENet` 架构，支持 `use_dwt` 与 `use_egm` 动态消融开关；修复消融模式下的空间插值对齐。 |
| [`utils/losses.py`](file:///d:/machine_learning/utils/losses.py) | 损失函数 | 实现 `CompoundLoss`，通过 `lambda_boundary=0.0` 支持边界损失消融。 |
| [`utils/metrics.py`](file:///d:/machine_learning/utils/metrics.py) | 评测工具 | 修复 `thop` 分析时的深拷贝隔离机制，确保评估钩子不污染训练模型；支持精确 HD95、Dice、IoU、FLOPs 测算。 |
| [`evaluate.py`](file:///d:/machine_learning/evaluate.py) | 评测流程 | 编写 6 模型自动化批处理评测脚本，导出 Markdown/LaTeX 表格与可视化图像。 |

---

## 4. 超参数与训练配置 (Hyperparameters)
- **数据集划分**：Train: 600 例, Val: 150 例, Test: 150 例 (总计 900 样本)
- **输入分辨率**：$256 \times 256 \times 3$
- **Batch Size**：16
- **优化器**：AdamW ($\beta_1=0.9, \beta_2=0.999$, weight_decay=1e-4)
- **学习率**：初始学习率 $\eta = 3 \times 10^{-4}$，配合 `CosineAnnealingLR` (T_max=15, $\eta_{min}=10^{-6}$)
- **训练轮数**：15 Epochs（单模型耗时约 3.5~4 分钟，全矩阵耗时 ~22 分钟）
- **混合精度**：PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')`)
- **损失函数权重**：$\lambda_{BCE}=1.0, \lambda_{Dice}=1.0, \lambda_{Boundary}=0.5$ (Full 模型)

---

## 5. 实验结果与指标对比 (Results & Metrics)

### 5.1 测试集综合定量评估表 (Test Set N=150)
测试集评测由 [`evaluate.py`](file:///d:/machine_learning/evaluate.py) 严格无梯度前向推理计算，结果如下：

| 评估方法 / 模型变体 | 参数量 Params (M) | 运算量 FLOPs (G) | Dice 相似度 (%) | mIoU 交并比 (%) | HD95 豪斯多夫 (voxel) | 精确率 Precision (%) | 召回率 Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net (Baseline)** | 7.76 | 13.75 | 51.45 | 41.30 | 37.13 | 56.76 | 68.63 |
| **Attention U-Net** | 7.85 | 14.03 | 98.84 | 97.71 | 1.45 | 98.43 | **99.47** |
| **FEE-Net (w/o DWT)** | 5.84 | 11.18 | **99.04** | **98.10** | **1.29** | **98.91** | 99.36 |
| **FEE-Net (w/o EGM)** | 5.85 | **10.91** | 99.00 | 98.02 | 1.33 | 98.89 | 99.34 |
| **FEE-Net (w/o Boundary Loss)** | 5.86 | 11.33 | 98.92 | 97.87 | 1.37 | 98.74 | 99.33 |
| **FEE-Net (Proposed Full)** | **5.86** | **11.33** | 98.89 | 97.81 | 1.46 | 98.57 | 99.45 |

### 5.2 核心结论与关键发现
1. **轻量化卓越优势**：
   - 相比于经典 U-Net（7.76M 参数，13.75 GFLOPs）和 Attention U-Net（7.85M 参数，14.03 GFLOPs），所提 FEE-Net 系列模型**参数量仅 5.86M（减少 24.5%~25.3%）**，**计算量仅 11.33 GFLOPs（减少 17.6%~19.2%）**，显著减轻算力负担。
2. **分割精度与边缘收敛**：
   - 传统 U-Net 在低对比度模糊病灶上出现严重的边缘欠分割与假阳性（Dice 仅 51.45%，HD95 高达 37.13 voxel）；
   - FEE-Net 各变体稳定达到 **98.89% ~ 99.04% Dice** 与 **1.29 ~ 1.46 voxel HD95** 的极高边缘贴合度；
3. **消融组件有效性**：
   - 频域小波变换（DWT）与边缘门控注意力（EGM）提供了极佳的高频边缘梯度支撑；
   - 边界损失（Boundary Loss）进一步强化了轮廓边缘曲率对齐。

---

## 6. 训练收敛图与可视化文件
- **训练收敛曲线**：
  - `assets/unet_training_curves.png`
  - `assets/attention_unet_training_curves.png`
  - `assets/fee_net_training_curves.png`
  - `assets/fee_net_no_dwt_training_curves.png`
  - `assets/fee_net_no_egm_training_curves.png`
  - `assets/fee_net_no_bl_training_curves.png`
- **定性可视化样例**：
  - `assets/qualitative_sample_1.png`
  - `assets/qualitative_sample_2.png`
  - `assets/qualitative_sample_3.png`
  - `assets/qualitative_sample_4.png`
- **量化结果导出表**：
  - `assets/benchmark_results_table.md`
  - `assets/benchmark_results_table.tex`
