---
title: "Exp-033: Kaggle v5 终版全量基准定稿、二次 Sigmoid 根治与分流坍缩深度复盘"
date: 2026-10-03
tags:
  - Experiment_Log
  - Benchmark_Result
  - Kaggle_DualT4
  - Conformal_Risk_Control
  - Root_Cause_Analysis
---

# Exp-033: Kaggle v5 终版全量基准定稿与单一真源落地

> 前序关联：[[Exp_031_Architecture_Aligned_Rerun_Results|Exp-031]]、[[Exp_032_Probe_Matrix_Final_and_v6_Launch|Exp-032]]  
> 运行平台：Kaggle Dual Tesla T4 GPU (16GB), PyTorch 2.10, 60 Epochs  
> 产物溯源：`scratch/kaggle_v5_results/benchmark_metrics_independent_test.csv` (SHA 校验通过)

---

## 1. 实验目的与动机

在 [[Exp_032_Probe_Matrix_Final_and_v6_Launch|Exp-032]] 中，v4 云端运行暴露出两大致命缺陷：
1. **二次 Sigmoid 评估漏洞**：`train_model` 验证循环调用 `model(v_imgs, force_full=True)` 时未传 `return_raw_logits=True`，模型默认返回概率值，而 `MetricTracker(from_logits=True)` 对概率值再次施加 Sigmoid，导致验证选择严重失真，`FEENetEarlyExit` 在独立测试集上仅取得 30.89% Dice 的灾难性误测；
2. **混合队列单元未导入 `subprocess`**：导致运行至 84 分钟在混合队列构建阶段发生 `NameError`，后续图表渲染与结果保存被中断。

**Exp-033 (Kaggle v5)** 的核心目标：在云端双卡 T4 环境下，彻底应用以上两处修复，对 7 个模型（U-Net、Attention U-Net、UNet++、FEE-Net、FEENetEarlyExit、FEE-Net w/o DWT、FEE-Net w/o EGM）执行严格的 60 轮单一真源训练与 150 例独立盲测集无泄漏评测，产出论文定稿数据。

---

## 2. 核心代码修复细节

1. **验证循环 `return_raw_logits=True` 强制对齐**：
   ```python
   # tools/generate_kaggle_all_in_one_notebook.py
   # 修复前：
   v_logits = model(v_imgs, force_full=True)
   # 修复后：
   if is_early_exit(name):
       v_logits = model(v_imgs, force_full=True, return_raw_logits=True)
   else:
       v_logits = model(v_imgs)
   ```
2. **自包含防御性环境导入**：
   在 Cell 2 头部与 Cell 12 混合队列单元分别显式执行 `import subprocess, os, sys`，并将混合队列构建单元置于独立的 `try...except` 保护块中，保障评测结果首要安全落盘。

---

## 3. 150 例独立测试集定稿基准对比 (Table I)

下表数据严格提取自 `scratch/kaggle_v5_results/benchmark_metrics_independent_test.csv`，由 `tools/ingest_kaggle_results.py` 完成 7 行边界完整性与散列校验：

| 模型架构 | Test Dice (%) | Test mIoU (%) | HD95 (px) ↓ | Precision (%) | Recall (%) | Params (M) | FLOPs (G) | Latency (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | **84.14** | 76.45 | 36.33 | 89.34 | 82.60 | 7.763 | 13.75 | 10.02 | 99.8 |
| **Attention U-Net** | 84.00 | 75.83 | 38.73 | 90.21 | 81.43 | 7.852 | 14.03 | 11.57 | 86.5 |
| **UNet++** | 84.07 | **76.57** | 33.38 | 89.67 | 83.23 | 9.160 | 34.62 | 16.92 | 59.1 |
| **FEE-Net (full)** | 79.47 | 70.65 | 38.67 | 86.55 | 75.33 | 8.539 | **12.59** | 15.66 | 63.9 |
| **FEENetEarlyExit** | **81.42** | 73.07 | 36.41 | 86.26 | 78.90 | 8.552 | 12.59 (门控 0.94) | 17.43 | 57.4 |
| **FEE-Net w/o DWT** | 80.78 | 72.73 | 36.11 | 86.83 | 80.19 | 8.546 | 12.87 | 14.35 | 69.7 |
| **FEE-Net w/o EGM** | **83.08** | 75.59 | **31.71** | 87.84 | 82.92 | 8.522 | 11.83 | 11.94 | 83.8 |

---

## 4. 关键科学发现与深度诊断

### 4.1 FEENetEarlyExit 修复成效与多任务正则化
- 修复二次 Sigmoid 之后，FEENetEarlyExit 性能从 30.89% **飙升至 81.42%**（mIoU 73.07%，HD95 36.41 px）。
- **核心现象**：FEENetEarlyExit 甚至较独立训练的单路径 FEE-Net（79.47%）取得了 **+1.95% Dice 增益与 -2.26 px HD95 改善**。
- **理论机理**：分流特征探针（Stop-Gradient + 4C GAP/GMP）在保护骨干密集分割梯度的同时，浅层分类损失迫使共享 stem 与 layer1 学习更具语义区分度的判别性表征，形成了优异的多任务正则化效应。

### 4.2 FEE-Net w/o EGM 优异的边界位置保真度
- 去除方向性八邻域门控的 `FEE-Net w/o EGM` 在 60 轮下取得 **83.08% Dice**，且其 **HD95 达到 31.71 px，为全场 7 个模型中最优**（优于 UNet++ 的 33.38 px 与 U-Net 的 36.33 px），同时 FLOPs 降至 11.83G，吞吐量升至 83.8 FPS。
- 这说明在 $256 \times 256$ 小样本结肠镜任务下，过度复杂的空间方向差分可能带来细微高频噪点敏感性，而小波频域解耦（DWT + SCA）单独配合简洁解码路径表现更具稳健性。

### 4.3 全阳性数据集导致的早退分流头坍缩深度复盘 (重要结论)
在混合队列保形筛查测试中：
- 认证集病灶实测漏诊率 FNR 为 **0.00%**，Clopper-Pearson 理论安全界为 **3.92% $\le 5.0\%$**，保形数学安全完全成立；
- 但正常黏膜旁路率回测为 **0.00%**（未实现期望的算力旁路节约）。
- **根因分析**：
  - Kvasir-SEG 数据集内**每一张帧均包含息肉（全阳性样本集）**；
  - 联合训练阶段，分流头接触的所有 680 张训练样本的标签全部为 1，在无任何阴性（健康黏膜）样本约束下，交叉熵损失的全局最优解即为将病灶风险恒定输出为 $\approx 0.9998$；
  - v3 中曾呈现 64.66% 旁路率的表象，实为检查点处于未充分收敛的瞬态随机波动（分离度仅 0.012），属于不可复现的侥幸现象；
  - **科学结论与解法**：临床筛查门控**必须依赖包含健康黏膜阴性样本的平衡数据微调**。两阶段方案（主干特征与 BN 统计量双重冻结 + 均衡队列微调分流头）已在 [[Exp_034_Triage_Collapse_and_Two_Stage_Fix|Exp-034]] 中彻底落地验证：在保持分割精度逐位不变（81.42%）的前提下，实现 $\tau^*=0.59$、FNR=0.00%（CP 上界 3.92% $\le 5\%$）、**正常黏膜旁路率 99.25%** 与 **1.91× 端到端加速**。

---

## 5. 结论与知识库留档

1. Kaggle v5 基准彻底消除了代码-数据漂移，完成了 7 模型同环境严肃基准测量，数据真实透明；
2. 论文 Manuscript 与 TeX Draft 已全面将 Table I 同步为定稿测试集数据；
3. 本次实验再次彰显了规范实验设计与零泄漏严谨协议对医学影像研究的重要性。
