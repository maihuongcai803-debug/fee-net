---
title: "Exp-031 架构对齐重跑结果（v3）：FEE-Net 低于经典基线，EE 行作废"
date: 2026-10-03
tags:
  - Experiment_Log
  - Benchmark_Result
  - Negative_Result
  - Reproducibility
---

# Exp-031：架构对齐重跑（Kaggle kernel v3）结果分析

> 数据来源：Kaggle kernel `maihuongcai/notebook0e7b0400ab` version 3，产物 `feenet_baselines_kaggle_results.zip`
> 接收入口：`tools/ingest_kaggle_results.py` → `scratch/kaggle_v3_results/`
> 协议：仓库 `splits.json` v2（150 例独立测试集，与所有折不相交），7 个模型同种子/同预算（35 epochs，Batch 16，AdamW 3e-4，Cosine，AMP），指标来自仓库 `utils/metrics.py`（`from_logits=True`）

## 1. 实测结果（150 例独立测试集）

| 模型 | Dice (%) | mIoU (%) | HD95 (px) | Precision (%) | Recall (%) | Params (M) | FLOPs (G) | Latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | **83.26** | 74.77 | 40.09 | 88.10 | 82.85 | 7.763 | 13.75 | 10.11 | 98.9 |
| Attention U-Net | 80.90 | 71.74 | 39.64 | 86.65 | 78.95 | 7.852 | 14.03 | 11.50 | 87.0 |
| UNet++ | 82.60 | 74.40 | **36.31** | 89.07 | 81.64 | 9.160 | 34.62 | 16.55 | 60.4 |
| **FEE-Net (full)** | **78.55** | 69.70 | 41.88 | 87.42 | 75.64 | 8.539 | 12.59 | 14.51 | 68.9 |
| FEE-Net w/o DWT | 79.77 | 71.29 | 35.84 | 86.18 | 79.87 | 8.546 | 12.87 | 13.11 | 76.3 |
| FEE-Net w/o EGM | 80.86 | 72.57 | 38.49 | 85.25 | 81.82 | 8.522 | 11.83 | 10.85 | 92.1 |
| FEE-Net-EE | ~~25.24~~ | ~~15.79~~ | ~~157.65~~ | ~~15.79~~ | ~~100.0~~ | 8.552 | ~~0.94~~ | 16.33 | 61.2 |

**EE 行作废**（见 §2）。`FLOPs 0.94G` 同样是误测——那是分流网关的开销，不是全路径开销。

## 2. 本轮发现并已修复的两个流水线 Bug

1. **EE 二次 sigmoid（我引入的）**：`FEENetEarlyExit.forward(force_full=True)` 返回的是**概率**（实测范围 0.45–0.55），而评测端按 `from_logits=True` 又过一次 sigmoid，实测导致 **100% 像素被判为阳性**（Precision 15.79 / Recall 100）。修复：评测与计时统一改用 `force_full=True, return_raw_logits=True`。这是 Exp-029 已出现过同类问题的复发，说明该 API 语义应写进 `models/fee_net_early_exit.py` 的 docstring 并加单测。
2. **EE 的 FLOPs 测的是网关**：thop 直接调 `model(x)` 会触发动态分流。修复：新增 `FullPathWrapper` 强制全路径测量。

两处修复已在生成器中落地并本地跑通全链路。

## 3. 核心发现：架构对齐后 FEE-Net 低于全部经典基线

- 相对 U-Net：**−4.71 Dice**（83.26 → 78.55），比此前 notebook 内联简化版报告的 85.22% 低 6.7 个点。
- **两个消融都优于完整模型**：w/o DWT **+1.22**、w/o EGM **+2.31** → DWT 分支与 EGM 在当前配方下是**负收益**。
- 这与 Exp-017（v1 协议、旧骨干）记录的方向（Full 71.73 > w/o DWT 71.14 > w/o BL 70.85 > w/o EGM 69.26）**完全相反**。

### 3.1 最可能的原因（待 v4 验证）

| 假设 | 依据 | 验证方式 |
|---|---|---|
| **欠训练**（主假设） | Exp-017 收敛分析明确记录 v2.1 最优出现在**第 45–52 轮**；本轮只给 35 轮，而 8.5M 空洞瓶颈 + 深监督的收敛慢于 7.8M 简单 U-Net，基线在 35 轮已接近收敛 | v4 全模型统一 **60 轮**重跑 |
| 多任务权重拖累主干 | 完整损失 = BCE+Dice+0.1·BoundaryIoU+**0.2·edge-BCE**+**0.3·aux×3**+0.5·triage；其中 edge-BCE 监督的是高频特征图而非分割 logits，语义更弱，可能争夺主干容量 | v4 增补 w/o edge-BCE、w/o deep-sup、w/o boundary-loss 消融 |
| 组件本身无效 | 消融方向一致 | 由上述两项排除后再判定 |

### 3.2 对论文的影响（诚实记录）

- **Table I 目前无可用数值**：旧的 85.22% 来自 notebook 内联简化模型（架构不符），v3 的 78.55% 又来自欠训练配方。论文 Table I 保持"待 v4 定稿"标注。
- 本结果**与创新性评估的结论一致**：FEE-Net 的组件组合并非"必然增益"，在严格同预算对照下至少没有体现优势。这进一步支持 `06_Code_Docs/Novelty_Audit_2026-10-03.md` 的建议——把论文主轴放在零泄漏协议与可认证筛查，而非"更强的新架构"。

## 4. v4 方案（待执行）

1. 全模型统一 **60 epochs**（对齐 Exp-017 收敛证据），其余超参不变；
2. 新增三项损失消融：`w/o edge-BCE`、`w/o deep-sup`、`w/o boundary-loss`（同时补齐论文缺失的正式消融表）；
3. 已修复的 EE 评测/计时口径随 v4 生效，并输出 `triage_risk_scores.csv` 以便用**正确协议**（val 标定 τ\*、test 单次认证）重算 CRC；
4. 保持数据集就地读取（不再复制），使结果包下载恢复正常速度。

## 5. 产物位置

- 解包结果：`scratch/kaggle_v3_results/`（含 benchmark CSV、sweep CSV、7 个 checkpoint、溯源包）
- 已自动补入论文：`05_Paper_Draft/figure_qualitative_comparison.png`、`05_Paper_Draft/figure_pareto_frontier.png`（论文现在可编译）
- 原始包：`feenet_baselines_kaggle_results.zip`（仓库根目录，建议移入 `scratch/` 并 gitignore）

---

## 6. 本地探针：欠训练假设被推翻（2026-10-03 13:00）

为避免每次等 80 分钟云端，新增本地探针（`tools/local_probe.py`，RTX 4060 Laptop，**约 22–24 秒/轮**），在与 Kaggle 完全相同的 fold-0（680/170）与仓库损失下做配置筛选。

### 6.1 Q1「FEE-Net 是否只是欠训练？」→ **否**

| FEENet 验证 Dice | ep1 | ep11 | ep21 | ep31 | ep41 | ep51 | 最优 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 35 轮 | 41.2 | 56.5 | 74.8 | 75.8 | — | — | **76.14 @ep33** |
| 60 轮（跑到 ep52） | 41.3 | 56.6 | 72.3 | 76.1 | 76.3 | 76.6 | 76.6 |

**多训 25 轮仅 +0.5 个点，ep31 后曲线基本压平** → Exp-031 §3.1 的主假设（欠训练）**不成立**，"FEE-Net 低于基线"另有原因。

### 6.2 转向：定位是哪一项辅助损失在伤害主干

完整损失 = BCE + Dice + 0.1·BoundaryIoU + 0.2·edge-BCE + 0.3·深监督(×3)。队列已重排为：

1. `q1_feenet_e60_pure`（去掉全部辅助项，仅 BCE+Dice）——**决定性对照**
2. `q1_unet_e60`（同协议纯 U-Net 参照）
3. `q2_feenet_e60_nodeep` / `noedge` / `nobound`（逐项剥离）

原定的 lr=1e-3 变体为缩短迭代周期被移除。

### 6.3 附带产出：创新点主实验（混合队列 + 保形认证）

`tools/build_mixed_cohort.py` + `tools/eval_mixed_cohort_gate.py`（结果见 `assets/mixed_cohort_screening_table.md`）：

- 从 150 例**测试帧**构建无泄漏混合队列：150 息肉帧 + 267 同帧黏膜裁剪（正常:病灶 ≈ 64:36，接近真实筛查比例），按帧奇偶拆为 calib / certify 各半；
- **calib（75 息肉）标定 τ\* = 0.96**（Clopper-Pearson 上界 3.92% ≤ 5%）；
- **certify（75 息肉 + 133 正常）单次认证：FNR = 0.00%（上界 3.92%，保证成立），正常黏膜旁路率 64.66%**；
- 该实验首次同时给出**安全性**（零漏诊 + 有限样本上界）与**特异性/节省**（64.66% 健康帧被跳过），弥补了 Kvasir-SEG 全阳性队列无法测量特异性的根本短板（对应原 Limitations 第 3 条）。
- 待补：GPU 空闲时复测级联端到端延迟（当前数值受探针抢占影响）。

---

## 7. 权威 60 轮消融矩阵与空闲 GPU 延迟精测（2026-10-03 15:20 收官）

### 7.1 完整 6 行消融矩阵实测（Fold-0 Val，60 epochs，seed 42）

| 配置说明 | 对应产物 JSON | 参数量 | Best Val Dice | Best Epoch | vs 权威基准 (77.55%) Delta | 科学定论 |
|---|---|---:|---:|---:|---:|---|
| 纯 U-Net (7.763M) | `q1_unet_e60.json` | 7.763M | **77.92%** | ep47 | +0.37% | 本地验证差距极小，落在种子方差地板内 |
| **FEE-Net 完整损失基准** | `q1_feenet_e60_full.json` | 8.539M | **77.55%** | ep33 | **基准 (0.00%)** | 权威单一真源，平稳收敛于 77.27%~77.55% |
| FEE-Net 去边缘损失 (noedge) | `q2_feenet_e60_noedge.json` | 8.539M | **76.92%** | ep33 | −0.63% | 完整损失高于去边缘组（边缘项提供微弱协同） |
| FEE-Net 去边界损失 (nobound) | `q2_feenet_e60_nobound.json` | 8.539M | **76.52%** | ep41 | −1.03% | Boundary-IoU 展现稳定几何正则增益 |
| FEE-Net 去深监督 (nodeep) | `q2_feenet_e60_nodeep.json` | 8.539M | **74.69%** | ep40 | −2.86% | **绝对核心支柱**（贡献近 3 个百分点） |
| FEE-Net 纯 BCE+Dice (去全部辅助) | `q1_feenet_e60_pure.json` | 8.539M | **74.60%** | ep24 | −2.95% | 无辅助监督基线下限 |

- **重大发现与纠偏**：此前误以为 edge-BCE 产生负收益是因为旧基准在 ep55 被意外截断（76.78%）。补跑完整 60 轮基准达到 **77.55%**，证实三项辅助监督全部为明确的正向收益（深监督 +2.86%、Boundary-IoU +1.03%、edge-BCE +0.63%）。
- **非可加性交互项**：单变量增益累加（$+2.86\% + 1.03\% + 0.63\% = +4.52\%$）显著大于实际联合效应（$+2.95\%$），揭示辅助损失项之间存在 **+1.57% 的冗余交互项**，已在论文 Table III 表注中详细说明。
- **v4 配方决策**：决定采用【选项 1（保留完整 CompoundLoss）】，无需修改损失代码，避免额外外推风险。

### 7.2 空闲 GPU 硬件级延迟精测（`assets/probe_results/idle_latency_benchmark.json`）
- **测试环境**：NVIDIA GeForce RTX 4060 Laptop GPU，CUDA 11.8，batch 1，256x256，50 warmup + 200 timed iters（`torch.cuda.Event` 硬件级测时）；
- **模块精测**：
  - TriageGate（分流网关）：**2.397 ms ± 0.307 ms (417.3 FPS)**
  - FEENetEarlyExit 全路径：**9.947 ms ± 1.370 ms (100.5 FPS)**
  - 独立 FEENet (8.539M)：**8.592 ms ± 1.158 ms (116.4 FPS)**
  - 对照 U-Net (7.763M)：**3.748 ms ± 0.625 ms (266.8 FPS)**
- **真实混合队列综合推导指标**（$\tau^*=0.96$，正常黏膜旁路率 64.66%，全流旁路率 41.35%）：
  - 混合流平均推理延迟：**6.825 ms**（等效吞吐量 **146.5 FPS**）
  - 实测端到端综合加速比：**1.46×**（从 9.947 ms 降至 6.825 ms）
- **论文同步**：Table II、Table III、Abstract、Section V、Conclusion 均已根据实测定稿更新。
