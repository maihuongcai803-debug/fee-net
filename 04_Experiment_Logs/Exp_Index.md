---
title: 实验总览与版本迭代历史矩阵
date: 2026-10-01
tags:
  - Experiment
  - Tracking
  - MOC
  - KnowledgeGraph
---

# 📊 实验总览与版本生命周期管理矩阵 (Experiment Lifecycle Matrix)

> **最新整改日期**：2026-10-01  
> **核心归档索引**：[[04_Experiment_Logs/Exp_Archive_Legacy_Prototypes|📦 历史实验与已废弃原型归档汇总]]  
> **全项目深度审查**：[[06_Code_Docs/Code_Review_2026-10-01|审查报告 (Code & Science Review)]]  
> **数据协议最新基准**：[[04_Experiment_Logs/Exp_027_Codebase_Hygiene_and_Independent_Test_Protocol|Exp-027 v2 splits 独立测试协议]]

> [!NOTE]
> **路径迁移说明（2026-10-26）**：本目录下 Exp-001 ~ Exp-034 为**历史快照**，其中出现的脚本路径按当时结构记录，未回改。现行对应关系：
> `eval_laya_jev_cascade.py` → `laya/eval_cascade.py` ｜ `train_laya_triage.py` → `laya/train_laya_triage.py`
> ｜ `models/modules/laya_*.py` → `laya/modules/laya_*.py` ｜ `tools/laya_router.py` → `laya/router.py`
> ｜ `eval_dual_system.py` / `run_*.py` → `experiments/`。详见 [[06_Code_Docs/Project_Architecture|架构说明]]。

---

## 🚦 实验数据可用性红绿灯规范 (Paper Writing Data Rules)

在撰写最终论文、提交审稿或汇报进展时，必须严格遵守以下数据引用规则，彻底杜绝学术硬伤：

* 🟢 **绿色（直接可用 / 核心论文支柱）**：
  * **网络架构与损失函数**：[[Exp_016_FEE_Net_v2_1_Bilinear_Depthwise_and_DeepSupervision|Exp-016]] (FEE-Net v2.1 双线性+深度可分离+深监督)、[[Exp_028_DualBranch_Triage_and_Spectral_Wavelet_Overhaul|Exp-028]] (双分支极值分流头 + 频域通道注意力 SCA + 八邻域全景小波滤波 ODWF) 与 [[Exp_012_Boundary_IoU_Loss_Overhaul|Exp-012]] (Boundary IoU Loss)。
  * **早退架构与梯度解耦**：[[Exp_025_FEENet_EarlyExit_Pipeline_Integration_and_Gradient_Decoupling|Exp-025]] (`FEENetEarlyExit` 共享骨干 + Stop-Gradient 梯度解耦) 与 [[Exp_028_DualBranch_Triage_and_Spectral_Wavelet_Overhaul|Exp-028]] (双分支极值池化安全早退)。
  * **统计风控与安全保障**：[[Exp_024_Root_Purification_and_Conformal_Risk_Control|Exp-024]] 与 [[Exp_026_Comprehensive_Practical_Verification_and_CRC_Calibration|Exp-026]] (有限样本保形风险控制 CRC，严格证明 $FNR \le 4.87\%$)。
  * **评测与数据协议**：[[Exp_027_Codebase_Hygiene_and_Independent_Test_Protocol|Exp-027]] (v2 splits：150 例全局独立测试集 + 850 例 5 折分层 CV)。
* 🟡 **黄色（算法组件有效，但数值基于旧版 v1 划分，待 v2 重新标定）**：
  * [[Exp_007_EGM_Residual_Gate_and_Dual_Return|Exp-007]] (EGM 阻尼门控)、[[Exp_008_UNetPlusPlus_Baseline|Exp-008]] (UNet++ 基线)、[[Exp_014_BERCA_Directional_Wavelet_and_Capacity_Expansion|Exp-014]] (BERCA 小波滤波)、[[Exp_017_Genuine_5Fold_Cross_Validation_Results|Exp-017]] (60轮收敛性证明)。
  * *使用规范*：证明算法收敛性与各模块有效性成立；但其绝对 Dice/mIoU 数值属旧版 `splits_v1_backup.json`（test=val），需在 Kaggle 以 v2 splits 重跑出最终独立盲测指标后替换。
* 🔴 **红色（已废弃 / 玩具数据 / 严禁写入最终论文）**：
  * **合成数据集指标**：[[Exp_001_FEENet_Architecture|Exp-001]] ~ [[Exp_005_Dual_System_Laya_FEENet|Exp-005]]（基于 `data/synthetic_medseg`，Dice 98.89% 属玩具数据）。
  * **概念通胀术语**：[[Exp_019_Laya_v2_Jev_Structured_Triage_Plan|Exp-019]] 与 [[Exp_020_Laya_Jev_Cascade_Benchmark|Exp-020]]（“Jev 启发”、“非自回归结构化分流”已被 Exp-024 批判剔除并规范化）。

---

## 📈 实验演进全景矩阵 (按生命周期分类)

### 阶段一：早期概念验证与合成数据原型（已废弃归档 🔴）
> 详见 [[04_Experiment_Logs/Exp_Archive_Legacy_Prototypes|历史原型归档汇总]]，本阶段数据一律不得出现在最终论文中。

| 实验编号 | 实验代号 | 核心修改 / 原型特性 | 历史记录指标 | 状态 | 详细留档笔记 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Exp-000** | Template | 标准留档格式定义 | - | 📋 模板规范 | [[Template_Experiment_Log]] |
| **Exp-001** | `FEENet_Arch` | FEE-Net 初构（5.86M，无阻尼，拉普拉斯边界损失） | 5.86M 参数 | 🔴 已废弃 | [[Exp_001_FEENet_Architecture]] |
| **Exp-002** | `Baselines_Pipeline` | U-Net、Att-UNet 基线与拟真合成数据集构建 | 管道调通 | 🔴 已废弃 | [[Exp_002_Baselines_and_Dataset_Pipeline]] |
| **Exp-003** | `Laya_System1_Routing` | 早期 NLP/规则路由与外挂 System 1 探索 | 30ms 文本路由 | 🔴 已废弃 | [[Exp_003_Laya_Integration_and_System1_Routing]] |
| **Exp-004** | `Benchmark_Ablation` | 6 变体在合成数据上的基准与消融对比 | Dice 98.89% (玩具指标) | 🔴 严禁引用 | [[Exp_004_Benchmark_and_Ablation_Results]] |
| **Exp-005** | `Dual_System_Triage` | 模拟 500 例队列双系统初筛与协同分割 | 算力省 78.2% (仿真) | 🔴 已废弃 | [[Exp_005_Dual_System_Laya_FEENet]] |

---

### 阶段二：严谨同行评审审计与真实数据集迁移（算法有效 / 数据属 v1 协议 🟡）
> 确立了真实数据集与 5 折交叉验证，但受限于当时的 v1 划分（test=val），数据需在新版 v2 splits 下重标。

| 实验编号 | 实验代号 | 核心修改 / 模型特性 | 核心贡献与理论价值 | 状态 | 详细留档笔记 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Exp-006** | `Review_Critique_Plan`| 严谨同行评审意见审计 (2致命+6重大缺陷) 与系统改进方案 | 奠定科学重构路线图 | 📜 纲领文献 | [[Exp_006_Reviewer_Critique_and_Improvement_Plan]] |
| **Exp-007** | `EGM_Residual_Gate` | EGM 可学习残差门控阻尼 ($\gamma=0.1$) + 前向双模输出支持 | 解决深层梯度弥散与震荡 | 🟡 算法有效 | [[Exp_007_EGM_Residual_Gate_and_Dual_Return]] |
| **Exp-008** | `UNetPlusPlus_Baseline` | UNet++ 密集嵌套跳跃连接 SOTA 医疗分割基线实现 | 9.16M / 34.62G 对比基准 | 🟡 算法有效 | [[Exp_008_UNetPlusPlus_Baseline]] |
| **Exp-009** | `Kvasir_SEG_5Fold_CV` | 真实 Kvasir-SEG 接入与初代 5 折分层交叉验证流水线 | 告别合成数据走向真实临床 | 🟡 协议属 v1 | [[Exp_009_Kvasir_SEG_Real_Dataset_and_5Fold_CV]] |
| **Exp-010** | `Laya_Visual_Triage_S1`| 真实非自回归 System 1 视觉初筛网络与非对称安全校准 | 0.048M 移动端轻量前置网络 | 🟡 启发式门禁 | [[Exp_010_Laya_Visual_Triage_System1]] |
| **Exp-011** | `Kvasir_SEG_Benchmark`| Kvasir-SEG 真实全量基准重训与 Wilcoxon 统计显著性评测 | 引入非参数秩和检验 ($p<0.001$) | 🟡 协议属 v1 | [[Exp_011_Real_Kvasir_SEG_Benchmark_and_Significance]] |
| **Exp-012** | `Boundary_IoU_Loss_Overhaul` | 彻底移除拉普拉斯噪声放大，升级 CVPR 2021 BoundaryIoULoss | 梯度平滑降噪 52 倍 | 🟢 现行有效 | [[Exp_012_Boundary_IoU_Loss_Overhaul]] |
| **Exp-013** | `Rigorous_MultiFold_MultiSeed`| 严谨多折多种子自动化训练与出版级 LaTeX 导出框架 | 确立跨折与跨种子统计框架 | 🟡 框架有效 | [[Exp_013_Rigorous_MultiFold_MultiSeed_Evaluation_Framework]] |
| **Exp-014** | `BERCA_Capacity_Expansion` | 基础通道扩至 48 (6.89M) + BERCA 双向协同 (DWF 1x3/3x1 滤镜) | 眩光滤除与边界导引 | 🟡 架构基础 | [[Exp_014_BERCA_Directional_Wavelet_and_Capacity_Expansion]] |
| **Exp-015** | `Genuine_Kvasir_Laya_Triage` | 100% 真实 Kvasir-SEG 内镜黏膜初筛重训与黑边捷径剔除 | 消除内镜黑框作弊特征 | 🟡 数据清洗有效 | [[Exp_015_Genuine_Kvasir_SEG_Laya_Triage]] |

---

### 阶段三：架构升级至 FEE-Net v2.1 与 Kaggle 全基准部署（架构核心 🟢 / 待 v2 标定 🟡）

| 实验编号 | 实验代号 | 核心修改 / 模型特性 | 核心贡献与理论价值 | 状态 | 详细留档笔记 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Exp-016** | `FEE_Net_v2_1_Bilinear_DeepSup` | 双线性插值平替反卷积 + DWF 深度可分离化 + 训练期多尺度深监督 | **FEE-Net 当前最新核心架构** (6.49M/8.52M) | 🟢 现行有效 | [[Exp_016_FEE_Net_v2_1_Bilinear_Depthwise_and_DeepSupervision]] |
| **Exp-017** | `Genuine_5Fold_Cross_Validation` | 真实 Kvasir-SEG 全 5 折评测与 60 轮长周期收敛性证明 | 证实 60 轮平坦收敛区，胜出 UNet++ | 🟡 待 v2 标定 | [[Exp_017_Genuine_5Fold_Cross_Validation_Results]] |
| **Exp-018** | `All_Baselines_Ablations_60Epochs` | 全量基线与消融变体 60 轮 5 折交叉验证统一部署方案 | 规范 6 大模型 30 个 Run 对齐 | 🟡 方案待跑 v2 | [[Exp_018_Kaggle_All_Baselines_and_Ablations_60Epochs_Plan]] |

---

### 阶段四：初筛系统去伪存真、CRC 保形风控与共享主干早退（现行核心 🟢 / 概念通胀已废 🟠）

| 实验编号 | 实验代号 | 核心修改 / 模型特性 | 核心贡献与理论价值 | 状态 | 详细留档笔记 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Exp-019** | `Laya_v2_Jev_Structured_Triage` | Laya v2.0 Jev 非自回归结构化分流规划 | 概念通胀，已被 Exp-024 批判剔除 | 🟠 概念已废弃 | [[Exp_019_Laya_v2_Jev_Structured_Triage_Plan]] |
| **Exp-020** | `Laya_Cascade_Router_Verification` | 独立路由引擎验证 | 独立双网络已被 FEENetEarlyExit 共享主干取代 | 🟠 架构已升级 | [[Exp_020_Laya_Cascade_Router_Verification]] |
| **Exp-020** | `Laya_Jev_Cascade_Benchmark` | 500 例队列三级级联评测 (Jev 包装版) | 启发式门禁已被 CRC 保形风控取代 | 🟠 概念已废弃 | [[Exp_020_Laya_Jev_Cascade_Benchmark]] |
| **Exp-021** | `Codebase_Rigorous_Refactoring` | 全工程深度重构、学术硬伤排雷、张量批处理并行化 | 满血 GPU 批处理，消除重复 | 🟢 现行有效 | [[Exp_021_Codebase_Rigorous_Refactoring]] |
| **Exp-022** | `Deep_Fusion_and_Statistical_Rigor` | 早期共享主干探索与 Wilson CI 统计严谨化 | 引入二项置信区间 | 🟡 阶段成果 | [[Exp_022_Deep_Fusion_Early_Exit_and_Statistical_Rigor]] |
| **Exp-023** | `Purify_Architecture_and_Physics_Augmentation` | 频域门控物理几何修正 (3x3 Depthwise) 与内窥镜物理反光增强 | 8 邻域几何覆盖，反光数据增强 | 🟢 现行有效 | [[Exp_023_Purify_Architecture_and_Physics_Augmentation]] |
| **Exp-024** | `Root_Purification_and_Conformal_Risk_Control` | **根源去伪存真 (剔除 Jev) 与保形风险控制 (CRC) 严谨数学保障** | **Clopper-Pearson 精确证明 $FNR \le \alpha$** | 🟢 现行核心 | [[Exp_024_Root_Purification_and_Conformal_Risk_Control]] |
| **Exp-025** | `FEENet_EarlyExit_Pipeline_and_Gradient_Decoupling` | **FEENetEarlyExit 共享骨干早退闭环与 Stop-Gradient 梯度解耦** | **架构合一，零负迁移，22 项单测全绿** | 🟢 现行核心 | [[Exp_025_FEENet_EarlyExit_Pipeline_Integration_and_Gradient_Decoupling]] |
| **Exp-026** | `Practical_Verification_and_CRC_Calibration` | **全流程实践验证、CRC 实证标定与四大范式决战基准** | **严格 $FNR \le 4.87\%$，303 FPS 超高吞吐** | 🟢 现行核心 | [[Exp_026_Comprehensive_Practical_Verification_and_CRC_Calibration]] |

---

### 阶段五：代码工程卫生、独立测试集协议与架构全景升级（当前最新规范 🟢）

| 实验编号 | 实验代号 | 核心修改 / 模型特性 | 核心贡献与理论价值 | 状态 | 详细留档笔记 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Exp-027** | `Codebase_Hygiene_and_Test_Protocol` | **全工程代码卫生整改与独立测试集协议 (v2 splits / 自检修复 / Git 瘦身)** | **150例零泄漏独立Test，Git 458MB→26MB，自检全绿** | 🟢 最新基准 | [[Exp_027_Codebase_Hygiene_and_Independent_Test_Protocol]] |
| **Exp-028** | `DualBranch_Triage_and_Spectral_Wavelet_Overhaul` | **双分支极值分流头 (GAP+GMP) + 频域高频注意力 (SCA) + 八邻域全景小波滤波 (ODWF)** | **阻断微小病灶漏诊风险，抑制镜面反光杂波，消除对角边缘盲区** | 🟢 最新架构 | [[Exp_028_DualBranch_Triage_and_Spectral_Wavelet_Overhaul]] |
| **Exp-029** | `Kaggle_DualT4_CPU_Spike_Diagnosis_and_Model_Bugfixes` | **Kaggle 188% CPU 占用激增诊断、U-Net 解码器通道修正与双卡零争抢直推云端** | **单线程硬锁定根除 CPU 激增；初版原型 150 例独立测试验证** | 🟡 原型已归档 | [[Exp_029_Kaggle_DualT4_CPU_Spike_Diagnosis_and_Model_Bugfixes]] |
| **Exp-030** | `Architecture_Alignment_and_Single_Source_Benchmark` | **单一真源 Benchmark 流水线：内联导入仓库 models/utils，补齐深监督与边缘监督** | **杜绝内联简化模型，实现 100% 仓库源码与 Kaggle 评测对齐** | 🟢 现行基准 | [[Exp_030_Architecture_Alignment_and_Single_Source_Benchmark]] |
| **Exp-031** | `Architecture_Aligned_Rerun_Results` | **架构对齐重跑实测（78.55%）、消融探针矩阵排查与真实混合队列保形认证** | **推翻欠训练假设；探针矩阵证实深监督(+2.86%)与边缘辅助正向收益** | 🟢 现行核心 | [[Exp_031_Architecture_Aligned_Rerun_Results]] |
| **Exp-032** | `Probe_Matrix_Final_and_v6_Launch` | **本地 60 轮消融探针矩阵定稿、非可加性交互项(+1.57%)发现与 v4 事故复盘** | **FEE-Net 验证集与 U-Net 持平(-0.37%)且 HD95 显著更优(40.14 vs 50.11 px)；排查二次 Sigmoid 隐患** | 🟢 现行核心 | [[Exp_032_Probe_Matrix_Final_and_v6_Launch]] |
| **Exp-033** | `Kaggle_v5_Final_Benchmark` | **Kaggle v5 终版基准定稿、二次 Sigmoid 根治、7模型权威落地与分流坍缩深度复盘** | **FEENetEarlyExit 恢复至 81.42%(+1.95%)；FEE-Net w/o EGM 获全场最优 31.71 px HD95；全阳性数据集分流坍缩诊断** | 🟢 现行定稿 | [[Exp_033_Kaggle_v5_Final_Benchmark]] |
| **Exp-034** | `Triage_Collapse_and_Two_Stage_Fix` | **分流筛查头坍缩根因（全阳性训练集）+ 三陷阱修复 + 两阶段训练** | **分割零代价（81.42 逐位不变）；τ*=0.59；认证 FNR 0.00%（CP 上界 3.92%）；正常黏膜旁路 99.25%；实测 1.91×** | 🟢 现行定稿 | [[Exp_034_Triage_Collapse_and_Two_Stage_Fix]] |

---

## 🎯 论文终极汇报对照目标 (Publication Targets)

1. **核心分割网络 (FEE-Net 架构与轻量化)**：
   - 参数量：8.539M，与基础 U-Net (7.763M) 处于同等紧凑量级，且显著低于 UNet++ (9.160M)；
   - 评测基准：在 **150 例全局独立盲测集** 上基于单一真源打包严格汇报无数据泄漏的真实指标；
   - 战略转型：放弃虚高 SOTA 声称，如实呈现有限样本下频域架构的归纳偏置特性与辅助损失正贡献。
2. **有限样本保形安全筛查分流系统 (FEE-Net + CRC Triage)**：
   - 共享 Encoder 骨干早退流水线，辅以 Stop-Gradient 梯度解耦；
   - 安全门禁：基于严格分离的校准与认证范式，采用 Clopper-Pearson 有限样本二项置信上界，在 95% 置信度下数学证明 $\mathrm{FNR} \le 3.92\% < 5.0\%$，实测 $\mathrm{FNR} = 0.00\%$；
   - 筛查吞吐加速：在 417 例混合队列中安全旁路 **64.66%** 的健康黏膜帧，取得 $1.46\times$ 的端到端硬件推理加速（有效平均延迟 6.83 ms，吞吐量 146.5 FPS）。
