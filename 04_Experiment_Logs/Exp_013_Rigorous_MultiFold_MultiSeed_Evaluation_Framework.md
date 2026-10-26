---
title: 实验留档 Exp_013 严谨多折多种子训练与 Wilcoxon 统计评测体系建设
date: 2026-09-22
tags:
  - Experiment
  - Automation
  - MultiFold
  - MultiSeed
  - StatisticalSignificance
  - WilcoxonSignedRank
---

# 🧪 实验留档：Exp_013_Rigorous_MultiFold_MultiSeed_Evaluation_Framework

## 1. 实验基本信息
- **实验日期**：2026-09-22
- **实验编号**：Exp-013
- **执行角色**：Experiment Automation Engineer
- **运行平台**：NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) | PyTorch AMP
- **评估数据集**：Simula Official Kvasir-SEG (1000 例真实临床内窥镜息肉图像及金标准掩模)
- **关联文档**：
  - [[00_Index|主索引]]
  - [[04_Experiment_Logs/Exp_Index|实验矩阵索引]]
  - [[04_Experiment_Logs/Exp_009_Kvasir_SEG_Real_Dataset_and_5Fold_CV|Exp-009 Kvasir-SEG 5折分层切分]]
  - [[04_Experiment_Logs/Exp_011_Real_Kvasir_SEG_Benchmark_and_Significance|Exp-011 真实数据基准初测]]
  - [[05_Paper_Draft/Draft_IEEE_FEE_Net.tex|IEEE 论文草稿]]

---

## 2. 实验目的与修改动机
为彻底满足 IEEE TMI / MICCAI 顶级医学图像计算期刊对**严格可复现性（Reproducibility）**与**严谨统计学验证（Statistical Rigor）**的标准，构建自动化多折（5-Fold Stratified Cross-Validation）与多随机种子（Multi-Seed: 42, 123, 2024）的训练调度与统计检验框架：
1. **重构训练流水线 (`train.py`)**：全面接入官方真实数据加载器 `datasets.real_medical_loader.get_kvasir_dataloaders`，支持统一随机种子固定（`random`、`numpy`、`torch`、`cudnn.deterministic`），支持 `--epochs 50` 与 `--patience 10` 早停；规范多折与多随机种子命名模型权重存档（`{model}_fold{fold}_seed{seed}_best.pth`），同时保留基线兼容性。
2. **升级全量自动化基准训练脚本 (`run_real_benchmark.py`)**：支持 CLI 灵活指定 `--folds 0..4` 与 `--seeds 42 123 2024`，实现 7 种网络架构自动化批量调度训练与执行记录，并无缝对接统计学评测。
3. **建立顶级期刊标准评测框架 (`run_rigorous_evaluation.py`)**：
   - 自动发现与解析多折/多随机种子模型权重；
   - 计算真实 **Run-level Mean ± Std**（多折/多种子宏观平均与标准差）与样本级分布；
   - 执行配对双侧 **Wilcoxon 符号秩检验**（Paired Two-sided Wilcoxon Signed-Rank Test）；
   - 一键生成符合 Booktabs 规范的出版级 LaTeX 表格（含准确科学计数法 $p$ 值与显著性星号标定 $^{*}/^{**}/^{***}$）与 Markdown 详细评测报告。

---

## 3. 涉及文件与核心代码改动
1. **`datasets/real_medical_loader.py`**：
   - 修复了数据增强转换返回值解包逻辑，同时兼容元组型 `MedicalCompose` 与字典型 `Albumentations` 接口；
   - 补充了 `create_kvasir_stratified_splits`、`download_or_generate_kvasir_seg` 与 `KvasirDataLoaderBundle` 等向后兼容别名，确保全局模块平滑调用。
2. **`train.py`**：
   - 新增 `set_seed()` 函数，实现全栈确定性随机种子初始化；
   - 完善 CLI 参数：`--epochs` (默认 50), `--patience` (默认 10), `--fold` (限制 `0..4`), `--seed` (支持多随机种子输入)；
   - 增强权重与训练日志命名机制：支持 `{model}_fold{fold}_seed{seed}_best.pth` 与 `history.json`，并为 `fold=0, seed=42` 维护传统引用别名。
3. **`run_real_benchmark.py`**：
   - 使用 `argparse` 进行全面参数化改造，支持 `--folds`、`--seeds`、`--models`、`--skip_existing` 以及 `--eval_after`；
   - 批量训练后自动调用 `run_rigorous_evaluation.py` 执行端到端统计检验。
4. **`run_rigorous_evaluation.py` (全新建立)**：
   - 全面支持 7 种医学分割模型（U-Net, Attention U-Net, UNet++, FEE-Net 及其 3 种消融变体）；
   - 支持多折数据加载器缓存管理；
   - 准确计算参数量 (M) 与 FLOPs (G)；
   - 针对多 Run 计算真实 Run-level 统计量，针对单 Run 计算样本级统计量；
   - 实现非参数配对 Wilcoxon 符号秩检验；
   - 导出 `assets/rigorous_evaluation_table.tex`、`assets/rigorous_evaluation_table.md`、`assets/rigorous_evaluation_metrics.json` 与 `assets/rigorous_evaluation_report.md`。

---

## 4. 超参数与配置
- **默认训练轮次 (Epochs)**：50 (Early Stopping: Patience = 10)
- **学习率与优化器**：AdamW (lr=3e-4, weight_decay=1e-4)
- **调度器**：CosineAnnealingLR (T_max=50, $\eta_{min}=10^{-6}$)
- **交叉验证折数**：Fold 0, 1, 2, 3, 4 (每折训练集 800 例，验证/测试集 200 例，严格零重叠)
- **随机种子配置**：42, 123, 2024
- **输入尺寸与批大小**：$256 \times 256$, Batch Size = 16

---

## 5. 框架功能自测与基准评测验证
在公认 Kvasir-SEG 数据集（Fold 0, Seed 42, $N=200$ 测试用例）上运行 `run_rigorous_evaluation.py` 验证端到端流水线，生成如下出版级定量对比表：

### 5.1 Markdown 评测对比表
| Method / Architecture | Params (M) | FLOPs (G) | Dice (%) | mIoU (%) | HD95 (voxel) | Wilcoxon p-val | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net (Ronneberger et al.)** | 7.76 | 13.75 | 31.48 $\pm$ 21.35 | 20.85 $\pm$ 17.35 | 122.39 $\pm$ 35.16 | $1.21 \times 10^{-16}$ | $p < 0.001$ ($^{***}$) |
| **Attention U-Net (Oktay et al.)** | 7.85 | 14.03 | 32.38 $\pm$ 21.70 | 21.55 $\pm$ 17.56 | 119.22 $\pm$ 34.26 | 0.0047 | $p = 0.0047$ ($^{**}$) |
| **UNet++ (Nested SOTA)** | 9.16 | 34.62 | 32.05 $\pm$ 22.77 | 21.56 $\pm$ 18.55 | 118.60 $\pm$ 35.42 | 0.0138 | $p = 0.0138$ ($^{*}$) |
| **FEE-Net (w/o DWT Frequency)** | 5.84 | 11.18 | 35.42 $\pm$ 23.06 | 24.19 $\pm$ 19.34 | 116.34 $\pm$ 35.59 | 0.0442 | $p = 0.0442$ ($^{*}$) |
| **FEE-Net (w/o Edge-Gated Module)**| 5.85 | 10.91 | 34.57 $\pm$ 22.28 | 23.44 $\pm$ 19.19 | 115.89 $\pm$ 35.15 | 0.6693 | $p = 0.6693$ (ns) |
| **FEE-Net (w/o Boundary Loss)** | 5.86 | 11.33 | 34.49 $\pm$ 22.07 | 23.29 $\pm$ 18.69 | 116.41 $\pm$ 34.82 | 0.2001 | $p = 0.2001$ (ns) |
| **FEE-Net (Proposed Full)** | 5.86 | 11.33 | 34.86 $\pm$ 21.95 | 23.56 $\pm$ 18.81 | 118.16 $\pm$ 35.39 | Ref | Proposed Target |

### 5.2 出版级 LaTeX 输出验证
导出的 `assets/rigorous_evaluation_table.tex` 完整符合 IEEE/Springer 论文格式要求，准确格式化了科学计数法与显著性星号标注。

---

## 6. 结论与后续工作
1. **自动化基础设施完备**：多折多随机种子训练调度流水线已全部就绪，支持一键在后台批量完成 5-Fold $\times$ 3-Seed 的全量网格训练；
2. **统计学评测合规**：能够自动化读取多 Run 输出真正的宏观 Run-level 统计均值与方差，并通过非参数 Wilcoxon 符号秩检验输出顶级期刊直接可用的表格；
3. **后续推进**：使用 `run_real_benchmark.py --folds 0 1 2 3 4 --seeds 42 123 2024 --epochs 50 --patience 10` 在正式算力环境下执行 15 组全量交叉验证训练，并沉淀最终学术论文表格。
