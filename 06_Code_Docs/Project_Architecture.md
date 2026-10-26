---
title: 代码工程架构与模块规范
date: 2026-10-26
tags:
  - Code_Docs
  - Architecture
  - Python
---

# 💻 项目代码工程架构设计说明

> 本文件描述 **2026-10-26 结构精简后**的真实结构（依据
> [[Structure_Optimization_Plan|结构优化方案]] 的 A→B→C→D 动作包执行）。
> 与旧版的主要差异：Laya 分流子系统已从 `models/` 剥离为独立包，一次性评测脚本已收敛到
> `experiments/`，根目录只保留两个规范入口。

---

## 1. 三层解耦原则

| 层 | 目录 | 职责 | 是否入库 |
|---|---|---|---|
| **知识库层** | `00_Index.md` ~ `06_Code_Docs/` | 方案、文献、推导、实验留档、论文、工程文档 | ✅ |
| **库代码层** | `models/`、`utils/`、`datasets/`、`laya/` | 可复用实现：网络、损失、指标、数据管道 | ✅ |
| **入口层** | `train.py`、`evaluate.py`、`laya/`、`experiments/`、`tools/` | 可执行命令行程序 | ✅ |
| **产物层** | `checkpoints/`、`assets/`、`data/`、`scratch/` | 权重、图表、数据、临时文件 | ❌ 全部 gitignore |

`models/` 现在只依赖 `models/modules/{dwt,egm}`，可独立作为纯分割构件发布；
`laya/` 单向依赖 `models.modules.dwt_module`，反向无依赖。

---

## 2. 目录树（当前真实结构）

```text
machine_learning/
├── 00_Index.md                   # Obsidian 知识库主入口 (MOC)
├── 01_Research_Proposal/         # 选题背景与可行性论证
├── 02_Literature_Review/         # 7 篇顶刊精读
├── 03_Methodology/               # 数学推导、网络架构、System1/2 框架
├── 04_Experiment_Logs/           # Exp_001~Exp_034 实验留档 + 索引 + 模板
├── 05_Paper_Draft/               # 中英双语稿件（tex/docx）+ 图表
│   ├── archive/                  #   被取代的历史稿件
│   └── zh/                       #   中文稿构建脚本 (content.py / build_*.py)
├── 06_Code_Docs/                 # 本目录：架构、环境、指标、审计
├── configs/default_config.yaml   # 训练超参默认值（train.py 读取，CLI 可覆盖）
│
├── models/                       # ── 纯分割库 (System 2) ──
│   ├── fee_net.py                #   FEE-Net 主模型
│   ├── fee_net_early_exit.py     #   + EarlyExitTriageHead
│   ├── unet.py / attention_unet.py / unet_plus_plus.py
│   └── modules/
│       ├── dwt_module.py         #   2D 离散小波高频分解
│       └── egm_module.py         #   边缘感知跨尺度门控 (EGM/BERCA)
│
├── laya/                         # ── 分流子系统 (System 1) ──
│   ├── modules/
│   │   ├── laya_visual_triage.py     # 超轻量视觉筛查头
│   │   ├── laya_clinical_triage.py   # 临床分层判定
│   │   └── laya_cascade_router.py    # 三级动态级联路由
│   ├── train_laya_triage.py      # 训练 + 临床安全校准
│   ├── eval_cascade.py           # 级联基准评测
│   └── router.py                 # 智能体任务分流器
│
├── utils/                        # losses / metrics / visualizer / conformal_triage
├── datasets/                     # real_medical_loader / kvasir_triage_builder / transforms
│
├── experiments/                  # ── 历史与一次性评测入口 ──
│   ├── eval_dual_system.py           # 双系统吞吐/延迟评测
│   ├── run_practical_verification.py # Exp-026 四大范式全景验证
│   ├── run_rigorous_evaluation.py    # 多折多种子 + Wilcoxon
│   └── run_statistical_eval.py       # 样本级配对显著性检验
│
├── tools/                        # 活跃工具（23 个，按需使用）
│   ├── generate_kaggle_all_in_one_notebook.py  # 唯一 notebook 生成器
│   ├── watch_kaggle_kernel_v5.py               # 唯一 Kaggle 轮询器
│   ├── ingest_kaggle_results.py / process_kaggle_v2_results.py
│   ├── calibrate_feenet_early_exit_crc.py      # CRC 校准
│   ├── train_triage_head.py / train_feenet_early_exit_head.py
│   ├── eval_5fold_results.py / evaluate_all_checkpoints.py / ...
│   ├── make_qualitative_figure.py / profile_models.py
│   └── build_mixed_cohort.py / run_probe_matrix.py
│
├── train.py                      # ★ 规范训练入口
├── evaluate.py                   # ★ 规范评测入口
├── tests/                        # 22 个 pytest 用例
├── scripts/archive/              # 已冻结的历史脚本
│
├── checkpoints/                  # 权重（gitignore）
│   ├── v5_final/                 #   ★ 论文全部数字的来源权重（7 模型）
│   ├── triage_head_stage2_final.pth
│   └── *_fold*_seed42_best.pth   #   5 折交叉验证权重
├── assets/kaggle_v5_final/       # v5 基准结果 CSV/JSON/LaTeX 表 + 结果 zip
├── data/                         # kvasir_seg / kvasir_triage / mixed_cohort（gitignore）
├── papers/                       # 文献 PDF（本地保留，不入库）
└── scratch/                      # 临时探索脚本（1.5 MB，较重构前 -1.4 GB）
```

---

## 3. 核心模块契约

| 模块 | 输入 → 输出 |
|---|---|
| `modules/dwt_module.py` | `(B,C,H,W)` → `LL/LH/HL/HH`，各 `(B,C,H/2,W/2)`；无参、可反向传播 |
| `modules/egm_module.py` | 深层语义特征 + 频域高频 → 跨尺度门控注意力权重 |
| `utils/losses.py` | BCE + Dice + Boundary IoU 复合损失（深监督 + 边缘监督） |
| `utils/metrics.py` | Dice / IoU / HD95 / 参数量 / FLOPs |
| `utils/conformal_triage.py` | 有限样本保形风险控制（CRC），输出阈值 τ* 与风险上界 |
| `laya/modules/laya_visual_triage.py` | `(B,3,128,128)` → 二分类 logit（正常 / 可疑） |
| `laya/modules/laya_cascade_router.py` | 单帧 → `FAST_BYPASS` / `LIGHT_CHECK` / `FULL_SEGMENTATION` |

---

## 4. 运行入口

```bash
# 训练与标准评测（根目录规范入口）
python train.py --config configs/default_config.yaml --model feenet --fold 0
python evaluate.py --checkpoint checkpoints/v5_final/best_feenet.pth

# 分流子系统
python laya/train_laya_triage.py
python laya/eval_cascade.py
python laya/router.py

# 历史/一次性评测
python experiments/run_practical_verification.py
python experiments/run_rigorous_evaluation.py

# 测试
python -m pytest tests/ -q
```

> 所有入口脚本均已内置 `PROJECT_ROOT` 到 `sys.path` 的引导，可从任意工作目录执行。

---

## 5. 产物存放约定

- **权重** → `checkpoints/`。论文引用的权威权重固定在 `checkpoints/v5_final/`，不得随意覆盖。
- **基准结果表** → `assets/kaggle_v5_final/`（CSV/JSON/LaTeX），论文数字以此处为准。
- **临时探索** → `scratch/`，可随时清空；超过一周的中间产物应删除或迁入正式目录。
- **实验结论** → 必须写入 `04_Experiment_Logs/Exp_XXX_*.md`（见 [[GEMINI|GEMINI.md]]）。

---

## 6. 变更记录

**2026-10-26 结构精简**
- `models/modules/laya_*.py` → `laya/modules/`；`models/` 回归纯分割包
- 根目录 8 个脚本 → 根保留 `train.py`/`evaluate.py`，其余入 `experiments/`，`train_laya_triage.py`/`eval_laya_jev_cascade.py` 入 `laya/`
- `tools/` 31 → 23：4 个 notebook 生成器收敛为 `generate_kaggle_all_in_one_notebook.py`，3 版 watcher 收敛为 v5，移除 `plot_pareto_clean.py` / `retry_download.py`
- 修正迁后 7 处 import 与 5 处 `sys.path` 引导；修复 `run_practical_verification.py` 的 argparse `%` 转义 bug
- `papers/` 与派生数据集解除 Git 跟踪；`scratch/` 1.4 GB → 1.5 MB

历史留档（`04_Experiment_Logs/`）中出现的旧路径（如 `eval_laya_jev_cascade.py`）为当时快照，未回改，对照关系见本节。
