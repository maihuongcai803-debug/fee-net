---
title: "Exp-030 架构对齐与单一真源 Benchmark 流水线重构（P0-1~P0-4 闭环）"
date: 2026-10-03
tags:
  - Experiment_Log
  - Code_Hygiene
  - Reproducibility
  - Benchmark_Pipeline
  - P0_Fix
---

# Exp-030：架构对齐与单一真源 Benchmark 流水线重构

> 触发文档：`06_Code_Docs/Code_Review_2026-10-03.md`（第二轮复查报告）
> 前序：[[Exp_029_Kaggle_DualT4_CPU_Spike_Diagnosis_and_Model_Bugfixes|Exp-029]]（150 例盲测基准）、[[Exp_028_DualBranch_Triage_and_Spectral_Wavelet_Overhaul|Exp-028]]（架构升级）

## 1. 实验日期与类型

- **编号**：Exp-030 ｜ **日期**：2026-10-03
- **类型**：实验流水线重构（无新训练结果产出）

## 2. 动机：第二轮复查发现的 4 个 P0

2026-10-03 复查确认工程侧整改无回归（pytest 22/22、v2 splits 150 例独立且与所有折不相交、Git 26.36 MiB），但通过**逐 Cell 解析线上 notebook** 发现实验产物与仓库代码脱钩：

| 编号 | 问题 | 证据 |
|---|---|---|
| P0-1 | 论文主模型 ≠ 仓库主模型 | notebook 内 `class FEENet` 只是 `FEENetEarlyExit` 薄包装，`return_aux=False`/`aux_preds=0` → 无深监督、无边缘头监督、无 `coarse_head`；仓库版 8.539M vs notebook 版 8.339M |
| P0-2 | 损失函数"写了但没跑" | 论文公式含 λ_ds=0.3 深监督与边缘 BCE 项，训练循环实际未启用 |
| P0-3 | models/ 与 utils/ 双重实现 | notebook 完整复制 5 个模型 + 2 个损失类，是历史 UNet 通道 bug、参数口径分歧的共同根因 |
| P0-4 | 测试集并非仓库 splits.json | notebook 用 `random.seed(42)` + 排序分位自行重划，与仓库 `splits.json`（`RandomState(42)`+`digitize`）**成员不同**，但论文统称"Exp-027 v2 协议" |

## 3. 核心改动

### 3.1 `tools/generate_kaggle_all_in_one_notebook.py`（重写，1300+ 行 → 单一真源版）

1. **源码注入（解 P0-1/P0-3）**：生成器在生成时读取 `models/{unet,attention_unet,unet_plus_plus,fee_net,fee_net_early_exit}.py`、`models/modules/{dwt_module,egm_module}.py`、`utils/{losses,metrics}.py` 共 **9 个模块**的**真实源码**并内联进 notebook；notebook 运行时将其物化为 `/kaggle/working/repo/{models,utils}` 真实包并 `sys.path` 导入，随后断言三类模块的 `__file__` 均位于该目录（provenance guard）。**仓库成为唯一真源**，notebook 不再自带模型副本。
2. **完整复合损失（解 P0-2）**：训练引擎直接调用 `utils.losses.CompoundLoss(seg, edge, masks)`，与 `train.py` 一致地叠加 0.3 权重多尺度深监督（`deep_sup4/3/2` 上采样后计入）与 0.5 权重 triage 交叉熵；边缘头 BCE 由 CompoundLoss 内部 `pred_edge` 分支自动加入。
3. **精确复用仓库 splits（解 P0-4）**：生成器把 `data/kvasir_seg/splits.json` 全文内联，notebook 在数据加载前断言 150/680/170 与"测试集 ∩ (train ∪ val) = ∅"，成员与仓库**逐一致**。
4. **分流门成本改为实测**：新增 `TriageGate`（stem + layer1 + Haar DWT + SCA + triage_head），用 CUDA Event 计时得到真实 FLOPs/延迟/FPS；FLOPs 解析表新增独立 `TriageGate: 1.45` 条目（此前误复用全路径 12.44，已修正）。图注中未经测量的 "285+ FPS" 声明**已删除**。
5. **v2 协议正式消融**：新增 `FEENet_noDWT`（8.546M）与 `FEENet_noEGM`（8.522M），`RUN_ABLATIONS=True` 时与主模型同协议同种子训练，输出 Dice 差值。
6. **指标口径统一**：评测改用仓库 `utils.metrics.MetricTracker(..., from_logits=True)`（Dice/mIoU/HD95/Precision/Recall），与本地 `evaluate.py` 一致；FLOPs 优先 thop 实测，缺失时回退解析表并标注来源。
7. **溯源打包**：结果 zip 内新增 `repo_provenance/`（9 个模块源码 + splits.json + PROVENANCE.txt），使每次运行自带可审计副本。

### 3.2 顺带修复的 3 个真实 Bug

| Bug | 位置 | 后果 | 修复 |
|---|---|---|---|
| `best_val_dice = 0.0` 初始化 | 训练引擎 | 首轮 val Dice 为 0 时**永不保存 checkpoint**，后续加载直接崩溃（与 `train.py` 的 `-1.0` 不一致） | 改为 `-1.0` |
| triage 阈值常量跨单元依赖 | 数据单元 → 训练/打包单元 | 跳过数据单元即 `NameError`，链路脆弱 | 收敛为 Cell 2 协议常量 `TRIAGE_PIXEL_THRESHOLD`，单点定义 |
| `REPO_SPLITS` 跨单元依赖 | 数据单元 → 打包单元 | 同上 | splits 嵌入随源码注入单元（Cell 3）并前置于数据单元，依赖方向单向 |

另修复 API 漂移：仓库 `FEENetEarlyExit.forward` 默认只返回掩码，取 risk 分数必须 `return_edge=True` 后取 `out[2]`（旧 notebook 私有实现返回三元组，已在 CRC 单元更正）。

### 3.3 目录治理

- 删除 `kaggle_feenet_baselines_all_in_one.py`（48.9 KB，审计修复**前**的旧提取，仍含 `fps_plot = fps_vals + [285.0]` 造假点）；
- 4 个 v1 时代 5-fold notebook 归档至 `scripts/archive/`；
- 根目录 `notebook0e7b0400ab.log` 移入已被 gitignore 的 `scratch/`。

### 3.4 论文（`05_Paper_Draft/`）

- 参数量口径统一为**仓库代码本地实测值**：U-Net 7.763M（原 7.241M）、Att-U-Net 7.852M（原 7.330M）、UNet++ 9.160M、FEE-Net 8.539M（原 8.339M）、FEE-Net-EE 8.552M（原 8.351M）；
- 新增 Table II「Repository Implementation Inventory」，逐模块列出参数与来源文件；
- 两张图改为 `\IfFileExists` 保护 → 缺图时给出占位框而非编译失败；
- Results 增加 **Provenance and status** 段落：明确 Table I 数值来自 2026-10-01 运行（notebook 内联简化版），架构对齐重跑为权威结果并將替换；指标在重跑完成前按 preliminary 对待，而参数量/协议/安全分析已反映发布代码。

## 4. 验证方式（重跑前本地全链路实测）

在本地 GPU 上按序执行生成 notebook 的全部 12 个代码单元（数据两单元以合成张量注入、`EPOCHS→1`、`thop→关闭`）：

- 9 个模块全部从注入目录解析（provenance guard 通过）；
- 7 个模型（UNet / AttUNet / UNet++ / FEE-Net / FEE-Net-EE / noDWT / noEGM）各完成 1 轮训练并保存 checkpoint；
- 独立测试评测、CRC 标定、两张图、CSV/LaTeX 表、溯源包全部产出；
- 分流门实测：`gate_params=0.075M`、`gate_gflops=1.45`（解析值，标注来源）、延迟与 FPS 由 CUDA Event 实测。

> 合成数据下的 Dice/HD95 无物理意义，仅验证链路；**权威指标需 Kaggle 重跑**。

## 5. 结论与后续

**结论**：4 个 P0 全部在代码层闭环，实验产物与仓库代码的对应关系现在是可验证的（运行期断言 + 溯源打包）。

**后续（需人工执行）**：
1. **Kaggle 重跑** `kaggle_feenet_baselines_all_in_one.ipynb`（约 60~75 分钟，含 2 个消融变体），产出权威 Table I、v2 消融、实测分流门数据与两张配图；
2. 从结果 zip 取回 `figure_qualitative_comparison.png` 补入 `05_Paper_Draft/`，并用新指标替换论文 Table I；
3. 把 Kaggle 产出的 v2 checkpoint 回传本地，替换 `checkpoints/` 中的旧架构权重，使 `evaluate.py` 本地可用；
4. 重跑后同步更新 Exp-029 §4.2 与论文摘要/结论中的数字。

---

## 6. 云端部署记录（2026-10-03 上午）

- **推送对象**：Kaggle Kernel `maihuongcai/notebook0e7b0400ab`（Private，GPU ×2，Internet 开启，数据集 `maihuongcai/kvasir`）
- **推送方式**：`tools/generate_kaggle_all_in_one_notebook.py` 生成 → `scratch/kaggle_push/`（notebook + `kernel-metadata.json`）→ `kaggle kernels push -p scratch/kaggle_push`
- **结果**：**Kernel version 3 推送成功并自动触发运行**（状态 `RUNNING`），原标题与 URL 保持不变
- **本轮运行内容**：7 个模型（U-Net / Attention U-Net / UNet++ / FEE-Net / FEE-Net-EE / w/o DWT / w/o EGM）× 35 epochs + 实测分流门 + CRC 标定 + 消融差值 + 两张图 + 溯源包，预计 60~75 分钟
- **结果自动回收**：`tools/watch_kaggle_kernel.py` 后台轮询 kernel 状态，训练完成后自动 `kaggle kernels output` 拉回 `scratch/kaggle_output_v3/`（日志 `scratch/kaggle_v3_watch.log`）
- **回传后待办**：用新指标替换论文 Table I、把 `figure_qualitative_comparison.png` 补入 `05_Paper_Draft/`、v2 checkpoint 回传本地替换旧架构权重、同步更新 Exp-029 §4.2 与摘要/结论数字

---

## 7. 创新性评估与两处硬伤修复（2026-10-03 晚，附 `06_Code_Docs/Novelty_Audit_2026-10-03.md`）

### 7.1 创新性逐条核查结论
| 声称创新点 | 判定 | 先前工作 |
|---|---|---|
| 2D Haar DWT 高频提取 | 非创新 | 小波分割是成熟路线（WRANet 2023、AdaptUNet 2024） |
| 频域通道注意力 SCA | 非创新（改名） | SENet (CVPR 2018) 通道注意力 |
| 八邻域 ODWF | 增量 | 3×3 depthwise 本身即覆盖 8 邻域，与已有 1×3/3×1 叠加冗余 |
| 双线性上采样 | 非创新 | Odena 2016（已正确引用） |
| 边界 IoU 损失 | 非创新 + 引文错误 | El Jurdi MICCAI 2021；Karimi TMI 2019 / MedIA 2022 |
| 深监督 + 边缘监督 | 非创新 | FPN/UNet++ 时代标准 |
| GAP+GMP 双分支分流头 | 增量（工程价值真实） | GAP 早退为标准（EENets Pool/Bnpool, arXiv:2409.05336） |
| CRC 标定早退阈值 | 应用层增量 | Conformal Lesion Segmentation、ICIPCA 2025 CRC-FDR、ML Research 2026 负面结果 |

**最危险事实**：同数据集上 WRANet（2023）报告 Kvasir-SEG Dice **88.89%**、AdaptUNet+wavelet（Heliyon 2024，已引 21 次）亦高于本项目 85.22% → "频域创新 + SOTA" 双重叙事不可维持。

### 7.2 硬伤 A：引文错误（已修）
`utils/losses.py` 原 docstring 写 "CVPR 2021 Standard"（该出处不存在）。已更正为 El Jurdi et al. (MICCAI 2021) / Karimi & Salcudean (IEEE TMI 2019) / Karimi & Salcudean (MedIA 2022)，并在论文损失小节补引三篇。

### 7.3 硬伤 B：CRC 校准与评测同集（已修）
原实现在 150 例测试集上同时做 τ 扫描、CP 标定与 FNR 汇报，违反 conformal risk control 的可交换性前提 → 形式保证不合法（与"用测试集调参"同类）。
**修复**：生成器 CRC 单元改为 **在验证折(170 例)标定 τ\\*** → **在 150 例测试集上仅认证与汇报一次**；新增 `triage_risk_scores.csv`（val/test 逐样本 risk 分数）供事后复核。本地合成数据验证：n=8、k=0 时 CP 上界 31.2%，与 1−0.05^(1/8) 一致，数学正确。
> ⚠️ 正在云端运行的 kernel v3 使用的是**修复前**协议，其 CRC 数字只能作 preliminary；v4 或用 v3 导出的 risk 分数重算才可作为正式保证。

### 7.4 论文定位调整
- Related Work 新增两段：小波+注意力的息肉分割现状（明确"不声称超越 WRANet/AdaptUNet"）、医学分割中的保形阈值控制现状（含"二元漏诊损失可能无法认证"的负面结果引用）
- Limitations 新增第 (5) 条组件级新颖性自评、第 (6) 条校准适用范围与负面结果引用
- 结论定位建议：主张从"提出更强新网络"改为"零泄漏协议 + 轻量分割 + 可认证筛查的系统级贡献"，投稿层级按 Novelty Audit 建议下调

### 7.5 意外收获（生成器自身缺陷）
修复过程中发现 notebook 单元里写 `"""docstring"""` 会提前闭合生成器的外层三引号字符串 → 生成器语法错误。已改为 `#` 注释。**这类错误只有真正执行生成器才会暴露**，印证了方案中"把生成器验证自动化为 pytest"的必要性。
