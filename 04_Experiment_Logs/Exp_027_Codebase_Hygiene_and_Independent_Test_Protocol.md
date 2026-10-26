---
title: "Exp-027 全工程代码卫生整改与独立测试集协议 (v2 splits)"
date: 2026-10-01
tags:
  - Experiment_Log
  - Code_Hygiene
  - Data_Protocol
  - P0_Fix
---

# Exp-027：全工程代码卫生整改与独立测试集协议（v2 splits）

> 关联文档：`06_Code_Docs/Code_Review_2026-10-01.md`（完整审查报告）

## 1. 实验日期与编号

- **编号**：Exp-027
- **日期**：2026-10-01
- **类型**：代码整改 + 数据协议升级（无新训练）

## 2. 实验目的与修改动机

源自 2026-10-01 全项目深度审查（同日报告），发现三类问题：

1. **P0 科学性**：v1 协议中 `test_ds` 与 `val_ds` 完全相同（`real_medical_loader.py` L194/L273），模型选择（早停/最优 checkpoint）与最终评测同源 → 选择偏差，论文审稿风险。
2. **P1 代码 Bug**：`models/fee_net.py` 自检引用已不存在的属性（`wf1.conv1`/`bot.conv1`，BERCA 重命名后未同步），直接运行必崩；`evaluate.py` 缺 checkpoint 时用随机权重静默评测；`configs/default_config.yaml` 从未被读取且数值与实际训练矛盾；`utils/metrics.py` 靠取值范围猜测 logits。
3. **P2 仓库卫生**：Git 追踪了 5800 个二进制文件（"results (1)" 4000 个 + synthetic_medseg 1800 个），pack 达 458MB；Kvasir 原始数据存 3 份；checkpoint 双命名体系。

## 3. 涉及文件与核心代码改动

### 3.1 数据协议（P0）
- `datasets/real_medical_loader.py`：
  - `generate_official_stratified_5fold` 升级为 **v2 协议**：全 1000 例按病灶面积比 5 层分层 → 每层固定 30 例（共 **150 例全局独立 test**）→ 剩余 850 例做 5 折分层 CV（每折 680 train / 170 val）。函数内 **断言 test 与所有折 train/val 不相交**。
  - **写保护**：splits.json 已存在时不重生成（除非 `force=True`）；v1 覆盖前自动备份为 `data/kvasir_seg/splits_v1_backup.json`。
  - `KvasirSegDataset` 支持 `split="test"`；旧 v1 splits 回退到 fold-val 并显式告警（Kaggle 兼容）。
  - `__getitem__` 的逐样本 glob 改为 `__init__` 内一次性 stem→path 缓存。
  - 硬编码 `d:\machine_learning\...` 默认路径全部改为仓库相对路径。
- `data/kvasir_seg/splits.json`：已迁移到 v2（v1 备份保留）。

### 3.2 模型与评测（P1）
- `models/fee_net.py` 自检：修复 `wf1.conv1→wf1.weight`、`bot.conv1→bot[0]/bot[1]`；参数量断言 6.5M→~8.52M（BERCA 扩容后实测 8,517,548）；checkpoint 兼容性检查对 missing keys 显著告警。**已实跑通过**。
- **架构漂移发现（重要）**：`fee_net_best.pth`/`fee_net_fold0_seed42_best.pth` 记录 params_m=6.8918（旧架构），载入当前 8.52M 模型静默缺 36 键（bot.* 等）——正是 strict=False 掩盖的问题。`evaluate.py` 现在对此打印显式警告。
- `evaluate.py`：缺 checkpoint 直接 `FileNotFoundError`（原来用随机权重继续）；加载后报告 missing/unexpected keys；`models_to_eval` 优先 canonical 命名（`_fold0_seed42_best`），旧名回退。
- `utils/metrics.py`：`calculate_dice/iou/hd95/precision_recall` 与 `MetricTracker.update` 全部增加显式 `from_logits` 参数（True=先 sigmoid / False=直接阈值 / None=旧启发式）；`train.py`、`evaluate.py` 显式传 `True`。
- `train.py`：`--config` 真正接通 YAML（此前解析但从未读取）；移除 fold0+seed42 的 `_best.pth`/曲线/history 三处 legacy 双写；新增 `--num_workers`（默认 0，Windows 安全）；triage 像素阈值常量化（`TRIAGE_POSITIVE_PIXEL_THRESHOLD`）；移除永假的 `--detach_triage` no-op flag（保留 `--no_detach_triage`，全仓库 grep 确认无调用方）。
- `configs/default_config.yaml`：对齐历史实验真实超参（epochs 50 / bs 16 / lambda_boundary 0.1），保证接通后行为不变。已验证 argparse 输出与历史值一致。

### 3.3 仓库与依赖（P2）
- Git：`git rm --cached` 5800 个二进制 + `git filter-repo --invert-paths` 重写历史。**pack 458MB → 26.36MB**（剩余为 papers PDF 与 assets 图表）。重写前全量备份：`D:/machine_learning_backup_before_slim.bundle`（480MB，可用 `git clone bundle` 恢复任意历史）。
- 数据归档（仓库外）：`D:/machine_learning_archives/` ← `results (1)`（457MB，Kaggle 运行 A）、`results.zip`（219MB，运行 B，两者非重复）、`kvasir_seg_dataset.zip`（44MB）、2 个 notebook 日志。
- `requirements.txt`：移除未使用的 `medpy`；`transformers` 移入新建 `requirements-tools.txt`（仅 `tools/laya_router.py` 使用）；补充 ~= 版本约束与已验证环境参考。
- 归档移动（git mv，保留历史）：`run_experiments.py`、`run_real_benchmark.py`、`tools/generate_kaggle_early_exit_notebook.py`（v1）、`kaggle_feenet_early_exit_benchmark.ipynb`（v1）→ `scripts/archive/`。

## 4. 超参数与训练配置

本轮无训练。配置基线（config.yaml，与历史实验一致）：epochs=50，batch_size=16，lr=3e-4，AdamW(wd=1e-4)，CosineAnnealing→1e-6，AMP，patience=10，seed=42，lambda_boundary=0.1。

## 5. 实验结果与指标对比

| 项 | 整改前 | 整改后 |
|---|---|---|
| 测试集独立性 | test == fold val（选择偏差） | 150 例全局独立 test，与所有折不相交（断言保证） |
| splits.json 复现性 | 误触 regeneration 即静默改变划分 | 写保护 + v1 备份 |
| fee_net 自检 | 运行必崩（AttributeError） | 全部通过（实跑验证） |
| Git 体积 | 458MB pack | **26.36MB** |
| 磁盘占用 | 3.1GB | ~1.9GB（约 1.2GB 移至 archives） |
| 随机权重评测风险 | 静默 | FileNotFoundError |
| 旧架构 checkpoint 载入 | 静默缺 36 键 | 显式 [LoadCheck][WARN] |
| config 与实际训练 | 矛盾且未生效 | 生效且对齐历史 |

## 6. 结论与后续改进建议

**结论**：P0/P1/P2 全部落地，训练/评测主链路行为与历史实验保持一致（config 对齐验证通过）。

**关键后续（影响论文）**：
1. ⚠️ **v2 协议下所有最终 test 指标需在 Kaggle 重训后重报**：现有 checkpoint 全部基于 v1 划分（且 fee_net 系 checkpoint 属旧 6.90M 架构，与当前 8.52M 代码不兼容），v2 test 集必须使用在 v2 fold 上训练、且 test 从未参与训练与模型选择的模型评测。建议复用 `kaggle_feenet_early_exit_benchmark_v2.ipynb` 改造为 v2 splits 版本。
2. checkpoints 目录的旧别名/旧架构文件（fee_net_best.pth 等）暂保留未动——重训完成后统一归档，避免破坏 Exp-017~026 的回溯链。
3. kvasir_triage 子数据集尚无独立 test（loader 内已注明），后续如论文引用 triage 结果需同样升级协议。
4. DiceLoss 仍为 batch 级全局计算（与 per-sample 评测指标不完全一致）——属训练行为变更，需独立实验（Exp 留档）后再改，本轮未动。
5. 深监督 aux loss 复用含 Boundary-IoU 的 criterion（3×7×7 max-pool），如需提速可让 aux 只算 BCE+Dice，同样需独立消融实验。

## 7. Git 提交链

- `36b7101` docs(review): 审查报告入库
- `e3c21f4` fix(data): P0 独立 test 集 v2 协议
- `ba2cbb8` fix(models): 自检修复 + checkpoint 漂移告警
- `d47fc0f` fix(train): config 接通 + 去重写 + workers 可配
- `869ae09` fix(metrics): 显式 from_logits
- `ddceab0` chore(repo): 解除 5800 二进制追踪
- `17c5e2d`(重写后) data(splits): v2 迁移 + scripts 归档 + requirements 拆分
