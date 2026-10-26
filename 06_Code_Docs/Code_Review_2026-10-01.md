# 项目深度审查报告（Code & Science Review）

> 审查日期：2026-10-01
> **✅ 整改状态（2026-10-01 当日落地）**：P0#1/#2、P1#3~#7、P2#8~#10 均已修复，详见 `04_Experiment_Logs/Exp_027_Codebase_Hygiene_and_Independent_Test_Protocol.md`。Git 458MB→26.36MB；重写前备份 `D:/machine_learning_backup_before_slim.bundle`。⚠️ 唯一遗留：v2 协议下的最终 test 指标需 Kaggle 重训后重报（现有 checkpoint 属 v1 划分 + 旧 6.90M 架构）。
> 审查范围：全部核心代码（train/evaluate/models/datasets/utils/tools）、数据与 Git 仓库管理、配置体系、实验规范落地情况
> 总体评价：项目结构与实验留档规范非常好（Obsidian MOC + Exp 留档 + Git 纪律），但存在 **1 个影响论文科学严谨性的关键缺陷**、**1 个已验证的代码 Bug**、以及 **仓库体积失控（Git 内嵌 457MB 结果图）** 三大问题。

---

## 🔴 P0 — 科学严谨性问题（论文审稿风险）

### 1. "测试集" 就是验证集（选择偏差 / 信息泄漏）
`datasets/real_medical_loader.py` L194 与 L273：

```python
val_ds  = KvasirSegDataset(split="val", fold=fold)
test_ds = KvasirSegDataset(split="val", fold=fold)   # ← test 完全等于 val
```

5 折交叉验证中，早停与最优模型选择（`val_dice` 最优才保存 checkpoint）都基于 val 集，而 `evaluate.py` 汇报的 "TEST SET" 结果用的就是同一批数据。**这等于用参与模型选择的数据来汇报最终性能**，是审稿人一眼就能抓住的问题。

**建议**：Kvasir-SEG 共 1000 张，可固定划出 200 张做独立 test（不参与任何 fold），5 折只在剩余 800 张上做；或至少在论文中明确说明"val 即 test，模型选择与评测同源"。

### 2. splits.json 可被静默覆盖（复现性隐患）
`generate_official_stratified_5fold()` 每次被直接调用都会**用当前 seed 重写** `data/kvasir_seg/splits.json`。若某次数据文件变动后误触 regeneration，所有历史 fold 实验与新版 splits 不再可比，且不会报错。

**建议**：splits.json 生成后加内容哈希校验或只读标记；regeneration 需显式 `--force`。

---

## 🟠 P1 — 已验证的代码 Bug

### 3. `models/fee_net.py` 自检脚本必崩（已运行验证）
L386 `model.wf1.conv1.weight` 与 L392 `model.bot.conv1` 引用的属性不存在（`wf1` 是 `nn.Conv2d`，`bot` 是 `nn.Sequential`，均已实测抛 `AttributeError`）。直接运行 `python models/fee_net.py` 会在第 2 步梯度检查处崩溃。
这说明该自检在架构重构（wf1/bot 改名）后**从未再跑通过**，文件尾部的 "ALL CHECKS PASSED PERFECTLY" 是历史遗留。建议修正为 `model.wf1.weight.grad` / `model.bot[0].weight.grad`，并把各模型 `__main__` 自检纳入 CI 或定期运行。

### 4. `evaluate.py` 的静默失败路径
- checkpoint 不存在时仅打 warning 然后用**随机初始化权重**继续评测，会产出无意义结果；
- 所有模型 `load_state_dict(strict=False)`，即使 checkpoint 与结构严重不匹配也不报错。
**建议**：缺 checkpoint 直接 raise；加载后校验 missing/unexpected keys 数量并打印。

### 5. `configs/default_config.yaml` 完全没被使用
`train.py` 导入了 `yaml`、定义了 `--config` 参数，但从未读取该文件。且两边数值互相矛盾：

| 项 | config.yaml | train.py 实际 |
|---|---|---|
| epochs | 30 | 50 |
| batch_size | 8 | 16 |
| lambda_boundary | 0.5 | 0.1 |
| device | cuda | 自动探测 |

配置文件会给后来者（或审稿人复现时）错误信息。**建议**：让 `parse_args` 从 YAML 读默认值（argparse `default=cfg.get(...)`），或在文件头声明 "deprecated, 以 train.py 为准"。

### 6. `utils/metrics.py` 用取值范围猜 logits
`_prepare_mask` 以 "min<0 或 max>1" 判断是否需要 sigmoid。当整张 logit 图恰好落在 [0,1]（训练早期很常见）时会被当成概率，指标失真且不可复现。**建议**：给 `MetricTracker.update()` 加显式 `from_logits=True` 参数。

### 7. 小问题
- `train.py --detach_triage`：`action="store_true"` + `default=True`，该 flag 永远为 True，语义上只能用 `--no_detach_triage` 关闭，易误用；
- `train.py` 中 triage 目标阈值 `masks.sum(dim=(1,2,3)) > 10` 硬编码，隐式依赖 256×256 分辨率；
- `DiceLoss` 按 batch 全局展平计算，而评测 Dice 是 per-sample 均值——小目标样本在训练损失中被稀释，训练/评测目标不一致；
- `real_medical_loader.__getitem__` 每个样本每次访问都 `glob` 两次定位文件，应在 `__init__` 建立 stem→path 缓存；
- 各 loader 默认路径硬编码 `r"d:\machine_learning\data\..."`，换机器即失效，应改相对路径或统一读配置。

---

## 🟡 P2 — 仓库与存储管理（磁盘/克隆性能）

### 8. Git 仓库内嵌约 457MB 实验输出 ⚠️
`git ls-files` 实测：**"results (1)/" 下 4000 个 jpg/png 全部被 Git 追踪**，另有 1801 个 data 文件被追踪，`size-pack` 达 458MB。这些是预测可视化输出，不属于代码资产。且目录名 `results (1)` 形如浏览器重复下载副本，与 `results.zip`（219MB）内容高度疑似冗余——三份（folder + zip + git objects）可能存的是同一批东西。

**建议（按顺序）**：
1. 先人工确认 `results (1)/`、`results.zip` 是否冗余，合并为一份归档（放网盘/移动硬盘，不进 Git）；
2. `git rm -r --cached "results (1)" data/*.jpg` + 补充 `.gitignore`；
3. 用 `git filter-repo` 重写历史瘦身（否则 pack 里旧对象仍在），操作前先 `git bundle` 全量备份；
4. `checkpoints/`（1.5GB）中 `*.pth` 已 ignore，但 `train.py` L302-304 的 legacy alias 机制把同一模型双份保存，可去掉。

### 9. 依赖管理
`requirements.txt` 全部为开放式 `>=`，无锁定版本；`transformers` 仅 `tools/laya_router.py` 一处使用，却被声明为主依赖（拖重安装）。**建议**：`pip freeze` 生成 `requirements.lock.txt`，把 `transformers` 移入 `requirements-tools.txt`。

### 10. 根目录脚本堆积
根目录有 7 个 `run_*.py` / `eval_*.py` / `train_laya_triage.py` 入口，与 `tools/` 职责重叠；`build_model` / `load_model` 在 `train.py`、`evaluate.py`、`run_rigorous_evaluation.py` 三处重复实现，改结构时容易漏改（fee_net 自检崩掉很可能就是这类漏改的产物）。**建议**：统一收敛到 `models/factory.py` 一个入口 + `scripts/` 目录收纳实验脚本。

---

## 🟢 P3 — 训练效率优化（可选）

- **num_workers=0 硬编码**（8 处）：Windows 下安全，但数据加载必然是瓶颈。建议做成 CLI 参数，Windows 用 `if __name__ == "__main__"` 保护 + `persistent_workers=True`；
- **深监督的 aux loss 复用了含 Boundary-IoU 的 criterion**：每个 aux 头都要过 3 次 7×7 max-pool 形态学边界提取，显存与耗时开销不小；aux 项通常只算 BCE+Dice 就够；
- `set_seed` 未设 `torch.use_deterministic_algorithms(True)` 与 DataLoader generator；checkpoint 里保存了整个 `optimizer.state_dict` + `history`（复现有用，但可加 `--save_optimizer false` 开关减小 1.5GB 的 checkpoints 目录）；
- `torch.load(..., weights_only=False)` 有反序列化执行任意代码的风险，自用可以，若论文要求开源需注意；
- `tests/` 只有 4 个测试文件且未覆盖 losses/metrics/数据加载——P0#1 和 P1#3 这类问题恰好是测试能兜住的。

---

## 修复优先级建议

| 优先级 | 事项 | 理由 |
|---|---|---|
| ① | 独立 test 划分（P0#1） | 直接影响论文结论可信度 |
| ② | Git 瘦身 + 冗余数据合并（P2#8、Part 2） | 458MB 仓库 + 约 1.3GB 冗余数据，越拖越难清 |
| ③ | fee_net 自检修复 + strict 加载校验（P1#3/4） | 已崩，且掩盖结构不匹配风险 |
| ④ | config.yaml 接通或废弃（P1#5）、metrics 显式 logits（P1#6） | 复现性 |
| ⑤ | build_model 收敛、路径去硬编码、num_workers 可配 | 可维护性 |

---
---

# 第二部分：目录与文件结构冗余分析（2026-10-01 补充）

> 全项目实测总占用 **3.1GB**，其中代码与文档不足 5MB——其余全部是数据、checkpoint 与结果存档，且存在大量**同一份数据的多副本**和**被 Git 误追踪的大文件**。

## 1. 冗余数据清单（实测）

### 1.1 Kvasir-SEG 原始数据集存了 3 份
| 位置 | 形式 | 大小 | 是否进 Git |
|---|---|---|---|
| `data/kvasir_seg/` (images 1000 + masks 1000) | 解压后 | ~389MB（含 triage/synthetic） | 部分追踪 |
| `"results (1)/data/kvasir_raw/"` (images 1000 + masks 1000) | Kaggle 下载副本 | ~2000 文件 | **✅ 4000 个文件全部被 Git 追踪** |
| `kvasir_seg_dataset.zip`（根目录） | 压缩包 | 46MB | 否（`*.zip` 已 ignore） |

**建议**：保留 `data/kvasir_seg/` 一份；确认 Kaggle 副本与本地一致（可用文件数 + 抽样哈希比对）后，整目录删除或移入离线归档盘。

### 1.2 Kaggle 结果存了 2 份，且 457MB 全部进了 Git
- `"results (1)/"`（457MB，4004 文件）：4 个 Kaggle 训练 checkpoint（attention_unet/unet/unet_plus_plus/feenet_early_exit 的 `_best.pth`）+ `data/` 子目录（kvasir_raw + kvasir_triage_v2，共 4000 文件）。**这 4000 个文件全部被 Git 追踪**，目录名形如浏览器重复下载副本。
- `results.zip`（228MB，2006 条目）：另一次 Kaggle 运行输出（best_model_fold0~4.pth + 2001 个预测图）。

两者是**不同运行**的输出（内容非重复），但都不属于代码资产。
**建议**：两份都移出仓库目录 → 统一归档为 `archives/kaggle_outputs/<run_name>/`（放网盘或移动硬盘），`results.zip` 解压后与 zip 二选一；执行前先确认 checkpoint 的 `val_dice` 元数据，保留指标更好的那份。

### 1.3 Git 被追踪的二进制文件（瘦身的真正目标）
`git ls-files` 实测被追踪的非代码文件：
| 范围 | 文件数 | 说明 |
|---|---|---|
| `results (1)/` | 4000 | `.gitignore` 规则缺失，整目录入库 |
| `data/synthetic_medseg/` | 1800 | `.gitignore` 里虽有 `data/synthetic_medseg/**`，但**加规则前已 commit**，规则对已追踪文件无效 |
| 根目录 5 个 `.ipynb` | 5 | 建议保留追踪（实验代码），但输出 cell 应清空 |
| `assets/` 训练曲线 png 等 | 41 | 体积尚可，可接受 |

Git `size-pack` 已达 **458MB**——即使现在删除文件，历史对象依然存在。
**建议流程**：
```bash
# 0) 先全量备份！
git bundle create ../machine_learning_backup.bundle --all

# 1) 移出冗余数据到仓库外归档（人工确认后）
# 2) 解除追踪（不删工作区文件）
git rm -r --cached "results (1)" data/synthetic_medseg

# 3) 补充 .gitignore
printf 'results*/\ndata/**\n!data/kvasir_seg/splits.json\n' >> .gitignore

# 4) 提交后用 filter-repo 重写历史瘦身（需 pip install git-filter-repo）
git filter-repo --path "results (1)" --path data/synthetic_medseg --invert-paths --force
```
完成后仓库预计从 458MB 降至 **<10MB**（保留全部代码、文档、笔记与论文）。若项目托管在远端（如 GitHub），重写历史后需 force-push 并通知协作者。

### 1.4 checkpoints/（1.5GB）：双命名体系并存，血统不清
实测存在**两套并行命名**，且部分"别名"内容并不相同（哈希比对为 DIFF，说明是不同轮次的训练结果，但文件名上看不出区别）：

| 规范命名 (`_fold0_seed42_best`) | 旧别名 (`_best`) | 尺寸对比 |
|---|---|---|
| `unet_fold0_seed42_best.pth` 89MB | `unet_best.pth` 27.7MB | **不同**（别名缺 optimizer state 或旧架构） |
| `unet_plus_plus_fold0_seed42_best.pth` 105MB | `unet_plus_plus_best.pth` 35MB | 不同 |
| `fee_net_fold0_seed42_best.pth` 79.1MB | `fee_net_best.pth` 79.1MB | 不同（同尺寸不同内容，疑似重训过） |
| `fee_net_early_exit_fold0_seed42_best.pth` | `feenet_early_exit_best.pth` | 命名风格还有第三种（`feenet` 连写） |

另有疑似历史遗留：`fee_net_v2_fold0~4_best.pth`（5×32.6MB，v2 命名已废弃，FEE-Net v2.1 即现架构）。

**建议**：
1. 确定 canonical 命名 `{model}_fold{k}_seed{s}_best.pth`，写进 `06_Code_Docs`；
2. 旧别名文件核对元数据（checkpoint 内含 `val_dice`/`history`）后归档或删除，避免"到底加载哪个"的歧义——`evaluate.py` 目前还在引用旧别名 `unet_best.pth` 等，改名需同步；
3. 删除 `fee_net_v2_fold*.pth` 前先确认 Exp-022/025 留档中是否声明复现依赖；
4. 关闭 `train.py` L302-304 的 legacy alias 双写；
5. 纯分类头的轻量文件（laya_triage 0.6MB、量化版 0.3MB）无需处理。

## 2. 脚本与文件组织冗余

### 2.1 评测入口多达 9 个，职责重叠
根目录：`evaluate.py`、`eval_dual_system.py`(412行)、`eval_laya_jev_cascade.py`(906行)、`run_rigorous_evaluation.py`、`run_real_benchmark.py`、`run_statistical_eval.py`、`run_practical_verification.py`；`tools/` 里还有 `eval_5fold_results.py`、`evaluate_all_checkpoints.py`。
其中 `run_real_benchmark.py` 仅 7.5KB、`run_experiments.py` 仅 1.6KB，属一次性脚本。
**建议**：一次性脚本（已完成历史使命、结论已写入 Exp 留档）移入 `scripts/archive/`；活跃评测收敛为 `tools/eval.py` 单入口 + 子命令（`eval.py benchmark --fold 0 --model fee_net`）。

### 2.2 Kaggle notebook 与生成器三重冗余
根目录 5 个 notebook 中，`kaggle_feenet_early_exit_benchmark.ipynb` 与 `_v2.ipynb` 是新旧两版并存；`tools/` 中 3 个 notebook 生成器（`generate_kaggle_notebook.py`、`generate_kaggle_early_exit_notebook.py`、`_v2.py`）+ `build_kaggle_notebooks.py` 也是迭代堆叠。
**建议**：旧版 notebook 与旧版生成器归档；保留 v2 一条链路。

### 2.3 根目录杂物
- `notebook15e5b3b6fd.log`、`notebook19aa62670b.log`：Kaggle 训练日志散落根目录，且**被 Git 追踪**。内容应已摘录进 `04_Experiment_Logs/`，建议移入 `archives/logs/` 并解除追踪；
- `__pycache__/`、`.pytest_cache/`、`tools/__pycache__/`、`tests/__pycache__/`：确认在 `.gitignore`（已覆盖），无需入库但可在磁盘清理时顺手删；
- `GEMINI.md` 与 `00_Index.md` 的规则章节存在少量重复描述，可接受（一个是面向 AI 的规范，一个是 MOC）。

## 3. 建议的目标结构

```
machine_learning/
├── 00_Index.md … 06_Code_Docs/     # Obsidian 知识库（保持现状，很规范）
├── configs/          # default_config.yaml（接通后为唯一超参来源）
├── datasets/  models/  utils/  tests/   # 库代码（保持）
├── tools/            # eval.py（收敛后单入口）+ 活跃工具
├── scripts/archive/  # run_*.py 等一次性脚本 + 旧版 notebook 生成器
├── notebooks/        # 活跃 Kaggle notebook（仅 v2）
├── data/             # 只保留数据本体 + splits.json（全部 gitignore）
├── checkpoints/      # 仅 canonical 命名，gitignore
├── assets/           # 结果图表（保留 git 追踪，体积可控）
├── papers/           # 文献 PDF
└── archives/         # 【仓库外】results.zip、旧 checkpoint、Kaggle 日志
```

## 4. 预期收益

| 动作 | 磁盘 | Git |
|---|---|---|
| 现状 | 3.1GB | 458MB pack |
| 移出 results (1)/results.zip/kvasir_seg_dataset.zip + 清 checkpoints 冗余 | **约 -1.2GB → 1.9GB** | - |
| filter-repo 重写历史（5800 个已追踪二进制） | - | **458MB → <10MB** |
| 清理 __pycache__/.pytest_cache | -0.1GB | 已 ignore |

合计可回收约 **1.6GB** 磁盘与 **98%** 的 Git 体积，克隆/备份从分钟级降到秒级。

## 5. 执行顺序与安全提示

1. **先备份**：`git bundle` + 外部拷贝整个目录（3.1GB 一次性成本）；
2. 数据合并删除属**不可逆操作**：逐项人工确认后再动手（尤其 `results (1)` 与 `results.zip` 是不同运行，先比对 checkpoint 内 `val_dice` 再决定取舍）；
3. `filter-repo` 会改写全部 commit 哈希：如有远端，需协调 force-push；
4. 改 checkpoint/脚本命名时同步更新 `evaluate.py` 的引用列表与 `04_Experiment_Logs` 中的文件名记录，保证留档可回溯。
