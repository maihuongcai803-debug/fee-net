# 项目结构优化方案（2026-10-03 盘点，2026-10-26 已执行 A~D）

> 盘点方式：目录体积/文件计数、`git ls-files` 追踪面、模块引用计数、硬编码路径扫描
> 原则：**不改变任何科学结论与已发表数字**，只调整组织形态；每步可独立回滚（纯 `git mv`）
> 状态：**A / B / C / D 已于 2026-10-26 执行完毕**（当时阻塞的 Kaggle v5 已定稿、论文已收尾）；
> E（工程化）部分完成（README 已补），F（索引一致性）已完成。执行结果见文末「六、执行记录」。

## 一、现状诊断（实测数据）

| 维度 | 实测 | 问题 |
|---|---|---|
| 根目录 | 13 个散落文件，其中 **8 个 py（3847 行）** | 入口脚本、知识库索引、依赖清单混在一层 |
| 论文 PDF | `papers/` **43MB，15 个 PDF 全部进 Git** | PDF 不可压缩，是仓库体积主要来源 |
| Laya 子项目 | `models/modules/laya_*` **1147 行** + `train_laya_triage.py`(693) + `eval_laya_jev_cascade.py`(906) + `eval_dual_system.py`(412) + `tools/laya_router.py` | 一个独立的" triage 级联"子系统寄居在分割网络包里，`models/` 无法作为纯分割构件发布 |
| 遗留数据生成器 | `datasets/dataset_loader.py` 499 行中约 240 行是**合成息肉数据生成器**，含 **3 处硬编码 `d:/machine_learning`** | 真实主干已换成 `real_medical_loader.py` + Kaggle 流水线；该文件只剩历史价值 + 3 个硬编码路径 |
| `tools/` | 22 个条目，其中 ≥5 个已被取代 | `generate_kaggle_notebook.py`、`generate_kaggle_early_exit_notebook_v2.py`、`build_kaggle_notebooks.py` 已被 `generate_kaggle_all_in_one_notebook.py` 取代；`plot_pareto_clean.py` 已被 notebook 内置实测 Pareto 取代 |
| 实验索引 | `04_Experiment_Logs/Exp_Index.md` **未收录 Exp-030** | 索引与日志漂移（Exp-027~029 仅各 1~3 处提及） |
| 工程化 | 无 README、无 `pyproject.toml`/pytest 配置、无 CI；`tests/` 4 文件 22 用例 | 新人/未来 agent 无统一入口；今天对 notebook 生成器的验证只能手工做 |
| 资产 | `assets/` 8MB 41 文件，混有 v1 时代基准表 | 论文用图与历史产物未分层 |

## 二、目标结构

```
machine_learning/
├── README.md                  # 新增：布局 + 入口 + 复现命令 + 数据出处
├── pyproject.toml             # 新增：依赖 + extras + pytest 配置（合并两个 requirements）
├── 00_Index.md ~ 06_Code_Docs/ # Obsidian 知识库（保持 .obsidian 入库）
├── configs/                   # 训练/评测配置（已被 train.py 真正读取）
├── src/  ← 建议新增            # 纯分割库：models / utils / datasets / tests
│   ├── models/  (unet, attention_unet, unet_plus_plus, fee_net, fee_net_early_exit, modules/{dwt,egm})
│   ├── utils/   (losses, metrics, visualizer, conformal_triage)
│   ├── datasets/(real_medical_loader, transforms)
│   └── tests/
├── laya/      ← 新增（隔离）  # triage 级联子系统：modules + train/eval/router
├── experiments/               # 一次性/历史评测入口（eval_dual_system、run_* 等）
├── tools/                     # 仅保留活跃工具（生成器、profiler、watcher、翻译工具）
├── scripts/archive/           # 已归档（8 个文件，不再动）
├── data/ checkpoints/ scratch/  # 全部 gitignore
└── papers/                    # 移出 Git（保留本地）
```

> 是否引入 `src/` 布局：**可选**。收益是"可 pip 安装的库"边界清晰；代价是所有 import 路径与
> 34 篇实验留档/论文中的 `models.xxx` 引用需要复核。若时间紧，优先做下面 A~C，src 布局可跳过。

## 三、动作包（按性价比排序）

### A. 仓库瘦身（低风险，立即可做）
1. `git rm -r --cached papers` + `.gitignore` 加 `papers/`（本地保留 43MB PDF）
2. `assets/` 分层：`assets/paper/`（论文用图）、`assets/archive/`（v1 历史表格与图）
3. `.pytest_cache/`、`__pycache__/` 已在 ignore；`scratch/` 定期清理

**收益**：clone 体积预计从 ~26MB 降到 <5MB；**风险**：0（不改代码）

### B. 根目录收敛（低风险，纯 `git mv`）
- 保留根层：`train.py`、`evaluate.py`（论文与文档引用的规范入口）、`README.md`、`00_Index.md`、`GEMINI.md`、两个 requirements
- 移入 `experiments/`：`eval_dual_system.py`、`eval_laya_jev_cascade.py`、`run_practical_verification.py`、`run_rigorous_evaluation.py`、`run_statistical_eval.py`
- 移入 `laya/`：`train_laya_triage.py`
- 同步更新：留档中出现的文件路径（可用 `git mv` 保留历史，Obsidian 链接需抽查）

### C. Laya 子项目隔离（中风险，一次性）
- `models/modules/laya_{cascade_router,clinical_triage,visual_triage}.py` → `laya/modules/`
- `eval_laya_jev_cascade.py` → `laya/eval_cascade.py`；`tools/laya_router.py` → `laya/router.py`
- 影响面：`eval_laya_jev_cascade.py` 被 11 处引用、`laya_visual_triage` 被 5 处；需全量改 import 并跑测试
- **收益**：`models/` 变成纯分割构件包，论文主线（FEE-Net + CRC）不再被 triage 子系统噪声干扰
- **注意**：论文 System 1/2 叙事会引用这些文件，迁移后需同步留档与 `03_Methodology/System1_System2_Framework`

### D. 遗留代码清理（低风险）
- `datasets/dataset_loader.py`：合成数据生成器部分迁至 `tools/generate_synthetic_dataset.py`，并**清除 3 处硬编码 `d:/machine_learning`**；或整体标记 deprecated
- `tools/` 归档已被取代的 5 个：`generate_kaggle_notebook.py`、`generate_kaggle_early_exit_notebook_v2.py`、`build_kaggle_notebooks.py`、`plot_pareto_clean.py`、`retry_download.py` → `scripts/archive/`

### E. 工程化补齐（低风险，高长期收益）
1. `README.md`：项目一句话定位、目录图、5 条复现命令（生成本地 benchmark / Kaggle 推送 / 论文表格）、数据与权重出处声明
2. `pyproject.toml`：把 `requirements.txt`（核心）与 `requirements-tools.txt`（transformers/pdf2zh）合并为 extras；内含 `[tool.pytest.ini_options]`
3. **`tests/test_notebook_generator.py`（今日经验自动化）**：
   - 调用 `build_all_in_one_notebook()` 生成到 tmp
   - 解析全部单元语法、断言关键不变量：内联了 `models/fee_net.py` 源码、`splits.json` 含 150 test、**不存在硬编码 FPS 绘图点**、provenance 断言存在、训练单元调用 `CompoundLoss(seg, edge, ...)`
   - 这条测试可永久防止"notebook 与仓库代码再次脱钩"
4. 远端 GitHub Actions 跑 pytest（需先建远端仓库，当前仓库无 remote）

### F. 文档一致性（低风险）
- `04_Experiment_Logs/Exp_Index.md` 补录 Exp-030 并复核 027~029 条目
- 统一"实验留档 ↔ 论文"引用路径（A/B/C 迁移后必做）

## 四、执行顺序建议

1. **kernel v3 结果回传**（进行中）→ 先更新论文数字与图表（依赖真实数据）
2. A（瘦身）→ B（根目录）→ D（遗留清理）→ F（索引）→ E（工程化）
3. C（Laya 隔离）单独一次提交，改完立刻跑 `pytest` + 关键脚本冒烟
4. 每步一个 commit，commit message 标注是否影响论文/留档引用

## 五、不建议现在做的

- ❌ 重命名 `models/`→`src/models/`：收益主要是"可发布"，但会波及全部 import 与留档引用，等论文定稿后再考虑
- ❌ 动 `checkpoints/`：等 v3 权重回传后按"旧架构归档"处理，避免误删可回溯产物
- ❌ 合并 `04_Experiment_Logs` 旧日志：历史可回溯性是项目核心资产，保持原样

---

## 六、执行记录（2026-10-26）

前置条件已满足：Kaggle v5 终版基准定稿（Exp-033/034），论文中文稿完成官方模板排版，
内核流水线无在跑任务，可安全动目录。

| 动作 | 结果 | 验证 |
|---|---|---|
| **A 仓库瘦身** | `papers/`、`data/mixed_cohort_train/`、`data/kvasir_triage_test/` 解除 Git 跟踪（本地保留）；`.gitignore` 补 `papers/` 与派生数据规则 | `git ls-files` 5600+ 二进制条目退出索引 |
| **B 根目录收敛** | 根层只留 `train.py` / `evaluate.py`；`eval_dual_system.py`、`run_*.py` ×4 → `experiments/`；`train_laya_triage.py`、`eval_laya_jev_cascade.py` → `laya/` | 9 个入口 `--help` 全通 |
| **C Laya 隔离** | `models/modules/laya_*.py` ×3 → `laya/modules/`；`tools/laya_router.py` → `laya/router.py`；`models/modules/` 仅剩 `dwt` + `egm` | `pytest` 22 通过 |
| **D 遗留清理** | `tools/` 31 → 23：4 个 notebook 生成器收敛为 `generate_kaggle_all_in_one_notebook.py`；3 版 watcher 收敛为 `_v5`；删 `plot_pareto_clean.py`、`retry_download.py` | 引用方仅历史文档 |
| **磁盘清理** | `scratch/` 1.4 GB → 1.5 MB；v5 终版权重迁入 `checkpoints/v5_final/`、结果迁入 `assets/kaggle_v5_final/`、最终 triage 头迁入 `checkpoints/` | 7 个权重 MD5 逐一校验一致 |
| **E 工程化** | 新增 `README.md`；重写 `Project_Architecture.md` | — |
| **F 索引一致性** | `00_Index.md` 论文与代码段落按新路径重写；`Laya_Routing_Guide.md` 5 处路径更新 | — |

**未执行项及理由**
- `src/` 布局：方案中即标注"可选"，会波及全部 `models.xxx` import 与留档引用，等论文定稿后再议。
- `checkpoints/` 整体归档：5 折权重与消融变体是论文可回溯性证据，仅新增 `v5_final/`，未删旧权重。
- `pyproject.toml` / CI：仓库暂无 remote，等投稿定稿后再补。

**过程中发现并拦下的风险**
- `scratch/kaggle_v5_results/` 中的 7 个终版权重在 `checkpoints/` **无任何副本**，是论文全部数字的唯一来源；
  若直接按"重复输出"删除将不可恢复。已先迁入 `checkpoints/v5_final/` 并逐一 MD5 校验后再清理。
- `scratch/triage_head_stage2e.pth` 为 Exp-034 两阶段修复的最终筛查头，同样唯一，已迁入 `checkpoints/`。
- `scratch/kaggle_output_v5/data/kvasir_splits.json` 与 `data/kvasir_seg/splits.json` **内容不同**，
  为 v5 独立测试协议所用划分，已保留为 `assets/kaggle_v5_final/kvasir_splits_v5.json`。
- 顺带修复 `experiments/run_practical_verification.py` 中 argparse help 字符串的 `%` 未转义 bug（`--help` 原本直接崩溃）。
