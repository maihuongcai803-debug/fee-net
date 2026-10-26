---
title: "Exp-032 探针矩阵定稿：FEE-Net 与 U-Net Dice 实质持平、HD95 显著更优；v4 事故与 v6 定稿"
date: 2026-10-03
tags:
  - Experiment_Log
  - Ablation
  - Benchmark_Result
  - Negative_Result
---

# Exp-032：本地探针矩阵定稿 + v4 事故复盘 + v6(60 轮) 定稿启动

> 前序：[[Exp_031_Architecture_Aligned_Rerun_Results|Exp-031]]（v3 架构对齐重跑）
> 本文数据全部来自 `assets/probe_results/*.json` 与 `scratch/local_eval_v4_35ep.csv`，可逐项复核

## 1. 探针矩阵定稿（fold-0 验证集，680/170，60 轮，同种子同协议）

| 配置 | best val Dice | @ep | val HD95 (px) | 参数量 |
|---|---:|---:|---:|---:|
| **纯 U-Net** | **77.92** | 47 | 50.11 | 7.763M |
| **FEE-Net 完整损失（权威基准）** | **77.55** | 33 | **40.14** | 8.539M |
| FEE-Net w/o edge-BCE | 76.92 | 33 | 45.81 | 8.539M |
| FEE-Net w/o Boundary-IoU | 76.52 | 41 | 43.28 | 8.539M |
| FEE-Net w/o 深监督 | 74.69 | 40 | 50.21 | 8.539M |
| FEE-Net 纯 BCE+Dice（三项全关） | 74.60 | 24 | 48.17 | 8.539M |

### 1.1 三项辅助损失的净贡献（逐项剥离）

| 损失项 | 单独贡献 | 说明 |
|---|---:|---|
| 多尺度深监督 | **+2.86** | 最大单项贡献 |
| Boundary-IoU | **+1.03** | 同时把 HD95 从 43.28 改善到 40.14 |
| 边缘头 BCE | **+0.63** | 辅助监督中最弱的一项 |
| 三项联合 | **+2.95** | 单独相加为 +4.52，**非可加（差 −1.57）** |

> **表注必须写明非可加性**：三项单独相加 +4.52 与联合效应 +2.95 相差 1.57，审稿人一定会做这个加法。

### 1.2 关键结论（Dice 持平、HD95 更优）

- **Dice 实质持平**：FEE-Net 77.55 vs U-Net 77.92（**−0.37**），此前"落后 1.3 点"的判断来自一次被中断 run 的 ep55 读数，已作废；
- **HD95 显著更优**：FEE-Net **40.14** vs U-Net **50.11**（**−19.9%**），且明显优于去掉 Boundary-IoU 的 43.28 → **频域与边界监督确实交付了它承诺的边界保真度收益**；
- 结论改写为："在 680 张训练样本下，FEE-Net 以 +0.8M 参数量与 U-Net 持平的 Dice 精度，换来约 20% 更低的 95% Hausdorff 边界误差。"

## 2. v4 云端运行事故复盘

| 项目 | 事实 |
|---|---|
| 状态 | `KernelWorkerStatus.ERROR`（84 分钟） |
| 已完成 | 7 个模型训练 + 7 个检查点 + `triage_risk_scores.csv` + 扫描 CSV 落盘 |
| 失败点 | 混合队列单元 `NameError: name 'subprocess' is not defined` |
| 连带损失 | benchmark CSV 与图未落盘（它们在崩溃单元之后的单元里） |
| 根因反思 | 我本地验证时把 `subprocess` 注入 harness globals，**掩盖了缺 import**；且未防御性落盘 |

### 2.1 已修的三处

1. **混合单元自包含 `import subprocess`**（不再依赖前面单元的隐式 import）；
2. **混合单元整体包 try/except**：该实验失败不再中断整轮（它只是加分项）；
3. **评测单元末尾立即落盘 CSV**：不再等后续单元，避免"训练 84 分钟却丢结果"。

### 2.2 v4 检查点的本地重评（`tools/eval_checkpoints_local.py`，150 例独立测试集）

| 模型 | Dice | mIoU | HD95 | Precision | Recall | 判定 |
|---|---:|---:|---:|---:|---:|---|
| U-Net | 83.26 | 74.77 | 40.09 | 87.79 | 83.50 | valid |
| Attention U-Net | 80.90 | 71.74 | 39.64 | 86.46 | 80.12 | valid |
| UNet++ | 82.73 | 74.44 | 38.57 | 88.53 | 83.42 | valid |
| FEE-Net | 78.29 | 69.19 | 41.87 | 85.82 | 78.73 | valid |
| FEE-Net w/o DWT | 78.74 | 70.14 | 42.49 | 83.97 | 79.71 | valid |
| **FEE-Net w/o EGM** | **81.77** | 73.49 | **39.01** | 87.41 | 80.47 | valid |
| **FEENetEarlyExit** | **30.89** | 22.00 | — | — | 31.83 | ❌ **REJECTED** |

**两条新发现**：
1. **去掉 EGM 后反而涨 3.5 Dice**（81.77 vs 78.29），且 HD95 更好（39.01）——**EGM（粗预测语义门控 + ODWF）是当前配方下唯一明显负贡献的组件**，与 v3 测试集方向一致（80.86 vs 78.55），两次独立运行互相印证；
2. **EE 检查点为欠训坏点**：权重无非有限值，但范数系统性偏小（mean 4.37 / max 25.0，对比 FEE-Net 6.38 / 87.3）。本地 5 轮探针证明 **EE 训练路径本身正常**（val Dice 42.4→54.4，与 FEE-Net 同轨迹），故属该次运行的偶发问题，交由 v6 的 60 轮重跑判定。**在校验拦截下该行已被标记为"不得进入论文表格"。**

## 3. v6（Kernel Version 6）已启动

- **EPOCHS = 60**（v4 误用 35 轮，会系统性低估 FEE-Net：探针显示 35 轮仅 76.14、60 轮 77.55，而基线早早收敛）；
- 携带本次三项修复（subprocess 自包含 / try/except / 早落盘）；
- 新增 `model_profiles.json` 产物（thop MACs + CUDA Event 延迟 + 网关开销 + 硬件说明）→ 解决 12.59 / 0.94 GFLOPs 的溯源问题；
- 守护进程 `tools/watch_kaggle_kernel.py`（`9p3g3O`，每 2 分钟轮询，上限 200 分钟）完成后自动下载到 `scratch/kaggle_output_v6/`。

## 4. 工具与流程改进（本轮沉淀）

| 工具 | 作用 |
|---|---|
| `tools/local_probe.py` | 本地 GPU 探针（22–24 s/epoch），把配置筛选从 80 分钟压到 22 分钟 |
| `tools/run_probe_matrix.py` | 探针矩阵驱动（可重排优先级） |
| `tools/eval_checkpoints_local.py` | 本地重评任意检查点目录 + 数值校验（拦截 v3 坏行、v4 坏 EE） |
| `tools/ingest_kaggle_results.py` | 云端产物接入：数值校验 + 图暂存（**不写论文目录**）+ CRC 重算 |
| `tools/build_mixed_cohort.py` / `eval_mixed_cohort_gate.py` | 无泄漏混合队列构建 + 保形认证筛查 |

**流程教训（写给未来的自己）**：验证 notebook 单元时**绝不能向 globals 注入任何名字**——必须共享命名空间按序 exec（Kaggle 语义），否则 harness 会替你补上缺失的 import，把 bug 藏起来。v4 的 84 分钟就是这么丢的。

---

## 5. 修正：EE 异常的真实根因（Agent A 发现，本文 §2.2 结论作废）

我在 §2.2 把 v4 的 `FEENetEarlyExit` 异常（Dice 30.89 / Recall 31.83、权重范数偏小）判断为"欠训的偶发坏点"。**该判断错误**，Agent A 在 2026-10-03 17:05 定位到确定性根因：

> `tools/generate_kaggle_all_in_one_notebook.py` 的 `train_model` **验证循环**（第 548 行）调用
> `model(v_imgs, force_full=True)` 时**漏传 `return_raw_logits=True`**。
> `FEENetEarlyExit.forward(force_full=True)` 返回的是**概率**，而验证循环把它交给
> `MetricTracker(from_logits=True)` → **二次 sigmoid** → 验证 Dice 被严重歪曲
> → `best_val_dice` 曲线失真 → 保存了"按错误指标最优"的 epoch。

这完整解释了三个现象：验证 Dice 停在 30.89 量级、权重范数系统性偏小（欠训表象）、以及本地 5 轮探针训练路径却正常（探针不走验证循环选模型）。

**修复**：验证循环与测试循环统一使用 `model(x, force_full=True, return_raw_logits=True)`（A 已写入共享生成器）。修复后需由下一次云端运行（v5，60 轮）验证 EE 行。

**流程教训（补充）**：本次我在本地验证 notebook 单元时，harness 向 globals 注入了名字（`subprocess` 等），
(1) 掩盖了真实的缺 import，(2) 也没有覆盖"训练循环内部验证"这条路径。
→ 验证必须**共享命名空间按序 exec**（Kaggle 语义），且**关键指标路径要单独断言**（例如"验证与测试必须用同一输出约定"）。
