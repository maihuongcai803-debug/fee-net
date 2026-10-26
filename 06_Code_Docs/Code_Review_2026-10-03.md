# 项目复查报告（Round 2）：2026-10-03

> 复查范围：Oct-01 整改完成后的仓库全状态（commit `88ae30c` 之后）
> 复查方式：git 状态核查 + pytest 22/22 + v2 splits 完整性 + 逐 Cell 解析 `kaggle_feenet_baselines_all_in_one.ipynb` + 与仓库 `models/` 代码逐项比对
> 结论：**工程侧整改全部生效、无回归**（22/22 测试通过、v2 splits 150 例独立且与所有折不相交、Git 26.36 MiB 无膨胀、工作区干净）。但存在 **3 个影响论文可信度的问题**，根因集中在"实验在 notebook 内联代码里跑，与仓库代码脱钩"。

---

## ✅ 已确认健康的部分（无回归）

| 项目 | 状态 |
|---|---|
| pytest 全量测试 | 22 passed（16.9s） |
| v2 splits 协议 | test=150，与 5 折 train/val 完全不相交，folds 均为 680/170 |
| Git 仓库 | 26.36 MiB，`scratch/`、`*.log` 已正确忽略，工作区干净 |
| 我上轮的修复 | 全部在位未被覆盖（splits v2 / config 接通 / metrics from_logits / 自检修复 / evaluate 严格加载） |
| Exp-028 组件 | SCA、ODWF `conv_dir_diag`、GMP 双分支、stop-gradient（`detach` 8 处）均已进入 benchmark 代码 |
| 审计整改 | Pareto 假点已清除（notebook `fps_plot = fps_vals`，仅绘实测点） |

---

## 🔴 P0-1：论文主模型 ≠ 仓库主模型（复现性缺口）

benchmark 里的 `FEE-Net (Ours Full)` 实际是 `FEENetEarlyExit` 的薄包装：

```python
class FEENet(nn.Module):          # notebook 内联，非仓库 models/fee_net.py
    def __init__(self, in_channels=3, num_classes=1, base_channels=48):
        self.ee = FEENetEarlyExit(in_channels, num_classes, base_channels)
    def forward(self, x):
        if self.training:
            seg, edge, _, _ = self.ee(x)     # edge / aux 输出被直接丢弃
            return seg
```

实测差异（notebook 代码 vs 仓库代码）：

| 组件 | 仓库 `models/fee_net.py` | notebook 内联版 |
|---|---|---|
| 多尺度深监督头 | `deep_sup4/3/2` + 0.3 权重 aux loss | **完全缺失**（`aux_preds` 出现 0 次） |
| 边缘头监督 | `head_edge` + `loss_edge_bce`（λ_b 附加项） | **未参与损失**（forward 丢弃 edge） |
| `coarse_head` 语义先验 | 有 | 无 |
| 参数量 | 8.539M | 8.339M |

**后果**：Table 1 的 85.22% 无法用仓库 `train.py` 复现——仓库版训练会启用深监督与边缘监督，是**另一个（未验证的）模型**。这也解释了此前 8.339M vs 8.539M 的参数口径分歧。

## 🔴 P0-2：论文损失函数公式与实际训练配方不符

论文 §Compound Loss 写的是
`L = L_BCE + L_Dice + λ_b·L_BoundIoU + λ_t·L_CE + λ_ds·L_deep`（λ_t=0.5, λ_ds=0.3），
但 benchmark 训练循环中 `return_aux=False`、`aux_preds=0`，即**深监督项与边缘 BCE 项从未被优化**。Boundary-IoU（作用于 seg 本身）与 BCE/Dice 有效，但 λ_ds 与边缘监督是"写了但没跑"。

**建议**：二选一——(a) 论文按实际配方改写并删除 λ_ds/edge 项；或 (b) 让 notebook 复用仓库 `train.py` 的复合损失（推荐 (b)，同时解决 P0-1）。

## 🔴 P0-3：全实验数字来自 notebook 内联代码副本，与仓库代码脱钩

notebook 完整复制了 `UNet`、`AttentionUNet`、`UNetPlusPlus`、`DWT_2D`、`SpectralChannelAttention`、`EdgeGatedModule`、`BilinearUpBlock`、`EarlyExitTriageHead`、`FEENetEarlyExit`、`BoundaryIoULoss`、`CompoundLoss`——即 `models/` 与 `utils/` 的**双重实现**。

这正是历史缺陷的根因：此前 UNet 解码器通道不匹配 bug、参数量口径分歧、旧架构 checkpoint 静默半载入，都源自"两份实现漂移"。仓库侧 `train.py` + `models/` 在 v2 协议下**从未重训**（本地 checkpoint 全部为 09-23/09-24，v2 协议建立于 09-24 之后），即仓库主链路至今没有任何 v2 协议下的验证。

**建议**：让 notebook 通过 `sys.path` 导入仓库 `models/`、`utils/`（Kaggle 侧用 `%%capture` git clone 或打包上传），实现单一真源。

---

## 🟠 P1 级问题

1. **根目录 `kaggle_feenet_baselines_all_in_one.py`（48.9 KB）是审计修复前的旧提取**：仍含 `fps_plot = fps_vals + [285.0]`（L1023）与旧 tau 扫描 `linspace(0.01, 0.90, 90)`，与已修复的 `.ipynb` 不一致 → 误用风险（有人拿 .py 跑就会复现造假点）。
2. **图注残留未实测数字**：notebook Pareto 图注释文本仍写 `Throughput: 285+ FPS | FLOPs: 1.45G`（ipynb L1202）。上轮审查建议的"1.45/12.44 ≈ 8.6× 吞吐余量（推算）"口径未同步进代码。
3. **论文无法编译**：`05_Paper_Draft/figure_qualitative_comparison.png` 缺失（`\includegraphics` 引用 2 张图，仅 pareto 存在）。该图在 Kaggle 产物 `feenet_baselines_kaggle_results.zip` 内，本地/仓库均无。
4. **v2 协议下无正式消融**：all-in-one 只跑 5 个模型，w/o DWT、w/o EGM 变体未纳入；论文消融章节目前依赖旧协议（v1、旧骨干）的 preliminary 数字并已标注口径，属可接受但偏弱。
5. **`evaluate.py` 的 canonical checkpoint 指向旧架构权重**：`unet_fold0_seed42_best.pth` / `fee_net_fold0_seed42_best.pth` 等仍是 6.90M 旧架构，本地跑 `evaluate.py` 会触发架构漂移告警并产出无效指标（v2 协议的新权重只存在于 Kaggle，未回传）。

## 🟡 P2 级问题（工程卫生）

6. **4 个旧版 5-fold notebook 仍留在根目录**（`kaggle_baselines_5fold_benchmark`、`kaggle_ablations_5fold_benchmark`、`kaggle_fee_net_5fold_benchmark`、`kaggle_feenet_early_exit_benchmark_v2`），均为 v1 时代产物，已被 all-in-one 取代。
7. **根目录日志散落**：`notebook0e7b0400ab.log` 应移入 `scratch/`（该目录已被 gitignore，是正确的临时产物区）。
8. **临时目录 `scratch/` 已有 203 个文件**（含 177 张 jpg 预览图 + 19 个 cell 提取 + 云端日志），已被正确忽略，但需定期清理以免磁盘膨胀。

---

## 修复优先级建议

| 优先级 | 事项 | 影响 |
|---|---|---|
| ① | notebook 改为导入仓库 `models/`、`utils/`（单一真源） | 消除 P0-1/P0-3 根因，让论文数字可复现 |
| ② | 用仓库 `train.py` 复合损失（含深监督 + 边缘监督）重跑 all-in-one | 消除 P0-2，模型与论文描述一致 |
| ③ | 补 `figure_qualitative_comparison.png`（从 Kaggle 产物 zip 提取） | 论文可编译 |
| ④ | 删除/归档根目录旧 `.py` 与 4 个旧 notebook，清理根目录 log | 消除误用与目录混乱 |
| ⑤ | 把 Kaggle 产出的 v2 checkpoint 回传本地，替换旧架构权重 | `evaluate.py` 本地可用 |
| ⑥ | v2 协议下补 w/o DWT、w/o EGM 消融（1 折即可） | 强化论文消融章节 |
