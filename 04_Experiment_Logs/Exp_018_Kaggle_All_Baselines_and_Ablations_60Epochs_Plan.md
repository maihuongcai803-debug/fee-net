# 实验留档 Exp-018：全量基线与消融模型 60 轮 5 折交叉验证升级方案 (Kaggle 双卡并行)

> [!NOTE] ⚠️ 【训练方案与超参设计有效 / 协议已升级至 v2 splits】
> 本方案设计的 60 轮统一收敛超参及 6 大模型 5 折对齐策略极为严谨。本方案制定时使用的是初代 `splits.json`（v1 划分，1000 例 5 折且无独立测试集）。在执行新一轮 Kaggle 云端训练时，必须基于 [[04_Experiment_Logs/Exp_027_Codebase_Hygiene_and_Independent_Test_Protocol|Exp-027]] 确立的 **v2 splits 协议**（850 例 5 折分层 CV + 150 例全局独立测试集）进行重跑与评估。详细归档见 [[04_Experiment_Logs/Exp_Archive_Legacy_Prototypes|历史实验归档]]。

## 1. 实验基本信息
- **实验日期**: 2026-09-23
- **实验编号**: `Exp-018`
- **训练环境**: Kaggle Cloud GPU (Dual Tesla T4 x2, 30GB VRAM, 4-core vCPU, 30GB RAM)
- **数据集**: 官方 Simula Kvasir-SEG (1000 例真实病变与掩膜，严格 5-Fold 分层划分 `splits.json`)
- **核心目标**: 彻底消除此前 25 轮（局部基线）与 60 轮（FEE-Net 最终版）的实验协议偏差，将全部 3 款主流基线模型与 3 款核心消融变体统一步调至 **真 5 折、60 轮长周期收敛、余弦退火、自动记录曲线与结构化打包** 的最高学术规范。
- **涉及模型清单 (共 6 款模型 $\times$ 5 折 = 30 个独立收敛 Run)**：
  1. **现代顶刊基准**: `UNet++` (Nested U-Net, IEEE TMI 2020, 9.16M, 34.62G)
  2. **经典分割基线**: `U-Net` (Ronneberger et al., MICCAI 2015, 7.76M, 13.75G)
  3. **空间门控基线**: `Attention U-Net` (Oktay et al., MIDL 2018, 7.85M, 14.03G)
  4. **频域消融变体**: `FEE-Net (w/o DWT)` (移除离散小波分解，采用标准池化)
  5. **门控消融变体**: `FEE-Net (w/o EGM)` (移除边缘共注意力门控，采用简单相加)
  6. **损失消融变体**: `FEE-Net (w/o Boundary Loss)` (移除边界形态学损失，仅保留 BCE+Dice)

---

## 2. 实验目的与改进动机

1. **严格同构对比，堵死审稿人一切质疑**：
   - 此前 FEE-Net 在 Kaggle 上训练了 60 轮，Dice 达到 81.69% $\pm$ 1.28%；而此前测试的基线与消融是 25 轮状态下的数据。
   - 极其挑剔的审稿人会质疑：“FEE-Net 之所以好，是因为它训练得更久（60 轮 vs 25 轮）”。
   - 将所有基线与消融全部统一按 60 轮长周期跑满 5 折，让每一个模型都充分达到平坦收敛平台（Flat Basin），实现百分之百的公平竞技。
2. **自动化过程留档与高分辨率收敛曲线绘制**：
   - 脚本不再单纯 `print`，而是将每轮的 `train_loss`、`val_dice`、`val_iou`、`val_hd95` 实时存入数组并导出为结构化 JSON；
   - 自动生成对比折线图（Figure: All Models Convergence Dynamics），直观展示 FEE-Net 与 UNet++、U-Net 以及消融变体的收敛过程差异。
3. **支持 Kaggle 双 Notebook 双轨并发，加速训练**：
   - Kaggle 允许同时运行 2 个 GPU Notebook。
   - 将 30 个 run 划分为两个独立的 Notebook：
     - **Notebook A (`kaggle_baselines_5fold_60epochs.ipynb`)**：专注于 3 款基线（UNet++、U-Net、Attention U-Net）的 5 折；
     - **Notebook B (`kaggle_ablations_5fold_60epochs.ipynb`)**：专注于 3 款消融变体（w/o DWT、w/o EGM、w/o BL）的 5 折；
   - 两台云端虚拟机同时起跑，总等待时间缩短 50%，预计 3~3.5 小时内全部自动完成打包。

---

## 3. 超参数与统一训练配置规范

| 训练配置项 | 统一参数设定 | 设计与临床理由 |
| :--- | :--- | :--- |
| **基础分辨率** | $256 \times 256 \times 3$ | 兼顾内窥镜病灶形态与 GPU 吞吐量 |
| **交叉验证折数** | 严格 5-Fold Stratified | 依据 `splits.json`，无任何跨折数据泄露 |
| **训练总轮数** | **60 Epochs** | 彻底进入平坦极值盆地，确保所有基线充分收敛 |
| **批大小 (Batch Size)**| **16** (梯度累加 `accum_steps=2`) | 等价于 Batch Size 32，双卡 T4 每卡仅 8，避免 UNet++ 密集跳连显存溢出与闪退 |
| **全量内存直读 (RAM Cache)**| 启动一次性预载 1000 例至 RAM (~250MB) | 彻底消除训练过程中的磁盘反复 IO、`glob` 扫描与 OpenCV 解码，CPU 磁盘开销归零 |
| **DataLoader 进程数** | `num_workers=0, pin_memory=False` | 直接在主进程纳秒级内存切片，杜绝子进程 IPC 通信开销与 `/dev/shm` 共享内存耗尽 |
| **多线程限制** | `cv2.setNumThreads(0)`, `torch.set_num_threads(2)` | 严格压制 OpenCV/OpenMP 多核竞争，杜绝 CPU 跑满 100% 导致的系统卡死 |
| **GPU 向量化验证** | Dice/IoU 全 GPU 矩阵运算 + 快速轮廓 HD95 | 验证耗时从每轮 30 秒暴跌至 0.16 秒，消除 CPU 验证算力峰值 |
| **优化器** | `AdamW` ($\text{lr}=3\times 10^{-4}$, weight\_decay=$1\times 10^{-4}$) | 配合 `optimizer.zero_grad(set_to_none=True)` 及时释放显存梯度缓存 |
| **显存主动回收** | 每轮与跨折执行 `gc.collect()` + `empty_cache()` | 杜绝长期 5 折运行中的显存碎片堆积与内存泄漏 |
| **学习率调度** | `CosineAnnealingLR` ($T_{\max}=60, \eta_{\min}=1\times 10^{-6}$) | 后期平滑微调，促进平坦收敛 |
| **混合精度** | `torch.amp.autocast('cuda')` | 显存减半，加速 1.8 倍 |
| **早停机制** | `patience=12`, 仅在 $\ge 30$ 轮激活 | 防止欠早停，确保充分探索 |
| **数据增强** | 随机水平翻转、垂直翻转、90度旋转 | 模拟内窥镜探头旋转与黏膜视角变化 |

---

## 4. 预期产出与结构化交付物

1. **基线检查点与结果包**:
   - `baselines_5fold_results.zip` 包含：
     - `unet_fold0..4_best.pth` (5个权重)
     - `attention_unet_fold0..4_best.pth` (5个权重)
     - `unet_plus_plus_fold0..4_best.pth` (5个权重)
     - `baselines_5fold_history.json` (逐轮详细曲线数据)
     - `baselines_convergence_curve.png` (高清收敛图)
2. **消融检查点与结果包**:
   - `ablations_5fold_results.zip` 包含：
     - `fee_no_dwt_fold0..4_best.pth`
     - `fee_no_egm_fold0..4_best.pth`
     - `fee_no_bl_fold0..4_best.pth`
     - `ablations_5fold_history.json`
     - `ablations_convergence_curve.png`
3. **最终论文 Table 1 & Table 2 升级**:
   - 所有基线与消融在 60 轮下的真实 5 折跨折均值 $\pm$ 跨折方差全部填满，论文数据无懈可击！
