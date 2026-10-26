---
title: "Exp_002_Baselines_and_Dataset_Pipeline: Standard Models and Medical Pipeline"
date: "2026-09-22"
tags:
  - Experiment
  - Baselines
  - Dataset
  - Pipeline
---

# 🧪 实验留档：Exp_002_Baselines_and_Dataset_Pipeline

> [!CAUTION] ⛔ 【已废弃 / 仅保留早期代码结构原型】
> 本实验引入的 `data/synthetic_medseg/` 属于早期数学合成病灶，已在 [[04_Experiment_Logs/Exp_006_Reviewer_Critique_and_Improvement_Plan|Exp-006]] 中被严格否定并彻底废弃。数据流水线已被真实世界官方数据集 **Kvasir-SEG** ([[04_Experiment_Logs/Exp_009_Kvasir_SEG_Real_Dataset_and_5Fold_CV|Exp-009]]) 及 [[04_Experiment_Logs/Exp_027_Codebase_Hygiene_and_Independent_Test_Protocol|Exp-027 v2 协议]] 彻底取代。详细归档见 [[04_Experiment_Logs/Exp_Archive_Legacy_Prototypes|历史实验归档]]。**本实验合成数据及指标不可用于论文**。

## 1. 实验基本信息
- **实验日期**：2026-09-22
- **实验编号**：`Exp_002_Baselines_and_Dataset_Pipeline`
- **运行平台**：本地 PyTorch 2.7.1 + CUDA 11.8
- **关联文档**：[[03_Methodology/Network_Architecture|网络架构]] | [[04_Experiment_Logs/Exp_Index|实验索引]] | [[06_Code_Docs/Project_Architecture|代码架构]]

---

## 2. 实验目的与动机
1. **建立公平基线网络**：
   - 实现医学图像分割黄金标准 Baseline：4 层 U-Net (Ronneberger et al., MICCAI 2015)。
   - 实现空间门控基线 Attention U-Net (Oktay et al., MIDL 2018)。
   - 保持严格相同的通道设置：编码器 [32, 64, 128, 256]，Bottleneck 512，解码器 [256, 128, 64, 32]，输出 1x1 卷积预测 logits，为后续与创新网络 FEE-Net 开展严谨公正的对比评测奠定基石。
2. **构建全自动医学图像数据流水线**：
   - 构建 `MedicalSegDataset` 与 `transforms.py`，支持无缝兼容 Tensor / NumPy / PIL 的多模态医学图像增强（旋转、水平/垂直翻转、对比度调节、ColorJitter、标准化）。
   - 内置高质量临床拟真病灶生成引擎（涵盖超声低回声衰减、皮损黑色素网络、CT 软组织磨玻璃低对比病灶与多谐波非凸病灶轮廓），自动生成 600 train / 150 val / 150 test 样本至 `data/synthetic_medseg/`，实现开箱即用的零等待闭环实验。

---

## 3. 代码变更记录 (Code Changes)

| 文件路径 | 变更类型 | 核心变动说明 |
| :--- | :--- | :--- |
| `models/unet.py` | 新增 | 标准 4 级 U-Net，DoubleConv 卷积块，支持转置卷积/双线性上采样，参数量 7.76M |
| `models/attention_unet.py` | 新增 | Attention U-Net，含空间注意力门控模块 `AttentionGate`，参数量 7.85M |
| `datasets/transforms.py` | 新增 | 同步空间几何变换与纯图像亮度/对比度变换，兼容 Tensor、NumPy 与 PIL |
| `datasets/dataset_loader.py` | 新增 | `MedicalSegDataset` 与病灶生成器，提供 `get_dataloaders` 工厂函数 |

---

## 4. 模型架构与参数量对比 (Architecture Sanity Check)

| 模型名称 | 编码器通道 | Bottleneck | 解码器通道 | 门控机制 | 可训练参数量 | 输出尺寸 (Batch=2, 256x256) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNet** | [32, 64, 128, 256] | 512 | [256, 128, 64, 32] | 无 | 7,763,041 (~7.76M) | `(2, 1, 256, 256)` |
| **Attention UNet** | [32, 64, 128, 256] | 512 | [256, 128, 64, 32] | AttentionGate | 7,851,773 (~7.85M) | `(2, 1, 256, 256)` |
| **FEE-Net (所提)** | [32, 64, 128, 256] | 512 | [256, 128, 64, 32] | DWT + EGM | ~5.48M | `(2, 1, 256, 256)` |

---

## 5. 数据流水线与拟真基准验证

1. **自动合成病灶数据集**：
   - 存储路径：`d:/machine_learning/data/synthetic_medseg/`
   - 划分规模：Train: 600, Val: 150, Test: 150 (总计 900 样本)
   - 图像模态仿真：
     - **Ultrasound**: 低回声暗区 + 散斑噪声 (Speckle Noise) + 垂直声衰减与声影。
     - **Dermoscopy**: 皮肤色素渐变 + 晕影 (Vignette) + 黑色素不规则网状分布。
     - **CT**: 软组织密度基底 + 磨玻璃浸润晕 (Ground-Glass Opacity) + 量子噪声。
   - 轮廓与边缘真实性：多谐波极坐标扰动生成非凸天然解剖轮廓，高斯模糊模拟部分容积效应 (Partial Volume Effect) 模糊边缘。
2. **端到端批次校验**：
   - DataLoader 批量生成：`images.shape = (8, 3, 256, 256)`, `masks.shape = (8, 1, 256, 256)`。
   - 掩码取值范围严格二值化在 `{0.0, 1.0}`。
   - 端到端前向传播与反向求导验证均 100% 通过。

---

## 6. 结论与后续计划
- 基线模型与数据流水线全部通过单元测试与集成测试。
- 下一步计划：实现 `train.py` 与 `evaluate.py`，启动基线与 FEE-Net 统一横向对比实验。
