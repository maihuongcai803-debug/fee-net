---
title: 实验与代码修改留档模板
date: 2026-09-21
tags:
  - Experiment
  - Template
---

# 🧪 实验留档：[实验代号，如 Exp_001_Baseline_UNet]

## 1. 实验基本信息
- **实验日期**：YYYY-MM-DD
- **实验版本/Tag**：v0.1.0
- **运行平台**：本地 RTX 3060 / Kaggle (T4 x 2) / Colab
- **关联文档**：[[03_Methodology/Network_Architecture|网络架构]] | [[00_Index|主索引]]

---

## 2. 实验目的与动机
> 简要说明为什么要进行本次修改或实验（例如：验证基础 U-Net 在 ISIC2018 上的基准分割表现，为后续引入频域小波分支做对比参照）。

---

## 3. 代码变更记录 (Code Changes)
| 文件路径 | 变更类型 (新增/修改/重构) | 核心变动说明 |
| :--- | :--- | :--- |
| `models/unet.py` | 新增 | 实现标准 4 层编码器-解码器 U-Net 结构 |
| `utils/losses.py` | 修改 | 新增 Compound Loss 结合 BCE 与 Dice |

### 核心代码片段 / 关键函数变更：
```python
# 示例：记录关键改动点
def compute_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(torch.sigmoid(pred), target)
    return bce + dice
```

---

## 4. 实验配置与超参数 (Hyperparameters)
- **数据集**：ISIC 2018 (Train: 1816, Val: 259, Test: 519)
- **输入分辨率**：256 $\times$ 256
- **Batch Size**：16
- **优化器**：AdamW (lr=1e-4, weight_decay=1e-2)
- **学习率策略**：CosineAnnealingLR (T_max=100, eta_min=1e-6)
- **训练轮数 (Epochs)**：100 (Early Stopping patience=15)
- **随机种子**：42

---

## 5. 实验结果与评测指标 (Results & Metrics)

### 5.1 定量评估表
| 模型 / 变体 | Dice (%) | mIoU (%) | HD95 (voxel) | Precision (%) | Recall (%) | 参数量 (M) | FLOPs (G) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (U-Net) | 88.52 | 80.21 | 12.34 | 89.10 | 88.02 | 13.4 | 14.2 |
| **本次修改版** | **--** | **--** | **--** | **--** | **--** | **--** | **--** |

### 5.2 训练曲线与收敛性
- Loss 曲线表现：（平稳下降 / 是否过拟合）
- 最佳权重保存于：`checkpoints/best_model_exp_001.pth`

### 5.3 定性分割效果可视化
- 预测掩码 vs 真实标注 (Ground Truth) 对比图存档于 `assets/exp_001_visual.png`

---

## 6. 分析与结论 (Discussion & Conclusion)
- **实验结论**：
- **存在的问题与瓶颈**：
- **下一步改进计划**：
