---
title: 评测指标、可视化工具与配置体系说明
date: 2026-09-22
tags:
  - Code_Docs
  - Metrics
  - Visualization
  - Configuration
---

# 📊 评测指标、可视化工具与实验配置体系说明

本文档详细说明 `d:\machine_learning` 下 `utils/metrics.py`、`utils/visualizer.py` 以及 `configs/default_config.yaml` 的实现与接口规范。

---

## 1. 核心评测指标模块 (`utils/metrics.py`)

模块已实现面向医学图像分割的全套客观定量评测体系，支持 PyTorch Tensor（自动 GPU/CPU 处理）与 NumPy 数组无缝兼容。

### 1.1 集合重合度指标
- **`calculate_dice(pred, target, threshold=0.5, smooth=1e-6)`**
  $$Dice = \frac{2 |P \cap T| + \epsilon}{|P| + |T| + \epsilon}$$
  支持批次 `(B, 1, H, W)`、`(B, H, W)` 或单张图 `(H, W)`。针对全负样本边缘场景（预测与真值皆空）精确返回 `1.0`。
- **`calculate_iou(pred, target, threshold=0.5, smooth=1e-6)`**
  $$IoU = \frac{|P \cap T| + \epsilon}{|P \cup T| + \epsilon}$$
  计算预测与真值的 Jaccard 重叠率。

### 1.2 边界贴合度指标 (HD95)
- **`calculate_hd95(pred, target, threshold=0.5, empty_value=100.0)`**
  - 利用 OpenCV `cv2.findContours` 提取预测掩码与真值掩码的闭合轮廓坐标集 $A$ 与 $B$。
  - 基于 `scipy.spatial.cKDTree` 高效检索最近邻双向欧氏距离分布，并由 `scipy.spatial.distance.directed_hausdorff` 锚定单向极大偏差：
    $$HD_{95}(A, B) = \max \left( P_{95}(\min_{b \in B} \|a - b\|), P_{95}(\min_{a \in A} \|b - a\|) \right)$$
  - **边缘场景保护机制**：
    - 若预测与真值均无病灶（双空），返回 `0.0`。
    - 若仅一方为空（漏检或严重误报），返回惩罚距离 `empty_value`（默认 `100.0`）。

### 1.3 像素级检测精度
- **`calculate_precision_recall(pred, target, threshold=0.5)`**
  返回 `PrecisionRecall(precision, recall)` 命名元组，支持属性访问（`.precision`, `.recall`）与解构赋值。

### 1.4 模型复杂度与算力分析
- **`compute_model_complexity(model, input_size=(1, 3, 256, 256), device='cpu')`**
  - 自动调用 `thop.profile` 计算模型的总参数量（单位：**Millions, M**）与浮点运算量（单位：**GigaFLOPs, G**）。
  - 若环境未装载 `thop`，自动无缝降级为精确参数累加与卷积理论 FLOPs 遍历估算。

### 1.5 批次指标聚合器
- **`MetricTracker(metrics=None)`**
  - 在验证集/测试集循环中逐批次累计各项指标（`dice`, `iou`, `hd95`, `precision`, `recall` 及任意自定义标量如 `loss`）。
  - `tracker.compute()` 输出各项指标的精确全局均值。
  - `tracker.summary()` 格式化输出对齐打印字符串。

---

## 2. 图像对比与训练监控可视化 (`utils/visualizer.py`)

### 2.1 四联预测对比图 (`save_prediction_comparison`)
- **面板 (a) Original Image**：自适应灰度或 RGB 原图展示。
- **面板 (b) Ground Truth Mask**：二值化标注金标准。
- **面板 (c) Predicted Probability Heatmap**：连续置信度预测热力图（搭配 `jet` 色板与侧边 Colorbar [0.0, 1.0]）。
- **面板 (d) Boundary Overlay**：原图背景叠加真值轮廓（**绿色**，GT）与模型预测轮廓（**红色**，Pred），并附带图例说明。
- 支持 300 DPI 高清导出，自动创建目标父级文件夹。

### 2.2 训练动态折线图 (`plot_training_curves`)
- 双子图布局：
  - **左图**：Train Loss 与 Val Loss 随 Epoch 演进曲线，自动标出并高亮**最低 Val Loss 点与 Epoch**。
  - **右图**：Train Dice 与 Val Dice 随 Epoch 演进曲线，自动标出并高亮**最高 Val Dice 点与 Epoch**。

---

## 3. 统一配置文件 (`configs/default_config.yaml`)

```yaml
dataset:
  img_size: 256
  batch_size: 8
  num_workers: 0

training:
  epochs: 30
  lr: 0.0003
  weight_decay: 0.0001
  early_stopping: 10

loss:
  lambda_bce: 1.0
  lambda_dice: 1.0
  lambda_boundary: 0.5

seed: 42
device: "cuda"
```

---

## 4. 单元测试与复现性验证

通过以下命令一键执行全部指标、可视化与配置解析验证：
```bash
python -c "
import utils, yaml
with open('configs/default_config.yaml') as f:
    cfg = yaml.safe_load(f)
print('Config loaded successfully!')
"
```
