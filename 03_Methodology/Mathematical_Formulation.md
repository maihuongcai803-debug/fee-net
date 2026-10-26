---
title: 算法设计与数学推导公式体系
date: 2026-09-21
tags:
  - Methodology
  - Mathematics
  - LaTeX
---

# 📐 算法设计与核心数学公式推导

本笔记详细推导 **FEE-Net** 的数学理论基础，所有公式与变量定义已对齐 **IEEE 标准期刊/会议** 规范，后续可直接嵌入 LaTeX 论文第 III 节 (Methodology)。

---

## 1. 频域小波多尺度边缘解耦 (Wavelet Multi-scale Decomposition)

设输入医学图像为 $X \in \mathbb{R}^{H \times W \times C}$。传统下采样（如 MaxPooling）会破坏图像的高频分量：
$$
X_{\text{down}}(i, j) = \max_{u, v \in \Omega} X(2i+u, 2j+v)
$$
导致高阶导数信息与微细边缘结构丢失。

为了实现信息无损的边缘提取，FEE-Net 采用二维离散小波变换（2D Discrete Wavelet Transform, 2D-DWT）。利用低通滤波器 $f_L$ 与高通滤波器 $f_H$（以 Haar 小波为例）：
$$
f_L = \frac{1}{\sqrt{2}}[1, 1], \quad f_H = \frac{1}{\sqrt{2}}[-1, 1]
$$
在行列方向上进行卷积下采样，将图像分解为四个正交子带：
$$
X_{LL}, X_{LH}, X_{HL}, X_{HH} = \text{DWT}(X)
$$
其中：
- $X_{LL} \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times C}$ 为低频近似分量（保持全局宏观结构）；
- $X_{LH} \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times C}$ 捕获水平高频边缘（垂直方向梯度突变）；
- $X_{HL} \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times C}$ 捕获垂直高频边缘（水平方向梯度突变）；
- $X_{HH} \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times C}$ 捕获对角线高频细节与微小病灶斑点。

我们将高频细节分量沿通道堆叠，经过自适应投影层得到频域边缘先验特征：
$$
F_{\text{freq}} = \text{Conv}_{1\times 1}\left(\left[X_{LH} \,\|\, X_{HL} \,\|\, X_{HH}\right]\right) \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times C'}
$$

---

## 2. 边缘感知跨尺度门控模块 (Edge-Gated Cross-Scale Attention, EGM)

设来自解码器深层的语义特征为 $F_{\text{dec}} \in \mathbb{R}^{H_l \times W_l \times C_l}$，对应尺度的频域边缘特征为 $F_{\text{freq}} \in \mathbb{R}^{H_l \times W_l \times C_l}$。

为了滤除背景噪声并精准聚焦于病灶边界，设计动态双重门控注意力权重 $A_e$：
$$
Q_e = W_q * F_{\text{dec}}, \quad K_e = W_k * F_{\text{freq}}
$$
$$
A_e = \sigma \left( \mathcal{F}_{\text{pw}} \left( \text{ReLU} ( Q_e + K_e ) \right) \right) \in \mathbb{R}^{H_l \times W_l \times 1}
$$
其中 $*$ 表示卷积操作，$\mathcal{F}_{\text{pw}}$ 为点卷积（Pointwise Conv），$\sigma(\cdot)$ 为 Sigmoid 函数。

最终强化后的边界敏感特征表示为：
$$
\tilde{F} = F_{\text{dec}} + A_e \odot \left( \mathcal{F}_{\text{proj}}(F_{\text{freq}}) \right)
$$
$\odot$ 代表逐元素空间广播乘法。该机制使得只有高置信度的病理边缘先验被注入到深层语义中。

---

## 3. 复合边界感知损失函数 (Boundary-aware Compound Loss)

设预测病灶概率图为 $P \in [0, 1]^{H \times W}$，真实病灶标注为 $Y \in \{0, 1\}^{H \times W}$。

总损失函数定义为三项联合优化：
$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{BCE}} + \lambda_2 \mathcal{L}_{\text{Dice}} + \lambda_3 \mathcal{L}_{\text{Boundary}}
$$

### 3.1 像素级交叉熵与区域 Dice 损失
$$
\mathcal{L}_{\text{BCE}}(P, Y) = - \frac{1}{N} \sum_{i=1}^N \left[ Y_i \log(P_i) + (1 - Y_i) \log(1 - P_i) \right]
$$
$$
\mathcal{L}_{\text{Dice}}(P, Y) = 1 - \frac{2 \sum_{i=1}^N P_i Y_i + \epsilon}{\sum_{i=1}^N P_i + \sum_{i=1}^N Y_i + \epsilon}
$$

### 3.2 轮廓形态学缎带损失 (Boundary IoU Loss, CVPR 2021)
为避免传统二阶拉普拉斯算子对高频图像噪声的指数级放大与梯度抖动，本项目采用连续可微的形态学膨胀与腐蚀差分（Morphological Dilation minus Erosion）提取几何连续的边界缎带带状区域：

设边界厚度半径为 $d$（默认 $d=3$，局部感受野 $(2d+1) \times (2d+1) = 7 \times 7$）：
$$
\text{Dil}(X) = \max_{k \in \mathcal{N}_d} X_k = \text{max\_pool2d}(X, \text{kernel\_size}=2d+1, \text{stride}=1, \text{padding}=d)
$$
$$
\text{Ero}(X) = \min_{k \in \mathcal{N}_d} X_k = - \text{max\_pool2d}(-X, \text{kernel\_size}=2d+1, \text{stride}=1, \text{padding}=d)
$$
$$
\partial X = \text{Dil}(X) - \text{Ero}(X)
$$
对预测概率图 $P$ 与真实掩码 $Y$，分别提取连续边界缎带 $\partial P$ 与真值边界缎带 $\partial Y$（反向传播时对 $\partial Y$ 截断梯度 `detach`）。连续可微的 Boundary IoU / Dice 目标函数定义为：
$$
\mathcal{L}_{\text{Boundary-IoU}}(P, Y) = 1 - \frac{2 \sum_{i=1}^N \partial P_i \cdot \partial Y_i + \epsilon}{\sum_{i=1}^N \partial P_i + \sum_{i=1}^N \partial Y_i + \epsilon}
$$
若网络输出辅助高频边缘分支特征 Logits $P_{\text{edge}}$，则进一步施加连续边界交叉熵约束：
$$
\mathcal{L}_{\text{Edge-BCE}}(P_{\text{edge}}, \partial Y) = \text{BCEWithLogits}(P_{\text{edge}}, \partial Y)
$$
总复合损失默认权重设定为 $\lambda_{\text{bce}}=1.0, \lambda_{\text{dice}}=1.0, \lambda_{\text{boundary}}=0.5$。该损失具有严格有界性 $[0, 1]$ 与极高抗噪鲁棒性，在边缘扰动下梯度范数比传统拉普拉斯算子稳定降低逾 50 倍。详细实验留档参见 [[04_Experiment_Logs/Exp_012_Boundary_IoU_Loss_Overhaul|Exp-012]]。

---

## 4. 参数量与计算复杂度分析
与传统纯 ViT 模型的 $\mathcal{O}((HW)^2)$ 空间注意力复杂度相比：
- DWT 变换由无参数的小波卷积矩阵实现，时间复杂度仅为 $\mathcal{O}(HW)$；
- EGM 模块采用低通道深度可分离卷积，增加的 FLOPs $< 3\%$；
- 整个网络总参数量控制在 **4.5M ~ 6.8M** 之间，适合显存小于 6GB 的环境超高速推理。
