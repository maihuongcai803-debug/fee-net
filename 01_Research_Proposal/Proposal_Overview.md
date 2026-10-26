---
title: 研究方案概览与论证 (FEE-Net)
date: 2026-09-21
tags:
  - Research_Proposal
  - Algorithm
  - FEE_Net
---

# 📑 研究方案：基于频域先验与边缘感知的轻量化医学图像分割网络 (FEE-Net)

## 1. 拟定英文论文题目
**FEE-Net: Frequency-guided Edge-aware Network for Accurate and Lightweight Medical Image Segmentation**
*(面向精准与轻量化医学图像分割的频域引导与边缘感知网络)*

---

## 2. 临床背景与核心痛点
1. **对比度低与边界模糊**：医学影像（如超声 BUSI、皮肤镜 ISIC、低剂量 CT）中，病灶组织与周围健康软组织的灰度相近，边缘极其模糊且形态不规则。
2. **常规深层网络的频域退化**：标准卷积与下采样（MaxPooling/Strided Conv）具有低通滤波性质，随着网络层数加深，高频纹理与微细边界信息被大量抹平，导致分割边界过平滑或漏诊。
3. **计算复杂度与落地难题**：Vision Transformer (ViT) 等大模型虽然具备全局建模能力，但计算开销巨大（$\mathcal{O}(N^2)$），难以在基层医疗设备或常规工作站上实时运行。

---

## 3. 核心算法创新设计 (Algorithm Novelty)

```mermaid
graph LR
    Input[输入图像 X] --> DWT[离散小波高频分解 DWT]
    Input --> SpatialEnc[空间编码器 CNN/Depthwise]
    
    DWT --> FreqBranch[频域高频交互分支 FHIB]
    SpatialEnc --> FEB[特征增强主干]
    
    FreqBranch --> EGM[边缘感知门控模块 EGM]
    FEB --> EGM
    
    EGM --> Dec[边缘引导解码器 Decoder]
    Dec --> PredSeg[病灶分割图]
    Dec --> PredEdge[辅助边缘预测图]
    
    PredSeg --> Loss[Compound Loss: Dice + BCE + Boundary Loss]
    PredEdge --> Loss
```

1. **频域先验分解（Discrete Wavelet Transform, DWT）**：
   - 将空间图像分解为低频逼近分量（LL）与三个高频细节分量（水平 LH、垂直 HL、对角 HH）。
   - 保留无损的多尺度高频边缘先验，避免传统池化带来的边缘丢失。
2. **边缘感知跨尺度门控模块（Edge-Gated Cross-Scale Module, EGM）**：
   - 将解码器深层语义特征与高频小波特征进行动态注意力门控，利用浅层高频边缘指导深层抽象语义的边界重构。
3. **复合边界损失函数（Boundary-aware Compound Loss）**：
   - 设计 $L_{total} = \alpha L_{BCE} + \beta L_{Dice} + \gamma L_{Boundary}$。
   - 显式惩罚轮廓距离偏差，使用 Hausdorff 距离近似或二阶拉普拉斯边缘轮廓匹配。

---

## 4. 数据集与基准任务
* **基准数据集**：
  * **ISIC 2018 (Skin Lesion)** 或 **BUSI (Breast Ultrasound Dataset)**
  * 数据特征：标准 2D 医学图像，病灶边界极具挑战性，学术界公认，评测公平且下载便捷。
* **评价指标**：
  * 区域重叠：Dice 相似系数 (DSC)、Mean IoU (mIoU)
  * 边界精度：95% 豪斯多夫距离 (HD95)、平均表面距离 (ASD)
  * 计算效率：参数量 (Params in M)、浮点运算量 (FLOPs in G)、推理速度 (FPS)

---

## 5. 论文发表目标
- 目标级别：**EI 检索国际会议或相关期刊**（如 IEEE EMBC, BHI, ICASSP, 或 Springer 计算机/医学影像工程期刊会议）。
- 论文规格：双栏 IEEE 格式，6-8 页，含完备的数学推导、定量消融实验与热力图对比。
