---
title: 开发环境配置与轻量运行指南
date: 2026-09-21
tags:
  - Code_Docs
  - Environment
  - PyTorch
---

# ⚙️ 开发环境配置与轻量运行指南

## 1. 本地 Anaconda / Python 环境准备

推荐使用 Python 3.10+ 环境：

```bash
# 创建虚拟环境
conda create -n medseg python=3.10 -y
conda activate medseg

# 安装 PyTorch (根据你的 CUDA 版本选择，以下为 CUDA 11.8/12.1 推荐命令)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目其余依赖
pip install -r requirements.txt
```

---

## 2. 免费云端 GPU (Kaggle / Google Colab) 运行指南

本项目完全兼容云端免费算力：
1. **显存消耗**：Batch Size = 8 或 16 时，显存占用约 3.5GB ~ 5.5GB（Tesla T4 16GB 显存充裕率超 65%）。
2. **训练耗时**：
   - 在 ISIC 2018（~2000 张图像）或 BUSI（~780 张图像）上，100 个 Epoch 的收敛时间通常在 **45 分钟至 1.5 小时** 以内。
3. **Kaggle 一键同步**：
   - 可以直接将工程文件夹打成 zip 包上传为 Kaggle Dataset，或者在 Notebook 中直接 `!git clone` 运行。
