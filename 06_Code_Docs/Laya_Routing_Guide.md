---
title: Laya 决策模型与任务分流器使用指南
date: 2026-09-22
tags:
  - Code_Docs
  - Laya
  - Dispatcher
  - Usage_Guide
---

# 📖 Laya 极速决策与任务分流器使用指南

本工程集成了 Convai Innovations 开发的开源轻量决策模型 **Laya**，提供两大核心功能模块：
1. **`laya/router.py`**：工程与智能体调度层（System 1 任务路由器）；
2. **`laya/modules/laya_clinical_triage.py`**：算法与临床推理层（System 1 临床初筛门控）。

---

## 1. 快速开始与命令行测试

### (1) 测试智能体任务路由器
```bash
python laya/router.py
```
* **输出示例**：
  ```text
  [任务]: 需要设计带有小波高频融合的编码器架构并推导梯度
   -> 决策路由: ALGO_ARCH_DEEP (置信度: 95.80%)
   -> 分配模型: pro | 对应 Agent: Core Algorithm Architect
   -> 响应延迟: 1.8 ms [local_fallback / hf_pipeline]
  ```

### (2) 测试临床先验预筛模块
```bash
python laya/modules/laya_clinical_triage.py
```
* **输出示例**：
  ```text
  [病例描述]: 肠镜检查：降结肠见一处大小约 15mm 扁平隆起，表面充血腺管结构不规整
   -> 预筛等级: HIGH_RISK_LESION (置信度: 90.00%)
   -> 是否触发 FEE-Net 精准分割: True | 临床处置: SEGMENT_HIGH_PRECISION
   -> 推理耗时: 0.12 ms
  ```

---

## 2. Python 代码集成调用

### 场景一：在自动化脚本中做任务分流
```python
from laya.router import LayaTaskRouter

router = LayaTaskRouter()
result = router.route_task("批量训练 UNet 和 Attention UNet 模型 25 个 epoch")

print("派发目标:", result["target_route"])       # BASELINE_DATA_PIPELINE
print("推荐模型级别:", result["recommended_model"]) # flash
print("负责角色:", result["agent_role"])          # Baseline & Data Engineer
```

### 场景二：在临床推理管道中结合 FEE-Net
```python
import torch
from models.modules.laya_clinical_triage import ClinicalTriageModule
from models.fee_net import FEENet

# 1. 实例化模块
triage = ClinicalTriageModule()
feenet = FEENet(in_channels=3, num_classes=1)

# 2. 模拟输入
clinical_notes = "回盲部粘膜平整光滑，未见息肉及异常隆起"
image_tensor = torch.randn(1, 3, 256, 256)

# 3. System 1 极速决策
decision = triage.evaluate_risk(clinical_notes)

# 4. 门控判断
if decision["trigger_feenet"]:
    print("触发深度分割...")
    feenet.eval()
    with torch.no_grad():
        mask = feenet(image_tensor)
else:
    print("System 1 判定为正常，直接放行免除 GPU 密集运算！")
```

---

## 3. 双模热备与离线高可用设计

为了确保无论在本地实验室无网环境、Kaggle/Colab 环境还是离线服务器中均能 100% 稳定运行，`LayaTaskRouter` 具备**自适应双模热备机制**：
* **模式 A (`laya_hf_pipeline`)**：当本地具备 Hugging Face 权重与网络连接时，调用完整的预训练序列分类管道；
* **模式 B (`local_fallback`)**：当处于离线或网络波动时，自动降级至基于关键词词频与温度 Softmax 的本地极速引擎，保证调度零阻塞、零异常。
