---
title: 实验留档 Exp_005 双系统临床预筛与协同分割实验评测 (Laya S1 + FEE-Net S2)
date: 2026-09-22
tags:
  - Experiment
  - DualSystem
  - Laya
  - ClinicalTriage
  - GreenAI
---

# 🧪 实验留档：Exp_005_Dual_System_Laya_FEENet

> [!CAUTION] ⛔ 【已废弃 / 模拟队列玩具数据 / 严禁写入论文】
> 本实验评估使用的 500 例队列为本地数学脚本模拟生成的假样本，分流机制为硬编码规则，不包含真实病理特征。真实临床双系统/早退基准请参考 **FEENetEarlyExit 共享骨干 + CRC 保形风控** 评测体系 ([[04_Experiment_Logs/Exp_024_Root_Purification_and_Conformal_Risk_Control|Exp-024]]、[[04_Experiment_Logs/Exp_025_FEENet_EarlyExit_Pipeline_Integration_and_Gradient_Decoupling|Exp-025]]、[[04_Experiment_Logs/Exp_026_Comprehensive_Practical_Verification_and_CRC_Calibration|Exp-026]])。详细归档见 [[04_Experiment_Logs/Exp_Archive_Legacy_Prototypes|历史实验归档]]。**本实验所有算力与漏诊数据不可用于论文发表**。

## 1. 实验基本信息
- **实验日期**：2026-09-22
- **实验编号**：Exp-005
- **运行平台**：本地 NVIDIA GeForce RTX 4060 Laptop GPU | PyTorch 2.7.1+cu118
- **关联文档**：[[00_Index|主索引]] | [[04_Experiment_Logs/Exp_003_Laya_Integration_and_System1_Routing|Laya初步集成笔记]] | [[04_Experiment_Logs/Exp_Index|实验矩阵索引]]

---

## 2. 实验目的与动机
在真实医院与人群防癌普查（如内镜大肠癌筛查、乳腺超声、皮肤癌筛查）场景中：
- 真实世界检查群体中，**约 80%~90% 的患者影像表现为正常组织/阴性**，仅有 10%~20% 存在良性或恶性病灶。
- **痛点**：传统单阶段 AI 系统（Single-System）强制对每一位到诊患者、每一帧图像都调用昂贵的像素级分割网络（如 90M 参数的 MedSAM 或重型 CNN），导致 GPU 算力极度浪费、设备发热严重、门诊检查排队延迟过长。
- **动机**：验证所构建的 **“System 1 (Laya 快速临床决策与视觉预筛) + System 2 (FEE-Net 高精度边缘分割)”** 双系统协同范式：
  1. 通过 System 1 极速初筛（<1ms），快速放行（Bypass）正常人群，释放 GPU 资源；
  2. 仅对可疑与高危患者自动唤醒 System 2 展开微观频域边缘分割；
  3. 严密验证高危病灶安全门限，确保 **0% 漏诊率 (False Negative Rate = 0.0%)**。

---

## 3. 涉及文件与代码改动
| 文件路径 | 模块类型 | 核心改动说明 |
| :--- | :--- | :--- |
| [`models/modules/laya_clinical_triage.py`](file:///d:/machine_learning/models/modules/laya_clinical_triage.py) | 临床决策与视觉初筛 | 实现 `ClinicalTriageModule` 语言征象分类器与 `LayaVisualTriageNet`（0.006M 超轻量网络，0.042 GFLOPs）。 |
| [`models/fee_net.py`](file:///d:/machine_learning/models/fee_net.py) | 核心分割模型 | 支持接收先验权重 `clinical_prior` 并实现 eval 模式兼容。 |
| [`eval_dual_system.py`](file:///d:/machine_learning/eval_dual_system.py) | 临床队列评测脚本 | 模拟 500 例临床队列（80% 正常、15% 良性、5% 高危），自动化对比传统单阶段系统与双系统在算力、延迟、吞吐量和安全性上的表现。 |

---

## 4. 实验结果与对比分析 (500 例队列实测)

执行命令：`python eval_dual_system.py --ckpt checkpoints/fee_net_best.pth --samples 500 --device cuda`

### 4.1 综合性能量化对比表
| 评测指标 | 传统单系统 (仅 System 2 全量分割) | 本文双系统 (Laya S1 预筛 + FEE-Net S2 分割) | 收益与临床价值 |
| :--- | :---: | :---: | :---: |
| **System 1 决策耗时** | N/A (强行全量分割) | **< 1.0 ms** | 极速实时初筛 |
| **正常病例旁路比例 (Bypass Rate)** | 0 / 500 (0.0%) | **393 / 500 (78.6%)** | 免除近 80% 无意义分割计算 |
| **队列总计算量 (Total FLOPs)** | 5665.0 GFLOPs | **1233.3 GFLOPs** | **节省 78.23% GPU 算力 (Green AI)** |
| **单例平均端到端延迟 (Latency)** | 5.83 ms / 例 | **2.00 ms / 例** | **2.92x 极速推理** |
| **系统临床吞吐量 (Throughput)** | 171.6 FPS | **500.9 FPS** | **提升近 3 倍吞吐能力** |
| **高危恶性病灶灵敏度 (Sensitivity)** | 100.00% | **100.00%** (25/25) | 零漏诊 |
| **高危病灶假阴性率 (FNR)** | 0.00% | **0.00%** | **绝对符合医疗临床安全底线** |

### 4.2 核心结论
1. **绿色高效能计算 (Green AI)**：
   在真实筛查分布下，双系统架构直接消减了 **78.23%** 的无效 GPU 浮点运算与显存占用，使得高精度的医疗 AI 系统能够在单张消费级显卡甚至工控机上稳定达到 **500+ FPS** 的超高吞吐率。
2. **临床安全保障机制 (Safety Guardrail)**：
   Laya System 1 设置非对称置信度门限，宁可过度触发 System 2，也绝不放行任何高危可疑组织，实测高危病灶漏诊率为 **0.00%**，真正达成“高吞吐”与“高可靠”的统一。

---

## 5. 保存的产物文件
- **评测图表**：[`assets/dual_system_clinical_triage_comparison.png`](file:///d:/machine_learning/assets/dual_system_clinical_triage_comparison.png) (包含延迟对比、吞吐量对比、算力节省与病灶分流饼图)
- **评测报告**：[`assets/dual_system_evaluation_report.md`](file:///d:/machine_learning/assets/dual_system_evaluation_report.md)
