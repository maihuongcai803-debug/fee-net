# 实验留档 Exp-024：根源去伪存真、术语脱虚向实与保形风险控制 (Conformal Risk Control) 统计严谨化

## 1. 实验基本信息
- **实验日期**: 2026-09-23
- **实验编号**: `Exp-024`
- **关联文件**: 
  - [[utils/conformal_triage.py]] (新建：有限样本无分布假设保形风险控制)
  - [[tests/test_conformal_triage.py]] (新建：保形数学证明与单测)
  - [[models/modules/laya_visual_triage.py]] (规范命名与去包装)
  - [[models/modules/laya_cascade_router.py]] (规范命名与去包装)
  - [[models/modules/__init__.py]] (模块纯净导出与兼容性别名)
  - [[eval_laya_jev_cascade.py]] (评测术语去伪存真)
- **技术标签**: `#ConformalRiskControl` `#StatisticalRigor` `#ClopperPearson` `#DeHyping` `#FiniteSampleGuarantees`
- **核心目标**: 彻底回应同行评审对“Jev / Laya 概念包装过度、非自回归术语滥用、经验阈值缺乏统计泛化保障”的尖锐批评。从根源剖析学术原貌，肃清所有浮夸营销术语，用扎实严谨的有限样本保形风险控制 (Conformal Risk Control, Angelopoulos et al., 2022) 构建具备数学证明的临床漏诊率上限保障。

---

## 2. 根源问题深度剖析与对账

| 审查痛点与学术硬伤 | 根因与机理实质 | 本次落地改进方案 | 科学严谨性评定 |
| :--- | :--- | :--- | :---: |
| **1. "Jev / 非自回归" 概念严重错配与包装** | Jev (TypeSafe AI, 2026) 是 NLP/LLM 领域的结构化输出与非自回归语法约束解析库。将一个前馈式的标准 2D 卷积网络 (CNN) 冠以“非自回归 Jev 结构化分流”，在 CV 顶会顶刊评审中属于典型的“概念生搬硬套与术语通胀”。 | 彻底剔除所有“非自回归”、“Jev 启发”浮夸修饰。类重构为 `ClinicalVisualTriageNet`，决策函数重命名为 `triage_decision_structured`，保留原有别名向后兼容已有权重与外部调用。 | 彻底消除学术被拒雷点 |
| **2. "Laya" 专有词泛化不足** | 工程内部使用的系统代号易给审稿人留下“内部私有系统封装”的印象。 | 规范重命名为 `ClinicalCascadeRouter` 与 `ClinicalVisualTriageNet`，明确其为计算机视觉中经典而实用的“非对称级联推理路由器 (Asymmetric Cascaded Inference Router)”。 | 符合 IEEE TMI/MedIA 学术规范 |
| **3. 经验安全阈值 ($\tau$) 缺乏泛化保证** | 原有脚本仅在验证集上硬找 `recall == 1.0` 的最大阈值，此启发式极值在测试集或跨中心数据分布轻微漂移时极易导致严重漏诊。 | 引入统计学习前沿理论：**无分布假设有限样本保形风险控制 (Conformal Risk Control, CRC)**，利用 Clopper-Pearson 精确二项反演构建假阴性率（FNR）上界 $\le \alpha$ 的确定性数学保证。 | 具备数学定理支撑的临床严谨度 |

---

## 3. 数学理论推导：有限样本保形风险控制 (CRC)

### 3.1 假阴性率风险控制形式化
设患者样本为 $(X, Y) \sim \mathcal{D}$，其中 $Y \in \{0, 1\}$ 表示阴性/阳性，$f(X) \in [0, 1]$ 为视觉初筛网络的病灶置信度。初筛决策规则为：
$$\hat{Y}_\tau(X) = \mathbb{I}(f(X) \ge \tau)$$
临床假阴性率 (False Negative Rate, FNR，即漏诊率) 定义为损失函数 $L(\tau; X, Y) = \mathbb{I}(Y=1 \land \hat{Y}_\tau(X)=0)$。临床目标是在置信水平 $1 - \delta$ (如 95%) 下，保证期望风险不超过预设安全上界 $\alpha$ (如 5% 或 1%)：
$$\mathbb{P}\left( \mathbb{E}[L(\tau; X, Y) \mid Y=1] \le \alpha \right) \ge 1 - \delta$$

### 3.2 基于 Clopper-Pearson 精确二项反演的校准
在独立校准集 $\mathcal{D}_{\text{cal}} = \{(X_i, Y_i)\}_{i=1}^n$ 中的 $n_1$ 个阳性病灶样本上，对于任意候选阈值 $\tau$，漏诊病灶数为 $K(\tau) = \sum_{i: Y_i=1} \mathbb{I}(f(X_i) < \tau)$。

根据二项分布参数单侧置信上界 (Clopper & Pearson, 1934)：
$$\text{UCB}_{1-\delta}(K, n_1) = \begin{cases} 1 - \delta^{1/n_1}, & K = 0 \\ \text{BetaQuantile}\left(1 - \delta; K + 1, n_1 - K\right), & K > 0 \end{cases}$$

由保形风险控制定理 (Angelopoulos et al., 2021, 2022)：
我们选取满足条件的最大阈值 $\tau^*$：
$$\tau^* = \max \left\{ \tau \in [0, 1] : \text{UCB}_{1-\delta}(K(\tau), n_1) \le \alpha \right\}$$
- 若校准集阳性样本量 $n_1$ 较小，导致即便 $K=0$ 时 $\text{UCB} = 1 - \delta^{1/n_1} > \alpha$，算法自动保守回退到 $\tau^* = 0.0$（即不跳过任何样本，全部转入深层网络），彻底杜绝过度激进导致的临床隐患；
- 若样本量充分满足样本量界限 ($n_1 \ge \frac{\ln \delta}{\ln(1-\alpha)}$)，则可在数学保证 $\text{FNR} \le \alpha$ 的前提下自适应选取最优节能阈值 $\tau^* > 0$。

### 3.3 检验期 Wilson Score 置信区间
在独立测试队列上评估灵敏度 (Sensitivity) 与特异度 (Specificity) 时，放弃点估计，全面汇报 95% Wilson Score 置信区间：
$$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2m} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{m} + \frac{z^2}{4m^2}}}{1 + \frac{z^2}{m}}, \quad z = 1.96$$

---

## 4. 代码工程落地点与单测全绿记录

1. **`utils/conformal_triage.py`**：
   - 实现 `ConformalClinicalTriage` 类，包含 `calibrate()`, `predict()`, `evaluate_cohort()`, `save_calibration()`, `load_calibration()`。
   - 实现精确 Clopper-Pearson 与 Wilson Score 计算函数，支持单张与批处理张量输入。
2. **`models/modules/laya_visual_triage.py`**：
   - 命名纯净化：`ClinicalVisualTriageNet`（别名 `LayaVisualTriageNet`）。
   - 去伪存真：清理所有非自回归与 Jev 营销术语。
3. **`models/modules/laya_cascade_router.py`**：
   - 命名纯净化：`ClinicalCascadeRouter`（别名 `LayaCascadeRouter`）。
4. **全量 PyTest 单元测试验证**:
   - 新增 `tests/test_conformal_triage.py`（4 项保形单测全绿）。
   - 运行工程全量测试套件 `pytest tests/ -v`：
     ```text
     ============================= 20 passed in 9.40s ==============================
     tests/test_conformal_triage.py: 4 passed
     tests/test_fee_net_early_exit.py: 4 passed
     tests/test_fee_net_refinement.py: 6 passed
     tests/test_laya_cascade_router.py: 6 passed
     ```
     **20 / 20 项单测全部 100% 通过**，无回归 Bug。

---

## 5. 论文呈现与审稿抗辩建议
在论文的 Method 与 Discussion 章节中：
1. **不要再提 Jev 或非自回归**；
2. 架构命名统一为：**Two-Tier Asymmetric Cascade Architecture with Conformal Risk Control**；
3. 将系统亮点从“非自回归决策”切换为“**有限样本无分布假设的严格临床风险可控性 (Distribution-Free Conformal Risk Guarantee)**”，直接引用 Angelopoulos et al. (2022) 与 Vovk et al. (2005) 的奠基性理论，大幅提升审稿人的学术信任度。
