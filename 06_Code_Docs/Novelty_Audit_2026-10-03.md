# 论文立意与创新性评估（Novelty Audit，2026-10-03）

> 评估对象：`05_Paper_Draft/Draft_IEEE_FEE_Net.tex`（标题 *FEE-Net: Frequency-Guided Edge-Aware Network with Conformal Triage…*）
> 评估方法：逐条拆解论文的四项贡献，对每条做先前工作（prior-art）核查，并核对论文实际数字与公开文献的可比性
> 结论提要：**架构层面的创新性立不住**（属于已知组件的组合，且核心频域+注意力路线已有 2023–2024 年的同数据集更强结果）；**但"零泄漏评测协议 + 可证安全筛查"的方法学立意可以立住**——前提是（1）修正两处引文错误与一处方法学缺陷，（2）把定位从"新 SOTA 架构"改为"轻量工程化分割 + 有安全保证的筛查系统"，（3）下调投稿层级。

---

## 一、逐条创新点核查

| 论文声称的创新点 | 先前工作 | 判定 |
|---|---|---|
| **2D Haar DWT 正交分解高频提取** | 小波用于医学分割是十年以上成熟路线（WRANet, Complex & Intelligent Systems 2023；AdaptUNet+wavelet, Heliyon 2024）；项目自己的 `02_Literature_Review` 也已把 DWT 定位为"成熟数学工具" | ❌ 非创新（可作为工程选择） |
| **频域通道注意力 SCA（GAP+GMP + FC 瓶颈）** | 本质是 SENet (CVPR 2018) 的通道注意力，"频域"只是把池化换到小波子带上；CBAM/ECA 同属此类 | ❌ 非创新（改名） |
| **全景八邻域滤波 ODWF（1×3 + 3×1 + 3×3 depthwise）** | 3×3 depthwise 卷积本身即覆盖 8 邻域；与已有 1×3/3×1 并行相加是增量设计。项目 Exp-023 自己记录它是"修复 3×3 盲区" | ⚠️ 增量（且存在冗余） |
| **双线性上采样替代反卷积** | Odena et al. 2016 起为标准做法，论文自己已引用 | ❌ 非创新（已正确引用） |
| **边界 IoU 损失（形态学膨胀-腐蚀 ribbon）** | **El Jurdi et al., MICCAI 2021**（A Surprisingly Effective Perimeter-based Loss）用的正是 max-pool 膨胀 − min-pool 腐蚀取轮廓；**Karimi & Salcudean, MedIA 2022**（Boundary IoU）给出该度量；**IEEE TMI 2019**（Reducing the HD）已是形态学 HD 损失 | ❌ 非创新 + ⚠️ **代码与论文存在引文错误**（详见 §三.1） |
| **深监督 + 边缘头监督** | FPN/UNet++ 时代标准做法 | ❌ 非创新 |
| **双分支 GAP+GMP 分流头（4C）** | GAP 早退是标准做法（EENets 的 Pool/Bnpool 出口，arXiv 2409.05336）；小目标早退旁路问题也已被专门研究（arXiv 2309.03530） | ⚠️ 增量（工程修复有价值：GMP 确实解决微小息肉稀释） |
| **CRC 保形风险控制标定早退阈值（Clopper-Pearson）** | 这是**最容易被视为新颖、也最容易被查穿**的一条：医学分割阈值已有 Conformal Lesion Segmentation（split conformal 保证 FNR 上限，6 数据集含 MDSC-Colon）、ICIPCA 2025 的 CRC-FDR 框架；且已有工作（ML Research, PET/CT 肿瘤分割）**明确指出**这类二元漏诊损失常常**无法**在 α=δ=0.1 下被认证 | ⚠️ 应用层增量 + ⚠️ **方法学缺陷**（校准与评测同集，见 §三.2） |

## 二、最危险的事实：数字不占优

同一数据集（Kvasir-SEG）上已有更早、且报告更高 Dice 的小波+注意力工作：

| 工作 | 年份 | 报告 Dice | 备注 |
|---|---|---|---|
| **WRANet**（wavelet + residual attention，含息肉分割） | 2023 | **88.89%** | Complex & Intelligent Systems 9(6) |
| **AdaptUNet + wavelet**（结直肠息肉） | 2024 | 90.4%（CVC-300）；在 Kvasir-SEG/ETIS 上外部验证 | Heliyon 10(13)，已被引 21 次 |
| FEE-Net（本项目，2026-10-01 运行） | 2026 | 85.22%（仓库版重跑后待更新） | 仅与自训的 3 个经典基线对比 |

**含义**：论文目前把 FEE-Net 定位为"定义新的 Pareto 前沿 / 超越 SOTA"，一旦审稿人翻到 WRANet 或 AdaptUNet，"SOTA"与"频域创新"两条立意同时失守。**必须**改为"在严格零泄漏协议下、与同预算经典基线相比的轻量高精度实现"，并明确不做跨文献 SOTA 声明（论文 Limitations 已部分做到，需强化）。

## 三、必须立即修的两处硬伤

### 3.1 引文错误（学术诚信问题，优先级最高）
`utils/losses.py` L27 / L61 的 docstring 写：
```
Morphological Boundary Ribbon Extractor (CVPR 2021 Standard).
Boundary IoU Loss (CVPR 2021 Standard).
```
**这是错的**：形态学膨胀-腐蚀轮廓来自 **El Jurdi et al., MICCAI 2021**；Boundary IoU 来自 **Karimi & Salcudean, MedIA 2022**；形态学 HD 损失来自 **Karimi & Salcudean, IEEE TMI 2019**。没有 "CVPR 2021" 这一出处。
论文正文目前**完全没有引用**这三篇任何一篇——审稿人一旦核对，这是"不懂相关工作"的负面印象，且损失函数这一贡献点会被直接判为非创新且未溯源。

### 3.2 CRC 校准与评测同集（保证失效）
当前流水线（以及 Exp-029 实测的 v3 代码）在 **150 例测试集上同时**做 τ 扫描、Clopper-Pearson 上界标定 **和** FNR 汇报。conformal 风险控制的成立前提是**校准集与测试集可交换（exchangeable）**；在同一批样本上既标定又报告，形式上的 95% 保证不成立（这与"用测试集调参"是同一类问题）。
> 注：数值上 CP 上界确实很小（n=150, k=0 → 上界 ≈ 1.98% < 5%），但**保证的合法性**与数值无关。

**修法（已实施于生成器）**：在 **验证折（170 例）** 上标定 τ\*，在 **150 例测试集**上仅做一次认证与汇报；两者分开，并导出逐样本 risk 分数以便事后复核。

## 四、立意能否立住：结论与建议定位

### 4.1 分层级判断

| 目标层级 | 现状判断 | 主要障碍 |
|---|---|---|
| IEEE TMI / Medical Image Analysis | ❌ 立不住 | 架构增量、单一数据集、缺专用息肉 SOTA 对比、无跨数据集泛化 |
| MICCAI / ISBI / MIACS / EMBC（会议论文） | ⚠️ 勉强，需补对比与消融 | 需补 PraNet/UNet++/WRANet 等同期强基线 + v2 协议消融 + CRC 校准分离 |
| EI / 中文期刊（计算机学报、软件学报、生物医学工程学报等）/ 学生竞赛 | ✅ 可以立住 | 把定位改为"轻量 + 可证安全"，补齐引文与协议诚实性即可 |

### 4.2 建议的重构定位（保住立意的方式）

把论文主轴从"提出新架构"换成"**在严格无泄漏协议下，把一个轻量频域分割器做成可部署、可认证的筛查系统**"：

1. **主贡献（可立）**：零泄漏 v2 协议（150 例分层独立测试集，与所有折不相交，代码级断言）+ 单一真源可复现流水线（源码内联导入 + 溯源打包）。这是**方法学贡献**，同赛道 reviewer 认可，且对临床 CAD 社区有实际价值。
2. **次贡献（可立，但需降调）**：分流门的**有限样本安全保证**——但必须先做 §3.2 的校准分离，并在 Limitations 中引用"二元漏诊损失可能无法认证"的负面结果文献，说明适用边界。
3. **工程贡献（可立，勿称 SOTA）**：8.5M/12.4 GFLOPs 的轻量分割器，在**同预算同协议**下相对 U-Net/Att-U-Net/UNet++ 的一致增益 + Pareto 优势。措辞统一为"consistently outperforms identically-retrained classical baselines under identical budgets"。
4. **明确不声称**：架构新颖性（列出 §一 的组件出处）、跨文献 SOTA、筛查特异性（Kvasir-SEG 全阳性无法验证）、级联实测吞吐（该队列零 bypass）。

### 4.3 补强清单（按性价比）
1. ✅ 修正 `utils/losses.py` 引文 + 论文补 3 篇损失文献（**已做，见提交**）
2. ✅ CRC 校准/评测分离（**已做**）
3. Related Work 增加两段：小波+注意力息肉分割（WRANet 2023、AdaptUNet 2024）、医学分割中的保形阈值控制（CLS、ICIPCA 2025、ML Research 负面结果）
4. 补一组"同期强基线"对比：至少 PraNet（作者论文明说未比）+ WRANet/AdaptUNet（若可复现则引其报告值并标注非同协议）
5. Limitations 增列"架构组件多为已有方法的新组合"这一自评（诚实反而降低被拒风险）

## 五、一句话结论

> **架构创新性立不住，但方法学立意能立住。** 把论文从"我们提出了一个更强的新网络"改写为"我们给出一个可复现、零泄漏、轻量、可认证安全的临床筛查系统"，补齐引文、修好校准分离、下调投稿层级——这是一个诚实且能过审的 EI 级工作；坚持当前"SOTA 架构"叙事则大概率在 TMI/MedIA/MICCAI 被拒。
