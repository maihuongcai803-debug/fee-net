# FEE-Net: 频域引导边缘感知分割 + 可证安全分流

[![repo](https://img.shields.io/badge/GitHub-maihuongcai803--debug%2Ffee--net-181717?logo=github)](https://github.com/maihuongcai803-debug/fee-net)

医学内镜图像分割研究工程。核心贡献是 **FEE-Net**（离散小波高频分解 + 边缘感知跨尺度门控 +
Boundary IoU 损失）与 **Laya 三级动态级联分流**（System 1 轻量筛查 → System 2 全量分割），
并以**有限样本保形风险控制（CRC）**为筛查环节提供可证安全界。

论文数字的权威来源：`checkpoints/v5_final/` + `assets/kaggle_v5_final/`
（Kaggle v5 终版基准，独立 150 例测试协议）。

> **仓库**：<https://github.com/maihuongcai803-debug/fee-net>

---

## 论文

| 文件 | 说明 |
|---|---|
| `05_Paper_Draft/FEE-Net_中文稿_官方模板版.pdf` | 中文定稿（EI 会议官方模板，10 页双栏） |
| `05_Paper_Draft/Manuscript_FEE_Net_2026.pdf` | 英文稿（IEEEtran） |
| `05_Paper_Draft/zh/` | 中文稿唯一真源：`content.py` + `meta_cn.py` + `build_docx_official.py` |
| `05_Paper_Draft/Manuscript_FEE_Net_2026.tex` | 英文稿 LaTeX 源 |

生成中文 Word：

```bash
python 05_Paper_Draft/zh/build_docx_official.py   # -> FEE-Net_中文稿_官方模板版.docx
```

---

## 目录

| 路径 | 内容 |
|---|---|
| `00_Index.md` | Obsidian 知识库主入口 |
| `01_` ~ `06_` | 选题 / 文献 / 方法推导 / 实验留档 / 论文 / 工程文档 |
| `models/` | 分割主干：`fee_net.py`、`fee_net_early_exit.py`、3 个基线、`modules/{dwt,egm}` |
| `laya/` | 分流子系统：视觉筛查头、临床分层、三级级联路由 |
| `utils/` | 损失、指标、可视化、保形风险控制 |
| `datasets/` | Kvasir-SEG 加载、triage 数据集构建、数据增强 |
| `experiments/` | 历史与一次性评测入口 |
| `tools/` | Kaggle 流水线、CRC 校准、绘图、探针矩阵 |
| `tests/` | 22 个 pytest 用例 |

架构细节见 [`06_Code_Docs/Project_Architecture.md`](06_Code_Docs/Project_Architecture.md)。

---

## 环境

```bash
pip install -r requirements.txt          # 核心：torch / numpy / opencv / scipy
pip install -r requirements-tools.txt    # 可选：文献翻译 pdf2zh 等
```

本地验证环境：Python 3.11 + torch 2.7.1+cu118。

---

## 复现命令

```bash
# 1) 训练单个模型（5 折）
python train.py --config configs/default_config.yaml --model feenet --fold 0

# 2) 标准评测（Dice / IoU / HD95）
python evaluate.py --checkpoint checkpoints/v5_final/best_feenet.pth

# 3) 三级级联分流基准
python laya/eval_cascade.py

# 4) 四大范式全景验证 + CRC 校准（Exp-026）
python experiments/run_practical_verification.py

# 5) 多折多种子 + Wilcoxon 显著性
python experiments/run_rigorous_evaluation.py

# 6) 生成 Kaggle 一体化 notebook
python tools/generate_kaggle_all_in_one_notebook.py

# 7) 测试
python -m pytest tests/ -q
```

---

## 数据与权重出处

- **数据**：Kvasir-SEG（公开内镜息肉分割集）+ 派生的混合队列（`tools/build_mixed_cohort.py`）。
  原始图像不入库（见 `.gitignore`），由构建脚本重建。
- **权重**：`checkpoints/` 不入库。论文引用权重固定在 `checkpoints/v5_final/`。
- **文献**：`papers/`（129 MB PDF）本地保留，不入库。

---

## 引用

```bibtex
@inproceedings{feenet2026,
  title     = {FEE-Net: Frequency-guided Edge-aware Network with Certifiable Safety Screening
               for Endoscopic Image Segmentation},
  author    = {TODO},
  booktitle = {IEEE ...},
  year      = {2026}
}
```
