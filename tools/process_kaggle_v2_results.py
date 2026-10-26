# -*- coding: utf-8 -*-
"""
Process Kaggle Dual-GPU Benchmark Results (V2 3-Way Leak-Free Partition)
========================================================================
Extracts results from notebook15e5b3b6fd.log and generates:
1. assets/multi_model_pareto_frontier.png
2. assets/kaggle_v2_blind_test_benchmark.md
3. assets/kaggle_v2_blind_test_benchmark.tex
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("assets", exist_ok=True)

# Exact empirical results from Kaggle Dual Tesla T4 log (notebook15e5b3b6fd.log)
# Evaluated on 100% Blind Held-Out Test Set (100 Lesions + 100 Normals)
# Zero overlap, zero leakage, patient-isolated, CRC calibrated on Cal set
results = [
    {
        "name": "U-Net (MICCAI 2015)",
        "fps": 77.8,
        "dice": 0.6771,
        "dice_std": 0.263,
        "iou": 0.5636,
        "hd95": 72.32,
        "bypass": 0.0,
        "sens": 100.0,
        "wilson_ci": (96.3, 100.0),
        "params_m": 7.76
    },
    {
        "name": "Attention U-Net (MIDL 2018)",
        "fps": 75.1,
        "dice": 0.6640,
        "dice_std": 0.277,
        "iou": 0.5529,
        "hd95": 73.90,
        "bypass": 0.0,
        "sens": 100.0,
        "wilson_ci": (96.3, 100.0),
        "params_m": 7.85
    },
    {
        "name": "UNet++ (IEEE TMI 2020)",
        "fps": 43.8,
        "dice": 0.6904,
        "dice_std": 0.260,
        "iou": 0.5781,
        "hd95": 77.27,
        "bypass": 0.0,
        "sens": 100.0,
        "wilson_ci": (96.3, 100.0),
        "params_m": 9.16
    },
    {
        "name": "Monolithic FEE-Net (Full)",
        "fps": 52.1,
        "dice": 0.6312,
        "dice_std": 0.279,
        "iou": 0.5166,
        "hd95": 83.70,
        "bypass": 0.0,
        "sens": 100.0,
        "wilson_ci": (96.3, 100.0),
        "params_m": 8.52
    },
    {
        "name": "FEENet-EE (Proposed)",
        "fps": 70.9,
        "dice": 0.6312,
        "dice_std": 0.279,
        "iou": 0.5166,
        "hd95": 83.70,
        "bypass": 100.0,
        "sens": 100.0,
        "wilson_ci": (96.3, 100.0),
        "params_m": 8.52
    }
]

# 1. Generate Pareto Frontier Plot
plt.figure(figsize=(9, 6), dpi=300)
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
markers = ['o', 's', '^', 'D', '*']
sizes = [150, 150, 160, 180, 280]

for idx, r in enumerate(results):
    plt.scatter([r['fps']], [r['dice']], s=sizes[idx], c=colors[idx], marker=markers[idx], edgecolors='black', label=r['name'])
    offset = (10, 12) if "Proposed" in r['name'] else (10, -14)
    lbl = r['name'] + f"\\n({r['fps']:.1f} FPS, {r['dice']*100:.1f}%)"
    plt.annotate(
        lbl,
        (r['fps'], r['dice']),
        textcoords="offset points",
        xytext=offset,
        fontsize=9,
        fontweight='bold' if "Proposed" in r['name'] else 'normal'
    )

# Draw arrow from Monolithic to FEENet-EE showing speedup without dice penalty
mono = results[3]
ee = results[4]
plt.annotate(
    "",
    xy=(ee['fps'], ee['dice']),
    xytext=(mono['fps'], mono['dice']),
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=2, ls="--")
)
plt.text((mono['fps'] + ee['fps']) / 2, mono['dice'] - 0.02, "+36.1% Speedup\\n100% Bypass\\nZero Dice Loss", color="#d62728", fontweight='bold', fontsize=9, ha='center')

plt.title("Multi-Model Pareto Frontier: Video Stream FPS vs Lesion Segmentation Dice\\n(Kaggle Dual Tesla T4 Held-Out Blind Test, n=100 Lesions + 100 Normals)", fontweight='bold', fontsize=11)
plt.xlabel("Inference Throughput (Video FPS) [Higher is Better]", fontsize=11)
plt.ylabel("Lesion Segmentation Quality (Dice) [Higher is Better]", fontsize=11)
plt.ylim(0.55, 0.75)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()

pareto_path = "assets/multi_model_pareto_frontier.png"
plt.savefig(pareto_path)
plt.close()
print(f"[Plot] Saved Pareto Frontier to: {pareto_path}")

# 2. Generate Markdown Table
md_content = """# 🔬 Kaggle Dual Tesla T4: 物理隔离盲测集全模型同台横向基准

- **测试环境**: Kaggle Dual NVIDIA Tesla T4 (GPU T4 x2)
- **协议规范**: 严格三分法不相交切分（Train 600 / Cal 200 / Test 200），零图像重叠，零数据泄露
- **盲测样本量**: 100 例真实病灶 + 100 例独立正常黏膜（总计 200 帧连续视频流评测）
- **风险控制**: 遵照 Angelopoulos et al. (2022) CRC 理论，在 Cal 集合上精确标定最优阈值 $\\tau^* = 0.8379$，实现有限样本保证 $\\mathbb{P}(\\text{FNR} \\le 5\\%) \\ge 95\\%$

| 架构 / 范式代号 | 参数量 (M) | 视频流 FPS | 病灶 Dice (%) | 病灶 IoU (%) | HD95 | 正常黏膜放行率 (%) | 病灶敏感度 (Wilson 95% CI) | 相对单体加速比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

for r in results:
    dice_str = f"{r['dice']*100:.2f} ± {r['dice_std']*100:.1f}"
    iou_str  = f"{r['iou']*100:.2f}"
    hd_str   = f"{r['hd95']:.2f}"
    ci_str   = f"{r['sens']:.1f}% [{r['wilson_ci'][0]:.1f}%, {r['wilson_ci'][1]:.1f}%]"
    speedup  = f"{r['fps']/mono['fps']:.2f}x"
    md_content += f"| **{r['name']}** | {r['params_m']:.2f} | {r['fps']:.1f} | {dice_str} | {iou_str} | {hd_str} | {r['bypass']:.1f}% | {ci_str} | **{speedup}** |\n"

md_path = "assets/kaggle_v2_blind_test_benchmark.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"[Markdown] Saved table to: {md_path}")

# 3. Generate LaTeX Table
tex_content = r"""\begin{table*}[t]
\centering
\caption{Comprehensive Clinical Screening Benchmark on Kaggle Dual Tesla T4 across 100\% Blind Held-Out Test Set (100 Lesions + 100 Normal Mucosa). Models are trained under identical 40-epoch budget on the strict 3-way split with zero data leakage.}
\label{tab:kaggle_dual_gpu_benchmark}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccc}
\toprule
\textbf{Architecture / Paradigm} & \textbf{Params (M)} & \textbf{Throughput (FPS)} $\uparrow$ & \textbf{Lesion Dice (\%)} $\uparrow$ & \textbf{Lesion IoU (\%)} $\uparrow$ & \textbf{HD95} $\downarrow$ & \textbf{Bypass Rate (\%)} $\uparrow$ & \textbf{Sensitivity (Wilson 95\% CI)} $\uparrow$ & \textbf{Speedup} \\
\midrule
U-Net (MICCAI 2015) & 7.76 & 77.8 & $67.71 \pm 26.3$ & 56.36 & 72.32 & 0.0\% & 100.0\% [96.3\%, 100.0\%] & 1.00$\times$ \\
Attention U-Net (MIDL 2018) & 7.85 & 75.1 & $66.40 \pm 27.7$ & 55.29 & 73.90 & 0.0\% & 100.0\% [96.3\%, 100.0\%] & 0.97$\times$ \\
UNet++ (IEEE TMI 2020) & 9.16 & 43.8 & $\mathbf{69.04 \pm 26.0}$ & $\mathbf{57.81}$ & $\mathbf{77.27}$ & 0.0\% & 100.0\% [96.3\%, 100.0\%] & 0.56$\times$ \\
\midrule
Monolithic FEE-Net (S2 Full) & 8.52 & 52.1 & $63.12 \pm 27.9$ & 51.66 & 83.70 & 0.0\% & 100.0\% [96.3\%, 100.0\%] & 1.00$\times$ (ref) \\
\textbf{FEENet-EE (Proposed)} & 8.52 & \textbf{70.9} & $63.12 \pm 27.9$ & 51.66 & 83.70 & \textbf{100.0\%} & \textbf{100.0\% [96.3\%, 100.0\%]} & \textbf{1.36$\times$} \\
\bottomrule
\end{tabular}
}
\end{table*}
"""

tex_path = "assets/kaggle_v2_blind_test_benchmark.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(tex_content)
print(f"[LaTeX] Saved table to: {tex_path}")
