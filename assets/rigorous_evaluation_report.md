# Rigorous Multi-Fold & Multi-Seed Statistical Benchmark Report

- **Dataset:** Simula Kvasir-SEG (Genuine Endoscopic Polyps)
- **Evaluated Folds:** [0]
- **Evaluated Seeds:** [42]
- **Reference Target Model:** fee_net
- **Hardware Platform:** CUDA

### Rigorous Quantitative Medical Segmentation Benchmark on Genuine Kvasir-SEG

| Method / Architecture           | Params (M) | FLOPs (G) | Dice (%)        | mIoU (%)        | HD95 (voxel)    | Wilcoxon p-val | Significance    |
| ------------------------------- | ---------- | --------- | --------------- | --------------- | --------------- | -------------- | --------------- |
| U-Net (Ronneberger et al.)      | 7.76       | 13.75     | 80.30 +/- 21.05 | 71.00 +/- 22.92 | 42.55 +/- 38.77 | 2.5427e-09     | p < 0.001 (***) |
| Attention U-Net (Oktay et al.)  | 7.85       | 14.03     | 79.24 +/- 21.43 | 69.66 +/- 23.42 | 42.75 +/- 37.57 | 4.7485e-09     | p < 0.001 (***) |
| UNet++ (Nested SOTA)            | 9.16       | 34.62     | 81.59 +/- 20.72 | 72.84 +/- 23.08 | 39.50 +/- 37.83 | 8.7511e-14     | p < 0.001 (***) |
| FEE-Net (w/o DWT Frequency)     | 6.88       | 14.53     | 71.14 +/- 26.62 | 60.66 +/- 26.99 | 51.07 +/- 39.33 | 0.8982         | p = 0.8982 (ns) |
| FEE-Net (w/o Edge-Gated Module) | 6.85       | 12.86     | 69.26 +/- 25.83 | 58.10 +/- 26.55 | 58.58 +/- 39.64 | 0.0035         | p = 0.0035 (**) |
| FEE-Net (w/o Boundary Loss)     | 6.89       | 14.68     | 70.85 +/- 25.20 | 59.79 +/- 25.90 | 54.57 +/- 39.12 | 0.0201         | p = 0.0201 (*)  |
| FEE-Net (Proposed Full)         | 6.89       | 14.68     | 71.73 +/- 26.00 | 61.17 +/- 26.57 | 55.92 +/- 39.58 | Ref            | Proposed Target |

**Notes & Statistical Significance:**
- Metrics reported as Mean +/- Std. For multi-run configurations (folds/seeds), true Run-level Mean +/- Std is computed.
- For single-run configurations, Sample-level Mean +/- Std across the evaluation set is reported.
- Paired two-sided Wilcoxon signed-rank test compares each baseline paired against FEE-Net (Proposed Full).
- Significance stars: * p < 0.05, ** p < 0.01, *** p < 0.001, ns: not statistically significant (p >= 0.05).

## LaTeX Source Code
```latex
\begin{table*}[htbp]
\centering
\small
\caption{\textbf{Rigorous Quantitative Benchmark on Kvasir-SEG Real Endoscopic Dataset.} Values represent Mean $\pm$ Std. Paired Wilcoxon signed-rank tests compare each baseline against the proposed FEE-Net architecture.}
\label{tab:rigorous_evaluation}
\begin{tabular}{lccccccc}
\toprule
\textbf{Method / Architecture} & \textbf{Params (M)} & \textbf{FLOPs (G)} & \textbf{Dice (\%)} & \textbf{mIoU (\%)} & \textbf{HD95 (voxel)} & \textbf{Wilcoxon p-val} & \textbf{Significance} \\
\midrule
U-Net (Ronneberger et al.) & 7.76 & 13.75 & 80.30 $\pm$ 21.05 & 71.00 $\pm$ 22.92 & 42.55 $\pm$ 38.77 & $2.54 \times 10^{-9}$ & $p < 0.001$ ($^{***}$) \\
Attention U-Net (Oktay et al.) & 7.85 & 14.03 & 79.24 $\pm$ 21.43 & 69.66 $\pm$ 23.42 & 42.75 $\pm$ 37.57 & $4.75 \times 10^{-9}$ & $p < 0.001$ ($^{***}$) \\
UNet++ (Nested SOTA) & 9.16 & 34.62 & 81.59 $\pm$ 20.72 & 72.84 $\pm$ 23.08 & 39.50 $\pm$ 37.83 & $8.75 \times 10^{-14}$ & $p < 0.001$ ($^{***}$) \\
FEE-Net (w/o DWT Frequency) & 6.88 & 14.53 & 71.14 $\pm$ 26.62 & 60.66 $\pm$ 26.99 & 51.07 $\pm$ 39.33 & 0.8982 & $p = 0.8982$ (ns) \\
FEE-Net (w/o Edge-Gated Module) & 6.85 & 12.86 & 69.26 $\pm$ 25.83 & 58.10 $\pm$ 26.55 & 58.58 $\pm$ 39.64 & 0.0035 & $p = 0.0035$ ($^{**}$) \\
FEE-Net (w/o Boundary Loss) & 6.89 & 14.68 & 70.85 $\pm$ 25.20 & 59.79 $\pm$ 25.90 & 54.57 $\pm$ 39.12 & 0.0201 & $p = 0.0201$ ($^{*}$) \\
FEE-Net (Proposed Full) & 6.89 & 14.68 & 71.73 $\pm$ 26.00 & 61.17 $\pm$ 26.57 & 55.92 $\pm$ 39.58 & Ref & Proposed Target \\
\bottomrule
\end{tabular}
\vspace{1mm}
\begin{flushleft}
\footnotesize \textit{Note:} $^*p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$, ns: not significant. Paired Wilcoxon signed-rank tests conducted against the proposed FEE-Net architecture.
\end{flushleft}
\end{table*}
```
