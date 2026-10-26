# Comprehensive Practical Clinical Verification & Benchmark Results (Exp-026)

## Multi-Paradigm & Baseline Performance Comparison Table

| Architecture / Paradigm | Latency (ms) | Video FPS | Mean GFLOPs | GPU Compute Saved | Lesion Sensitivity (Wilson 95% CI) | Miss Count | Normal Bypass Rate | Cohort Dice (Mean±Std) | Speedup Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monolithic (S2)** | 7.88 | **126.8** | 11.33 | 0.0% | 100.0% [67.6%-100.0%] | 0 missed | 0.0% | 0.5560±0.316 | **1.00x** |
| **Binary Triage** | 1.32 | **755.0** | 0.02 | 99.8% | 0.0% [0.0%-32.4%] | 8 missed | 100.0% | 0.0000±0.000 | **5.95x** |
| **3-Tier Cascade** | 7.56 | **132.2** | 8.14 | 28.1% | 100.0% [67.6%-100.0%] | 0 missed | 0.0% | 0.5560±0.316 | **1.04x** |
| **FEENet-EE (Unified)** | 8.13 | **123.0** | 11.78 | 0.0% | 100.0% [67.6%-100.0%] | 0 missed | 5.0% | 0.3553±0.293 | **0.97x** |
| **U-Net (Baseline)** | 3.92 | **255.0** | 13.75 | 0.0% | 100.0% [67.6%-100.0%] | 0 missed | 0.0% | 0.3210±0.279 | **2.01x** |
| **Attention U-Net** | 4.53 | **220.7** | 14.28 | 0.0% | 100.0% [67.6%-100.0%] | 0 missed | 0.0% | 0.3932±0.238 | **1.74x** |
| **UNet++ (Nested SOTA)** | 14.98 | **66.7** | 34.62 | 0.0% | 100.0% [67.6%-100.0%] | 0 missed | 0.0% | 0.4024±0.276 | **0.53x** |

*Note: High-Risk Lesion Sensitivity is reported with exact 95% Wilson Score Intervals.*