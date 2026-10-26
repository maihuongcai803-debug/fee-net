# Mixed-cohort conformal screening (leakage-free)

- checkpoint: `scratch\triage_stage2e.pth`
- cohort: calib 75 lesion + 134 normal | certify 75 lesion + 133 normal
- calibrated tau* = **0.59** (CP upper on calib = 3.92%)
- certify FNR = **0.0%** (CP upper 3.92%) -> guarantee holds: **True**
- normal bypass rate = **99.25%**; overall bypassed = 63.5%
- measured latency: gate 2.837 ms, full 11.295 ms -> effective 5.928 ms (1.91x)

| tau | FNR (%) | Normal bypass (%) | CP upper FNR (%) | Bypassed frac | Eff. latency (ms) | Speedup (x) |
|---|---|---|---|---|---|---|
| 0.05 | 0.0 | 98.5 | 3.92 | 0.6298 | 5.97 | 1.89 |
| 0.10 | 0.0 | 98.5 | 3.92 | 0.6298 | 5.97 | 1.89 |
| 0.20 | 0.0 | 98.5 | 3.92 | 0.6298 | 5.97 | 1.89 |
| 0.30 | 0.0 | 98.5 | 3.92 | 0.6298 | 5.97 | 1.89 |
| 0.40 | 0.0 | 99.25 | 3.92 | 0.6346 | 5.93 | 1.91 |
| 0.50 | 0.0 | 99.25 | 3.92 | 0.6346 | 5.93 | 1.91 |
| 0.60 | 0.0 | 99.25 | 3.92 | 0.6346 | 5.93 | 1.91 |
| 0.70 | 0.0 | 100.0 | 3.92 | 0.6394 | 5.89 | 1.92 |
| 0.80 | 0.0 | 100.0 | 3.92 | 0.6394 | 5.89 | 1.92 |
| 0.90 | 0.0 | 100.0 | 3.92 | 0.6394 | 5.89 | 1.92 |
| 0.95 | 0.0 | 100.0 | 3.92 | 0.6394 | 5.89 | 1.92 |
| 0.59 | 0.0 | 99.25 | 3.92 | 0.6346 | 5.93 | 1.91 |
