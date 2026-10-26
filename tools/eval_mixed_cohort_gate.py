# -*- coding: utf-8 -*-
"""
Conformal-certified screening cascade evaluated on a LEAKAGE-FREE MIXED cohort.

This is the experiment Kvasir-SEG alone cannot support: because every Kvasir-SEG
frame contains a polyp, it can measure safety (miss rate) but not *specificity*
(how often healthy frames are actually bypassed) and therefore not the real
compute saving. Here we use the mixed cohort built by tools/build_mixed_cohort.py
from held-out test frames only:

    lesion samples : the polyp frame                       (safety denominator)
    normal samples : mucosa crops from the same frame      (specificity / savings)

Protocol (matches the corrected CRC formulation):
    * calibrate tau* on the CALIB half   -> Clopper-Pearson bound on lesions
    * CERTIFY once on the CERTIFY half   -> FNR, normal-bypass rate, savings

Outputs
-------
assets/probe_results/mixed_cohort_screening.json   metrics + operating-point table
assets/mixed_cohort_screening_table.md             paper-ready markdown table
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.fee_net_early_exit import FEENetEarlyExit   # noqa: E402
from scipy import stats                                # noqa: E402

COHORT = os.environ.get("MIXED_COHORT_DIR", os.path.join(ROOT, "data", "kvasir_triage_test"))
OUTDIR = os.environ.get("PROBE_RESULTS_DIR", os.path.join(ROOT, "assets", "probe_results"))


def load_split(sub):
    man = json.load(open(os.path.join(COHORT, "manifest.json"), encoding="utf-8"))
    rows = [s for s in man["samples"] if s["split"] == sub]
    imgs, kinds = [], []
    for r in rows:
        p = os.path.join(COHORT, sub, "images", r["file"])
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR)
        imgs.append(np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.0)
        kinds.append(r["kind"])
    return np.stack(imgs), kinds


def cp_upper(k, n, conf=0.05):
    if n == 0 or k >= n:
        return 1.0
    return float(stats.beta.ppf(1 - conf, k + 1, n - k))


class Gate(torch.nn.Module):
    """stem + layer1 + Haar DWT + SCA + triage head (the screening path)."""

    def __init__(self, ee):
        super().__init__()
        self.stem, self.layer1, self.dwt = ee.stem, ee.layer1, ee.dwt
        self.wf1, self.attn = ee.wf1, getattr(ee, "wf1_attn", None)
        self.head = ee.triage_head

    def forward(self, x):
        xs = self.stem(x)
        e1 = self.layer1(xs)
        _, lh, hl, hh = self.dwt(xs)
        cat = torch.cat([lh, hl, hh], 1)
        if self.attn is not None:
            cat = self.attn(cat)
        return self.head(e1, self.wf1(cat))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "scratch", "kaggle_v3_results",
                                                   "best_feenetearlyexit.pth"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target-fnr", type=float, default=0.05)
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        raise SystemExit(f"[gate] checkpoint not found: {a.ckpt}")

    ee = FEENetEarlyExit(3, 1, base_channels=48)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    missing, unexpected = ee.load_state_dict(sd, strict=False)
    print(f"[gate] loaded {os.path.basename(a.ckpt)} | missing={len(missing)} unexpected={len(unexpected)}")
    ee.eval().to(a.device)

    gate = Gate(ee).eval().to(a.device)
    full = ee.eval()

    data = {}
    for sub in ("calib", "certify"):
        x, kinds = load_split(sub)
        data[sub] = (torch.from_numpy(x).to(a.device), kinds)
        n_l = sum(1 for k in kinds if k == "lesion")
        print(f"[gate] {sub:8s}: {len(kinds)} samples ({n_l} lesion / {len(kinds)-n_l} normal)")

    # --- risk scores (bypassed when risk < tau) ---
    risks = {}
    with torch.no_grad():
        for sub, (x, kinds) in data.items():
            out = []
            for i in range(0, len(x), 8):
                # return_edge=True -> risk at index 2 (repo API)
                out.append(full(x[i:i + 8], tau=0.5, return_edge=True)[2].float().cpu().numpy())
            risks[sub] = np.concatenate(out)

    # --- calibrate tau* on CALIB lesions only ---
    cal = data["calib"][1]
    cal_les = np.array([r for r, k in zip(risks["calib"], cal) if k == "lesion"])
    tau, bound = 0.0, 1.0
    for t in [i / 100.0 for i in range(1, 100)]:
        k = int((cal_les < t).sum())
        u = cp_upper(k, len(cal_les))
        if u <= a.target_fnr:
            tau, bound = float(t), u
        else:
            break
    print(f"[gate] calibrated tau* = {tau:.4f} (CP upper on calib lesions = {bound*100:.2f}%)")

    # --- certify once on the CERTIFY half ---
    cer = data["certify"][1]
    r_cer = risks["certify"]
    les_idx = [i for i, k in enumerate(cer) if k == "lesion"]
    nor_idx = [i for i, k in enumerate(cer) if k == "normal"]
    les_miss = int((r_cer[les_idx] < tau).sum())
    nor_bypass = int((r_cer[nor_idx] < tau).sum())
    fnr = les_miss / max(len(les_idx), 1) * 100.0
    bpr = nor_bypass / max(len(nor_idx), 1) * 100.0
    upper = cp_upper(les_miss, len(les_idx))
    bypass_frac = (les_miss + nor_bypass) / max(len(cer), 1)

    # --- measured latencies (real savings) ---
    xb = data["certify"][0][:1]
    with torch.no_grad():
        for _ in range(20):
            gate(xb); full(xb, tau=0.5, return_edge=True)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            gate(xb)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_gate = (time.time() - t0) / 100 * 1000
        t0 = time.time()
        for _ in range(20):
            full(xb, tau=0.5, return_edge=True)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t_full = (time.time() - t0) / 20 * 1000
    eff_ms = bypass_frac * t_gate + (1 - bypass_frac) * t_full
    speedup = t_full / max(eff_ms, 1e-6)

    # --- operating-point table ---
    table = []
    for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, tau]:
        b = r_cer < t
        lm = int(b[les_idx].sum()); nb = int(b[nor_idx].sum())
        frac = (lm + nb) / max(len(cer), 1)
        table.append({
            "tau": round(float(t), 2),
            "fnr_pct": round(lm / max(len(les_idx), 1) * 100, 2),
            "normal_bypass_pct": round(nb / max(len(nor_idx), 1) * 100, 2),
            "cp_upper_fnr_pct": round(cp_upper(lm, len(les_idx)) * 100, 2),
            "bypassed_frac": round(frac, 4),
            "eff_latency_ms": round(frac * t_gate + (1 - frac) * t_full, 2),
            "speedup_x": round(t_full / max(frac * t_gate + (1 - frac) * t_full, 1e-6), 2),
        })

    try:
        ckpt_ref = os.path.relpath(a.ckpt, ROOT)
    except ValueError:          # different drive on Windows
        ckpt_ref = os.path.abspath(a.ckpt)
    res = {
        "checkpoint": ckpt_ref,
        "device": a.device,
        "cohort": {"calib": data["calib"][1].count("lesion") and int(sum(1 for k in data["calib"][1] if k == "lesion")),
                   "calib_normals": int(sum(1 for k in data["calib"][1] if k == "normal")),
                   "certify_lesions": len(les_idx), "certify_normals": len(nor_idx)},
        "tau_star": round(tau, 4),
        "calib_cp_upper_pct": round(bound * 100, 2),
        "certify_fnr_pct": round(fnr, 2),
        "certify_cp_upper_fnr_pct": round(upper * 100, 2),
        "guarantee_holds": bool(upper <= a.target_fnr),
        "normal_bypass_pct": round(bpr, 2),
        "bypassed_fraction": round(bypass_frac, 4),
        "gate_latency_ms": round(t_gate, 3),
        "full_latency_ms": round(t_full, 3),
        "effective_latency_ms": round(eff_ms, 3),
        "measured_speedup_x": round(speedup, 2),
        "operating_points": table,
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "mixed_cohort_screening.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    # per-sample risk scores for audit / re-calibration
    with open(os.path.join(OUTDIR, "mixed_cohort_risk_scores.csv"), "w", encoding="utf-8") as f:
        f.write("split,kind,risk_score,bypassed\n")
        for sub in ("calib", "certify"):
            for r, k in zip(risks[sub], data[sub][1]):
                f.write(f"{sub},{k},{float(r):.6f},{int(r < tau)}\n")

    md = ["# Mixed-cohort conformal screening (leakage-free)", "",
          f"- checkpoint: `{res['checkpoint']}`",
          f"- cohort: calib {res['cohort']['calib']} lesion + {res['cohort']['calib_normals']} normal | "
          f"certify {res['cohort']['certify_lesions']} lesion + {res['cohort']['certify_normals']} normal",
          f"- calibrated tau* = **{res['tau_star']}** (CP upper on calib = {res['calib_cp_upper_pct']}%)",
          f"- certify FNR = **{res['certify_fnr_pct']}%** (CP upper {res['certify_cp_upper_fnr_pct']}%) -> guarantee holds: **{res['guarantee_holds']}**",
          f"- normal bypass rate = **{res['normal_bypass_pct']}%**; overall bypassed = {res['bypassed_fraction']*100:.1f}%",
          f"- measured latency: gate {res['gate_latency_ms']} ms, full {res['full_latency_ms']} ms "
          f"-> effective {res['effective_latency_ms']} ms ({res['measured_speedup_x']}x)", "",
          "| tau | FNR (%) | Normal bypass (%) | CP upper FNR (%) | Bypassed frac | Eff. latency (ms) | Speedup (x) |",
          "|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['tau']:.2f} | {r['fnr_pct']} | {r['normal_bypass_pct']} | {r['cp_upper_fnr_pct']} | "
                  f"{r['bypassed_frac']} | {r['eff_latency_ms']} | {r['speedup_x']} |")
    with open(os.path.join(ROOT, "assets", "mixed_cohort_screening_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
