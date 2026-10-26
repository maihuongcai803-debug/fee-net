# -*- coding: utf-8 -*-
"""
Ingest a Kaggle result bundle (v3 / v4 / v5 ...) with validation.

Design rules (2026-10-03, agreed with Agent A in AGENT_COMMUNICATION.md):
  * NEVER writes into 05_Paper_Draft/ (locked by the other agent). Figures are
    staged under scratch/figures_for_paper/ for a human/agent to move.
  * --dest makes the output location explicit; both a .zip and an already
    unpacked directory are accepted (the v4 watcher unpacks for us).
  * Every metric row is validated against a sanity envelope before anything is
    reported, so an invalid row (e.g. the v3 double-sigmoid FEENetEarlyExit:
    Dice 25.2 with Recall 100) can never leak into a paper table again.

Usage
-----
    python tools/ingest_kaggle_results.py --bundle scratch/foo.zip --dest scratch/kaggle_v4_results
    python tools/ingest_kaggle_results.py --bundle scratch/kaggle_output_v4 --dest scratch/kaggle_v4_results
"""
import argparse
import csv
import glob
import json
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_DIR = os.path.join(ROOT, "scratch", "figures_for_paper")   # NOT the paper folder

# Sanity envelopes for 150-case polyp segmentation (Kvasir-SEG, 256x256)
BOUNDS = {
    "Test Dice (%)": (40.0, 99.0),
    "Test mIoU (%)": (30.0, 99.0),
    "Precision (%)": (40.0, 100.0),
    "Recall (%)": (40.0, 100.0),
    "HD95 (px)": (1.0, 200.0),
    "Params (M)": (1.0, 200.0),
    "FLOPs (G)": (0.1, 500.0),
    "FPS": (1.0, 1000.0),
    "Latency (ms)": (0.1, 1000.0),
}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _table(headers, rows):
    w = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]
    return "\n".join(["  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)),
                      "  ".join("-" * x for x in w),
                      "\n".join("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) for r in rows)])


def locate(bundle):
    if os.path.isdir(bundle):
        return None, bundle                     # already unpacked
    zp = os.path.abspath(bundle)
    if not os.path.exists(zp):
        cands = sorted(glob.glob(os.path.join(ROOT, "scratch", "*.zip")),
                       key=os.path.getmtime, reverse=True)
        if not cands:
            raise SystemExit("[ingest] no bundle found; pass --bundle")
        zp = cands[0]
    return zp, None


def unpack(bundle, dest):
    zp, unpacked = locate(bundle)
    os.makedirs(dest, exist_ok=True)
    if unpacked:
        for item in os.listdir(unpacked):
            s, d = os.path.join(unpacked, item), os.path.join(dest, item)
            (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
        print(f"[ingest] copied unpacked bundle {unpacked} -> {dest}")
        return
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    print(f"[ingest] extracted {len(z.namelist())} entries from {os.path.basename(zp)} -> {dest}")


def validate(headers, rows):
    """Split rows into (accepted, problems) using the sanity envelope."""
    col = {h: i for i, h in enumerate(headers)}
    problems, ok = [], []
    for r in rows:
        name = r[0]
        bad = []
        for metric, (lo, hi) in BOUNDS.items():
            if metric not in col:
                continue
            try:
                v = float(r[col[metric]])
            except (ValueError, IndexError):
                bad.append(f"{metric}=unparsable({r[col[metric]]!r})")
                continue
            if v != v:
                bad.append(f"{metric}=NaN")
            elif not (lo <= v <= hi):
                bad.append(f"{metric}={v} outside [{lo},{hi}]")
        if "Test Dice (%)" in col and "Test mIoU (%)" in col:
            try:
                d, m = float(r[col["Test Dice (%)"]]), float(r[col["Test mIoU (%)"]])
                if d - m > 12:
                    bad.append(f"Dice-mIoU gap {d-m:.1f} implausible")
            except ValueError:
                pass
        (problems.append((name, bad)) if bad else ok.append(r))
    return ok, problems


def report(headers, rows):
    print(f"\n=== benchmark_metrics_independent_test.csv ({len(rows)} rows) ===")
    print(_table(headers, rows))
    ok, problems = validate(headers, rows)
    if problems:
        print("\n!!! VALIDATION FAILURES -- these rows must NOT reach any paper table:")
        for name, bad in problems:
            print(f"  - {name}: {'; '.join(bad)}")
    else:
        print("\n[validate] all rows inside the sanity envelope")
    return ok, problems


def stage_figures(dest):
    os.makedirs(STAGE_DIR, exist_ok=True)
    staged = []
    for name in ["figure_qualitative_comparison.png", "figure_pareto_frontier.png",
                 "figure_mixed_cohort_tradeoff.png", "idle_latency_benchmark.json",
                 "model_profiles.json", "triage_gate_profile.json", "mixed_cohort_screening.json"]:
        hits = glob.glob(os.path.join(dest, "**", name), recursive=True)
        if hits:
            shutil.copy2(hits[0], os.path.join(STAGE_DIR, name))
            staged.append(name)
        else:
            print(f"[ingest] (absent) {name}")
    print(f"[ingest] staged for the paper -> {os.path.relpath(STAGE_DIR, ROOT)}: {staged}")
    return staged


def recompute_crc(dest):
    hits = glob.glob(os.path.join(dest, "**", "triage_risk_scores.csv"), recursive=True)
    if not hits:
        print("[ingest] no triage_risk_scores.csv -> CRC not recomputed "
              "(keep the run's own calib/test-separated numbers)")
        return None
    headers, rows = _read_csv(hits[0])
    if "split" not in headers:
        print("[ingest] risk file has no 'split' column; cannot separate calibration/test")
        return None
    from scipy import stats
    c = {h: i for i, h in enumerate(headers)}

    def cp_upper(k, n, conf=0.05):
        return 1.0 if (n == 0 or k >= n) else float(stats.beta.ppf(1 - conf, k + 1, n - k))

    val = [float(r[c["risk_score"]]) for r in rows
           if r[c["split"]] == "val" and float(r[c["label"]]) > 0.5]
    test = [float(r[c["risk_score"]]) for r in rows
            if r[c["split"]] == "test" and float(r[c["label"]]) > 0.5]
    tau, bound = 0.0, 1.0
    for t in [i / 100.0 for i in range(1, 100)]:
        k = sum(1 for v in val if v < t)
        u = cp_upper(k, len(val))
        if u <= 0.05:
            tau, bound = float(t), u
        else:
            break
    misses = sum(1 for v in test if v < tau)
    res = {"tau_star": round(tau, 4), "n_val_positive": len(val),
           "cp_upper_on_val_pct": round(bound * 100, 2),
           "n_test_positive": len(test), "test_misses": misses,
           "test_fnr_pct": round(misses / max(len(test), 1) * 100, 2),
           "cp_upper_on_test_pct": round(cp_upper(misses, len(test)) * 100, 2),
           "guarantee_holds": bool(cp_upper(misses, len(test)) <= 0.05)}
    print("\n=== CRC recomputed (calibrate on val -> certify on test) ===")
    for k, v in res.items():
        print(f"  {k:24s} {v}")
    with open(os.path.join(dest, "crc_recomputed.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None, help="zip or unpacked directory")
    ap.add_argument("--dest", default=os.path.join(ROOT, "scratch", "kaggle_ingest"),
                    help="output directory (never the paper folder)")
    a = ap.parse_args()
    bundle = a.bundle or sorted(glob.glob(os.path.join(ROOT, "scratch", "*.zip")),
                                key=os.path.getmtime, reverse=True)[0]
    print(f"[ingest] bundle={bundle}")
    unpack(bundle, a.dest)

    hits = glob.glob(os.path.join(a.dest, "**", "benchmark_metrics_independent_test.csv"), recursive=True)
    ok, problems = ([], [])
    if hits:
        headers, rows = _read_csv(hits[0])
        ok, problems = report(headers, rows)
    else:
        print("[ingest] benchmark CSV missing")
    stage_figures(a.dest)
    recompute_crc(a.dest)

    print("\n[ingest] SUMMARY")
    print(f"  valid rows   : {len(ok)}")
    print(f"  rejected rows: {len(problems)}")
    print(f"  dest         : {os.path.relpath(a.dest, ROOT)}")
    print("  NOTE: 05_Paper_Draft/ untouched; figures staged in scratch/figures_for_paper/")


if __name__ == "__main__":
    main()
