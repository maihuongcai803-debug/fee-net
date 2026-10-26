# -*- coding: utf-8 -*-
"""
Probe matrix driver (v2 -- reordered after the 2026-10-03 interim finding).

Interim result that drove the reordering:
    FEENet val Dice  35 ep = 76.14 %   |   60 ep (to ep52) = 76.6 %
=> under-training is NOT the main cause (curve flattens after ~ep31), so the
   decisive question is which auxiliary loss term is competing with segmentation.

Queue order therefore puts the "pure BCE+Dice" ablation first (it isolates the
whole auxiliary-loss stack in one run), then the UNet reference, then one
factor at a time. The lr=1e-3 variant was dropped to shorten the loop.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PROBE = os.path.join(ROOT, "tools", "local_probe.py")
LOG = os.path.join(ROOT, "assets", "probe_results", "matrix2.log")

MATRIX = [
    # decisive: remove the entire auxiliary stack -> pure BCE + Dice
    ("q1_feenet_e60_pure", ["--model", "feenet", "--epochs", "60",
                           "--no-deep", "--no-edge", "--no-boundary"]),
    # reference: how strong is a plain U-Net under the identical protocol?
    ("q1_unet_e60",       ["--model", "unet", "--epochs", "60"]),
    # one factor at a time
    ("q2_feenet_e60_nodeep",  ["--model", "feenet", "--epochs", "60", "--no-deep"]),
    ("q2_feenet_e60_noedge",  ["--model", "feenet", "--epochs", "60", "--no-edge"]),
    ("q2_feenet_e60_nobound", ["--model", "feenet", "--epochs", "60", "--no-boundary"]),
]


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as lf:
        lf.write(f"\n#### matrix v2 started {time.strftime('%H:%M:%S')} ####\n")
    for tag, args in MATRIX:
        cmd = [PY, PROBE, "--tag", tag] + args
        t0 = time.time()
        with open(LOG, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {tag} :: {' '.join(args)} ({time.strftime('%H:%M:%S')}) ===\n")
            lf.flush()
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               cwd=ROOT, timeout=3 * 3600)
            lf.write(f"--- {tag} rc={r.returncode} ({(time.time()-t0)/60:.1f} min)\n")
            lf.flush()
    with open(LOG, "a", encoding="utf-8") as lf:
        lf.write(f"\nALL DONE {time.strftime('%H:%M:%S')}\n")


if __name__ == "__main__":
    main()
