# -*- coding: utf-8 -*-
"""
Rigorous Multi-Fold & Multi-Seed Benchmark & Ablation Campaign Execution Pipeline
================================================================================
Automates sequential training across 7 medical segmentation models:
(U-Net, Attention U-Net, UNet++, FEE-Net Full, and 3 ablation variants)
on genuine Kvasir-SEG across multiple folds (0..4) and seeds (42, 123, 2024),
followed by comprehensive rigorous evaluation and Wilcoxon hypothesis testing.
"""

import os
import sys
import argparse
import subprocess
import time
import json
import torch

BENCHMARK_MODELS = [
    ("unet", "Exp-001b: Baseline U-Net (Ronneberger et al.)"),
    ("attention_unet", "Exp-002: Baseline Attention U-Net (Oktay et al.)"),
    ("unet_plus_plus", "Exp-008: Modern SOTA UNet++ (Nested Skips)"),
    ("fee_net_no_dwt", "Exp-004a: Ablation (w/o DWT Frequency Branch)"),
    ("fee_net_no_egm", "Exp-004b: Ablation (w/o Edge-Gated Module)"),
    ("fee_net_no_bl", "Exp-004c: Ablation (w/o Boundary Loss)"),
    ("fee_net", "Exp-007: Proposed FEE-Net Full (with Gamma Residual Gate)")
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Rigorous Multi-Fold & Multi-Seed Medical Segmentation Benchmark"
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs per run (default: 50)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--folds", type=int, nargs="+", default=[0], choices=[0, 1, 2, 3, 4],
                        help="Folds to train on: 0..4 (default: [0])")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds to train with: e.g. 42 123 2024 (default: [42])")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific model keys to train (default: all 7 benchmark models)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--dataset", type=str, default="kvasir", choices=["kvasir", "synthetic"],
                        help="Dataset to train on (default: kvasir)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda/cpu)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip training if model checkpoint already exists")
    parser.add_argument("--eval_after", action="store_true", default=True,
                        help="Run rigorous statistical evaluation after all training finishes")
    parser.add_argument("--no_eval", dest="eval_after", action="store_false",
                        help="Do not run evaluation after training")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Checkpoints save directory")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    # Filter models to run
    if args.models:
        selected_keys = set(args.models)
        models_to_run = [m for m in BENCHMARK_MODELS if m[0] in selected_keys]
        if not models_to_run:
            print(f"Error: No valid models matched from {args.models}")
            sys.exit(1)
    else:
        models_to_run = BENCHMARK_MODELS

    total_runs = len(models_to_run) * len(args.folds) * len(args.seeds)

    print("=" * 85)
    print(" RIGOROUS MULTI-FOLD & MULTI-SEED BENCHMARK CAMPAIGN")
    print(f" Dataset: {args.dataset.upper()} | Device: {args.device.upper()}")
    print(f" Folds: {args.folds} | Seeds: {args.seeds} | Models: {len(models_to_run)}")
    print(f" Epochs: {args.epochs} (Patience: {args.patience}) | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f" Total Planned Runs: {total_runs}")
    print("=" * 85)

    overall_start = time.time()
    run_idx = 0
    results_summary = []

    for seed in args.seeds:
        for fold in args.folds:
            for model_key, model_desc in models_to_run:
                run_idx += 1
                ckpt_expected = os.path.join(args.save_dir, f"{model_key}_fold{fold}_seed{seed}_best.pth")
                legacy_ckpt = os.path.join(args.save_dir, f"{model_key}_best.pth")
                
                print(f"\n[{run_idx}/{total_runs}] Starting: {model_desc} (Fold {fold}, Seed {seed})")
                print("-" * 75)

                if args.skip_existing and (os.path.exists(ckpt_expected) or (fold == 0 and seed == 42 and os.path.exists(legacy_ckpt))):
                    print(f"  [Skipped] Checkpoint already exists for {model_key} (Fold {fold}, Seed {seed}).")
                    results_summary.append({
                        "model": model_key,
                        "fold": fold,
                        "seed": seed,
                        "status": "SKIPPED_EXISTING",
                        "time_s": 0.0
                    })
                    continue

                t0 = time.time()
                cmd = [
                    sys.executable,
                    "train.py",
                    "--model", model_key,
                    "--dataset", args.dataset,
                    "--fold", str(fold),
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--patience", str(args.patience),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--device", args.device,
                    "--save_dir", args.save_dir
                ]

                res = subprocess.run(cmd)
                dt = time.time() - t0

                if res.returncode == 0:
                    print(f"  [SUCCESS] Finished in {dt:.1f}s ({dt/60:.2f} mins).")
                    status = "SUCCESS"
                else:
                    print(f"  [ERROR] Training failed with code {res.returncode}.")
                    status = f"FAILED_{res.returncode}"

                results_summary.append({
                    "model": model_key,
                    "fold": fold,
                    "seed": seed,
                    "status": status,
                    "time_s": dt
                })

    total_time = time.time() - overall_start
    print("\n" + "=" * 85)
    print(f" BENCHMARK CAMPAIGN FINISHED in {total_time/60:.2f} minutes.")
    print("=" * 85)

    summary_file = os.path.join("assets", "benchmark_run_summary.json")
    with open(summary_file, "w") as fp:
        json.dump(results_summary, fp, indent=2)
    print(f"Run execution summary saved to {summary_file}")

    if args.eval_after:
        print("\n" + "=" * 85)
        print(" EXECUTING RIGOROUS STATISTICAL EVALUATION...")
        print("=" * 85)
        eval_cmd = [
            sys.executable,
            "run_rigorous_evaluation.py",
            "--dataset", args.dataset,
            "--folds", *[str(f) for f in args.folds],
            "--seeds", *[str(s) for s in args.seeds],
            "--ckpt_dir", args.save_dir,
            "--device", args.device,
            "--batch_size", str(args.batch_size)
        ]
        subprocess.run(eval_cmd)

if __name__ == "__main__":
    main()
