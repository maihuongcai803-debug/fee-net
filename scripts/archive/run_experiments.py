import os
import sys
import subprocess
import time

models = [
    ("unet", "Exp_001: Baseline U-Net"),
    ("attention_unet", "Exp_002: Baseline Attention U-Net"),
    ("fee_net", "Exp_003: Proposed FEE-Net (Full)"),
    ("fee_net_no_dwt", "Exp_004a: Ablation (w/o DWT Frequency Branch)"),
    ("fee_net_no_egm", "Exp_004b: Ablation (w/o Edge-Gated Module)"),
    ("fee_net_no_bl", "Exp_004c: Ablation (w/o Boundary-Aware Loss)")
]

epochs = 15
batch_size = 16

print("="*75)
print(" STARTING COMPLETE BENCHMARK & ABLATION EXPERIMENTAL CAMPAIGN")
print(f" Target Hardware: GPU with PyTorch AMP | Total Models: {len(models)}")
print("="*75)

overall_start = time.time()

for model_name, desc in models:
    print(f"\n>>> Running {desc}...", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable,
        "train.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", "0.0003"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode == 0:
        print(f"  [SUCCESS] {desc} finished in {dt:.1f}s.", flush=True)
    else:
        print(f"  [ERROR] {desc} failed:\n{res.stderr}", flush=True)


print("\n" + "="*75)
print(" RUNNING FINAL BENCHMARK EVALUATION ACROSS ALL TRAINED MODELS")
print("="*75)

eval_cmd = [sys.executable, "evaluate.py", "--batch_size", str(batch_size)]
eval_res = subprocess.run(eval_cmd, capture_output=False, text=True)

total_elapsed = time.time() - overall_start
print(f"\nAll experiments completed successfully in {total_elapsed/60:.2f} minutes!")
