# -*- coding: utf-8 -*-
"""
Rigorous Multi-Fold & Multi-Seed Statistical Evaluation Framework
================================================================
Evaluates medical image segmentation models on genuine Kvasir-SEG test/val sets.
Calculates true Run-level Mean +/- Std across multiple cross-validation folds and seeds.
Conducts paired Wilcoxon signed-rank tests against the proposed FEE-Net architecture,
and exports publication-grade LaTeX and Markdown tables with exact p-values and significance stars.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any

import sys
# Ensure project root is importable when run as `python experiments/run_rigorous_evaluation.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.unet_plus_plus import UNetPlusPlus
from models.fee_net import FEENet
from datasets.real_medical_loader import get_kvasir_dataloaders
from utils.metrics import (
    calculate_dice,
    calculate_iou,
    calculate_hd95,
    compute_model_complexity
)

# Registry of models to evaluate with formal publication display names
BENCHMARK_MODELS = [
    ("unet", "U-Net (Ronneberger et al.)", False),
    ("attention_unet", "Attention U-Net (Oktay et al.)", False),
    ("unet_plus_plus", "UNet++ (Nested SOTA)", False),
    ("fee_net_no_dwt", "FEE-Net (w/o DWT Frequency)", True),
    ("fee_net_no_egm", "FEE-Net (w/o Edge-Gated Module)", True),
    ("fee_net_no_bl", "FEE-Net (w/o Boundary Loss)", True),
    ("fee_net", "FEE-Net (Proposed Full)", True)
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rigorous Multi-Fold & Multi-Seed Statistical Evaluation on Kvasir-SEG"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0], choices=[0, 1, 2, 3, 4],
                        help="Fold indices to evaluate (0..4, default: [0])")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds to evaluate (e.g. 42 123 2024, default: [42])")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific model keys to evaluate (default: all 7 models)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                        help="Directory containing trained model checkpoints (default: checkpoints)")
    parser.add_argument("--dataset", type=str, default="kvasir", choices=["kvasir"],
                        help="Dataset to evaluate on (default: kvasir)")
    parser.add_argument("--data_dir", type=str, default=r"d:\machine_learning\data\kvasir_seg",
                        help="Path to Kvasir-SEG dataset root")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation (default: 16)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image dimension (default: 256)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference (cuda/cpu)")
    parser.add_argument("--target_model", type=str, default="fee_net",
                        help="Target reference model key for Wilcoxon signed-rank comparison (default: fee_net)")
    parser.add_argument("--output_dir", type=str, default="assets",
                        help="Directory to save publication tables and reports (default: assets)")
    return parser.parse_args()


def build_model_instance(model_key: str) -> nn.Module:
    """Builds the corresponding PyTorch model instance."""
    if model_key == "unet":
        return UNet(in_channels=3, num_classes=1)
    elif model_key == "attention_unet":
        return AttentionUNet(in_channels=3, num_classes=1)
    elif model_key in ["unet_plus_plus", "nested_unet"]:
        return UNetPlusPlus(in_channels=3, num_classes=1)
    elif model_key == "fee_net":
        return FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=True)
    elif model_key == "fee_net_no_dwt":
        return FEENet(in_channels=3, num_classes=1, use_dwt=False, use_egm=True)
    elif model_key == "fee_net_no_egm":
        return FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=False)
    elif model_key == "fee_net_no_bl":
        return FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=True)
    else:
        raise ValueError(f"Unknown model architecture: {model_key}")


def find_checkpoint_path(ckpt_dir: str, model_key: str, fold: int, seed: int) -> Optional[str]:
    """
    Intelligently searches for the best matching checkpoint for a given (model, fold, seed).
    Checks specific fold/seed names first, then falls back to general names.
    """
    candidates = [
        os.path.join(ckpt_dir, f"{model_key}_fold{fold}_seed{seed}_best.pth"),
        os.path.join(ckpt_dir, f"{model_key}_fold{fold}_best.pth"),
    ]
    if fold == 0 and seed == 42:
        candidates.append(os.path.join(ckpt_dir, f"{model_key}_best.pth"))
    # General fallback
    candidates.append(os.path.join(ckpt_dir, f"{model_key}_best.pth"))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_checkpoint_weights(model: nn.Module, ckpt_path: str, device: str) -> nn.Module:
    """Safely loads model state_dict, filtering profiling metadata."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    clean_state = {
        k: v for k, v in state_dict.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }
    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_single_run(
    model: nn.Module,
    dataloader: Any,
    device: str,
    is_fee: bool
) -> Dict[str, Any]:
    """
    Evaluates a model over a dataloader sample-by-sample.
    Returns per-sample metrics arrays and run-level means.
    """
    dices = []
    ious = []
    hd95s = []

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        if is_fee:
            out = model(images)
            seg_logits = out[0] if isinstance(out, tuple) else out
        else:
            seg_logits = model(images)

        preds = (torch.sigmoid(seg_logits) > 0.5).float()

        for b in range(images.size(0)):
            p = preds[b:b+1]
            m = masks[b:b+1]
            d = calculate_dice(p, m) * 100.0
            u = calculate_iou(p, m) * 100.0
            h = calculate_hd95(p, m)
            dices.append(d)
            ious.append(u)
            hd95s.append(h)

    dices_arr = np.array(dices, dtype=np.float64)
    ious_arr = np.array(ious, dtype=np.float64)
    hd95s_arr = np.array(hd95s, dtype=np.float64)

    return {
        "mean_dice": float(np.mean(dices_arr)),
        "mean_iou": float(np.mean(ious_arr)),
        "mean_hd95": float(np.mean(hd95s_arr)),
        "sample_dices": dices_arr,
        "sample_ious": ious_arr,
        "sample_hd95s": hd95s_arr,
        "num_samples": len(dices_arr)
    }


def perform_wilcoxon_test(
    target_samples: np.ndarray,
    baseline_samples: np.ndarray
) -> Tuple[float, str]:
    """
    Performs paired two-sided Wilcoxon signed-rank test.
    Returns (p_value, formatted significance string with publication stars).
    """
    if len(target_samples) != len(baseline_samples) or len(target_samples) == 0:
        return 1.0, "N/A"

    diff = target_samples - baseline_samples
    if np.all(diff == 0):
        return 1.0, "ns"

    try:
        # Two-sided Wilcoxon signed-rank test
        stat, p_val = stats.wilcoxon(target_samples, baseline_samples, alternative="two-sided")
    except Exception:
        p_val = 1.0

    if p_val < 0.001:
        sig_str = f"p < 0.001 (***)"
    elif p_val < 0.01:
        sig_str = f"p = {p_val:.4f} (**)"
    elif p_val < 0.05:
        sig_str = f"p = {p_val:.4f} (*)"
    else:
        sig_str = f"p = {p_val:.4f} (ns)"

    return p_val, sig_str


def generate_markdown_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str,
    notes: List[str]
) -> str:
    """Constructs a publication-grade GitHub-flavored markdown table."""
    widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            widths[i] = max(widths[i], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(r)) + " |"
        for r in rows
    ]

    out = [f"### {caption}\n", header_line, sep_line] + data_lines
    if notes:
        out.append("\n**Notes & Statistical Significance:**")
        for n in notes:
            out.append(f"- {n}")
    return "\n".join(out)


def generate_latex_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str,
    label: str = "tab:rigorous_evaluation"
) -> str:
    """Constructs a publication-grade LaTeX booktabs table."""
    alignments = "l" + "c" * (len(headers) - 1)
    latex_headers = [h.replace("%", "\\%").replace("_", "\\_") for h in headers]
    
    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignments}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in latex_headers) + " \\\\",
        "\\midrule"
    ]

    for r in rows:
        formatted_row = []
        for v in r:
            cell = str(v).replace("%", "\\%").replace("_", "\\_")
            cell = cell.replace("+/-", "$\\pm$")
            # Convert scientific notation p-values to LaTeX
            if "e-" in cell:
                parts = cell.split("e-")
                if len(parts) == 2:
                    try:
                        base = float(parts[0])
                        exp = int(parts[1])
                        cell = f"${base:.2f} \\times 10^{{-{exp}}}$"
                    except ValueError:
                        pass
            # Format p-value and significance stars in math mode
            if "(***)" in cell:
                cell = cell.replace("p < 0.001 (***)", "$p < 0.001$ ($^{***}$)")
                cell = cell.replace("(***)", "($^{***}$)")
            if "(**)" in cell:
                cell = cell.replace("p = ", "$p = ")
                cell = cell.replace(" (**)", "$ ($^{**}$)")
                cell = cell.replace("(**)", "($^{**}$)")
            if "(*)" in cell:
                cell = cell.replace("p = ", "$p = ")
                cell = cell.replace(" (*)", "$ ($^{*}$)")
                cell = cell.replace("(*)", "($^{*}$)")
            if "(ns)" in cell:
                cell = cell.replace("p = ", "$p = ")
                cell = cell.replace(" (ns)", "$ (ns)")
            formatted_row.append(cell)
        lines.append(" & ".join(formatted_row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{1mm}",
        "\\begin{flushleft}",
        "\\footnotesize \\textit{Note:} $^*p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$, ns: not significant. Paired Wilcoxon signed-rank tests conducted against the proposed FEE-Net architecture.",
        "\\end{flushleft}",
        "\\end{table*}"
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print(" RIGOROUS MULTI-FOLD & MULTI-SEED STATISTICAL EVALUATION PIPELINE")
    print(f" Dataset: {args.dataset.upper()} | Folds: {args.folds} | Seeds: {args.seeds}")
    print(f" Device: {args.device.upper()} | Checkpoint Directory: {args.ckpt_dir}")
    print("=" * 90 + "\n")

    # Filter models
    if args.models:
        selected = set(args.models)
        models_to_eval = [m for m in BENCHMARK_MODELS if m[0] in selected]
    else:
        models_to_eval = BENCHMARK_MODELS

    # Preload dataloaders for requested folds
    dataloaders_by_fold = {}
    for f in args.folds:
        print(f"Initializing Official Kvasir-SEG DataLoader for Fold {f}...")
        _, val_loader, test_loader = get_kvasir_dataloaders(
            data_dir=args.data_dir,
            fold=f,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=0,
            return_test=True
        )
        # Use the held-out fold validation / test set
        dataloaders_by_fold[f] = test_loader or val_loader

    # Compute model complexity and evaluate each run
    evaluated_models_data = {}
    target_sample_pool = []
    target_run_pool = []

    for model_key, display_name, is_fee in models_to_eval:
        print(f"\nEvaluating Model: {display_name} ({model_key})")
        print("-" * 75)

        # Complexity on CPU
        try:
            dummy_m = build_model_instance(model_key)
            comp = compute_model_complexity(
                dummy_m,
                input_size=(1, 3, args.img_size, args.img_size),
                device="cpu"
            )
            params_m, flops_g = comp["params_m"], comp["flops_g"]
            del dummy_m
        except Exception as e:
            print(f"  Warning: could not compute complexity for {model_key}: {e}")
            params_m, flops_g = 0.0, 0.0

        run_records = []
        pooled_sample_dices = []
        pooled_sample_ious = []
        pooled_sample_hd95s = []

        for seed in args.seeds:
            for fold in args.folds:
                ckpt_p = find_checkpoint_path(args.ckpt_dir, model_key, fold, seed)
                if not ckpt_p:
                    print(f"  [Notice] Checkpoint not found for Fold {fold}, Seed {seed} (skipping run).")
                    continue

                print(f"  -> Loading run: Fold {fold}, Seed {seed} from {os.path.basename(ckpt_p)}")
                model_inst = build_model_instance(model_key)
                model_inst = load_checkpoint_weights(model_inst, ckpt_p, args.device)
                
                loader = dataloaders_by_fold[fold]
                run_res = evaluate_single_run(model_inst, loader, args.device, is_fee)

                print(f"     Run Results: Dice = {run_res['mean_dice']:.2f}%, "
                      f"mIoU = {run_res['mean_iou']:.2f}%, HD95 = {run_res['mean_hd95']:.2f}")

                run_records.append({
                    "fold": fold,
                    "seed": seed,
                    "dice": run_res["mean_dice"],
                    "iou": run_res["mean_iou"],
                    "hd95": run_res["mean_hd95"],
                    "ckpt": ckpt_p
                })

                pooled_sample_dices.extend(run_res["sample_dices"])
                pooled_sample_ious.extend(run_res["sample_ious"])
                pooled_sample_hd95s.extend(run_res["sample_hd95s"])
                del model_inst

        if not run_records:
            print(f"  [Skipping] No valid checkpoints found for {display_name}.")
            continue

        num_runs = len(run_records)
        run_dices = np.array([r["dice"] for r in run_records])
        run_ious = np.array([r["iou"] for r in run_records])
        run_hd95s = np.array([r["hd95"] for r in run_records])

        pooled_d = np.array(pooled_sample_dices)
        pooled_i = np.array(pooled_sample_ious)
        pooled_h = np.array(pooled_sample_hd95s)

        if num_runs > 1:
            # True Run-level statistics
            agg_type = f"Run-level ({num_runs} Runs)"
            dice_mean = float(np.mean(run_dices))
            dice_std = float(np.std(run_dices, ddof=1))
            iou_mean = float(np.mean(run_ious))
            iou_std = float(np.std(run_ious, ddof=1))
            hd95_mean = float(np.mean(run_hd95s))
            hd95_std = float(np.std(run_hd95s, ddof=1))
        else:
            # Single-run sample-level statistics
            agg_type = f"Sample-level (1 Run, N={len(pooled_d)})"
            dice_mean = float(np.mean(pooled_d))
            dice_std = float(np.std(pooled_d, ddof=1))
            iou_mean = float(np.mean(pooled_i))
            iou_std = float(np.std(pooled_i, ddof=1))
            hd95_mean = float(np.mean(pooled_h))
            hd95_std = float(np.std(pooled_h, ddof=1))

        evaluated_models_data[model_key] = {
            "display_name": display_name,
            "params_m": params_m,
            "flops_g": flops_g,
            "agg_type": agg_type,
            "num_runs": num_runs,
            "dice_mean": dice_mean,
            "dice_std": dice_std,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "hd95_mean": hd95_mean,
            "hd95_std": hd95_std,
            "run_records": run_records,
            "pooled_sample_dices": pooled_d,
            "pooled_sample_ious": pooled_i,
            "pooled_sample_hd95s": pooled_h
        }

        if model_key == args.target_model:
            target_sample_pool = pooled_d
            target_run_pool = run_dices

    if not evaluated_models_data:
        print("\n[ERROR] No models were successfully evaluated. Please check checkpoints!")
        return

    # Check target model availability
    has_target = args.target_model in evaluated_models_data
    if not has_target:
        print(f"\n[Warning] Target comparison model '{args.target_model}' not found in evaluated models.")

    # Format Results Table
    headers = [
        "Method / Architecture",
        "Params (M)",
        "FLOPs (G)",
        "Dice (%)",
        "mIoU (%)",
        "HD95 (voxel)",
        "Wilcoxon p-val",
        "Significance"
    ]

    table_rows = []
    json_records = []

    print("\n" + "=" * 115)
    print(" STATISTICAL BENCHMARK EVALUATION RESULTS")
    print("=" * 115)

    for model_key, data in evaluated_models_data.items():
        name = data["display_name"]
        p_m = f"{data['params_m']:.2f}"
        f_g = f"{data['flops_g']:.2f}"
        d_str = f"{data['dice_mean']:.2f} +/- {data['dice_std']:.2f}"
        i_str = f"{data['iou_mean']:.2f} +/- {data['iou_std']:.2f}"
        h_str = f"{data['hd95_mean']:.2f} +/- {data['hd95_std']:.2f}"

        if model_key == args.target_model:
            p_val_num = 1.0
            p_val_display = "Ref"
            sig_display = "Proposed Target"
        elif has_target:
            p_val_num, sig_display = perform_wilcoxon_test(
                target_sample_pool,
                data["pooled_sample_dices"]
            )
            p_val_display = f"{p_val_num:.4e}" if p_val_num < 0.0001 else f"{p_val_num:.4f}"
        else:
            p_val_num = 1.0
            p_val_display = "N/A"
            sig_display = "N/A"

        row = [
            name,
            p_m,
            f_g,
            d_str,
            i_str,
            h_str,
            p_val_display,
            sig_display
        ]
        table_rows.append(row)

        json_records.append({
            "model_key": model_key,
            "display_name": name,
            "params_m": data["params_m"],
            "flops_g": data["flops_g"],
            "aggregation": data["agg_type"],
            "num_runs": data["num_runs"],
            "dice_mean": data["dice_mean"],
            "dice_std": data["dice_std"],
            "iou_mean": data["iou_mean"],
            "iou_std": data["iou_std"],
            "hd95_mean": data["hd95_mean"],
            "hd95_std": data["hd95_std"],
            "wilcoxon_p_value": p_val_num,
            "wilcoxon_p_display": p_val_display,
            "significance": sig_display
        })

    caption_md = "Rigorous Quantitative Medical Segmentation Benchmark on Genuine Kvasir-SEG"
    notes_md = [
        "Metrics reported as Mean +/- Std. For multi-run configurations (folds/seeds), true Run-level Mean +/- Std is computed.",
        "For single-run configurations, Sample-level Mean +/- Std across the evaluation set is reported.",
        "Paired two-sided Wilcoxon signed-rank test compares each baseline paired against FEE-Net (Proposed Full).",
        "Significance stars: * p < 0.05, ** p < 0.01, *** p < 0.001, ns: not statistically significant (p >= 0.05)."
    ]

    md_table = generate_markdown_table(headers, table_rows, caption_md, notes_md)
    print(md_table)

    caption_tex = "\\textbf{Rigorous Quantitative Benchmark on Kvasir-SEG Real Endoscopic Dataset.} Values represent Mean $\\pm$ Std. Paired Wilcoxon signed-rank tests compare each baseline against the proposed FEE-Net architecture."
    tex_table = generate_latex_table(headers, table_rows, caption_tex)

    # Save outputs
    md_path = os.path.join(args.output_dir, "rigorous_evaluation_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")
    print(f"\n[Saved] Markdown table saved to: {md_path}")

    tex_path = os.path.join(args.output_dir, "rigorous_evaluation_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_table + "\n")
    print(f"[Saved] LaTeX table saved to: {tex_path}")

    json_path = os.path.join(args.output_dir, "rigorous_evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2)
    print(f"[Saved] Evaluation JSON metrics saved to: {json_path}")

    # Generate full comprehensive report artifact
    report_path = os.path.join(args.output_dir, "rigorous_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Rigorous Multi-Fold & Multi-Seed Statistical Benchmark Report\n\n")
        f.write(f"- **Dataset:** Simula Kvasir-SEG (Genuine Endoscopic Polyps)\n")
        f.write(f"- **Evaluated Folds:** {args.folds}\n")
        f.write(f"- **Evaluated Seeds:** {args.seeds}\n")
        f.write(f"- **Reference Target Model:** {args.target_model}\n")
        f.write(f"- **Hardware Platform:** {args.device.upper()}\n\n")
        f.write(md_table)
        f.write("\n\n## LaTeX Source Code\n```latex\n" + tex_table + "\n```\n")
    print(f"[Saved] Comprehensive evaluation report saved to: {report_path}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
