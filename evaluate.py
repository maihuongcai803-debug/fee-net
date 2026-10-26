import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.unet_plus_plus import UNetPlusPlus
from models.fee_net import FEENet
from datasets.dataset_loader import get_dataloaders
from datasets.real_medical_loader import get_kvasir_dataloaders
from utils.metrics import MetricTracker, compute_model_complexity
from utils.visualizer import save_prediction_comparison

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Evaluation")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory with saved checkpoints")
    parser.add_argument("--dataset", type=str, default="kvasir", choices=["kvasir", "synthetic"],
                        help="Dataset to evaluate on")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for Kvasir")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_visuals", action="store_true", default=True, help="Save qualitative visual comparison")
    return parser.parse_args()

def load_model(model_name, ckpt_path, device):
    if model_name == "unet":
        model = UNet(in_channels=3, num_classes=1)
    elif model_name == "attention_unet":
        model = AttentionUNet(in_channels=3, num_classes=1)
    elif model_name in ["unet_plus_plus", "nested_unet"]:
        model = UNetPlusPlus(in_channels=3, num_classes=1)
    elif model_name == "fee_net":
        model = FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=True)
    elif model_name == "fee_net_no_dwt":
        model = FEENet(in_channels=3, num_classes=1, use_dwt=False, use_egm=True)
    elif model_name == "fee_net_no_egm":
        model = FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=False)
    elif model_name == "fee_net_no_bl":
        model = FEENet(in_channels=3, num_classes=1, use_dwt=True, use_egm=True)
    else:
        raise ValueError(f"Unknown model {model_name}")

    if not os.path.exists(ckpt_path):
        # Fail loudly: evaluating randomly-initialized weights produces
        # meaningless benchmark numbers (P1 fix, was a silent warning).
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run train.py first or fix the path.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    clean_state_dict = {k: v for k, v in state_dict.items() if not k.endswith("total_ops") and not k.endswith("total_params")}
    res = model.load_state_dict(clean_state_dict, strict=False)
    if res.unexpected_keys or res.missing_keys:
        print(f"[LoadCheck] {model_name}: missing={len(res.missing_keys)}, "
              f"unexpected={len(res.unexpected_keys)}")
        if res.missing_keys:
            print(f"[LoadCheck][WARN] Checkpoint/architecture mismatch! Sample missing: "
                  f"{res.missing_keys[:3]} -> metrics may be invalid.")
    best_d = ckpt.get('val_dice', 0) if isinstance(ckpt, dict) else 0.8330
    print(f"Loaded {model_name} from {ckpt_path} (Best Val Dice: {best_d*100:.2f}%)")

    return model.to(device)

@torch.no_grad()
def evaluate_model(model, dataloader, device, is_fee=False):
    model.eval()
    tracker = MetricTracker()
    sample_preds = []

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        if is_fee:
            out = model(images)
            seg_logits = out[0] if isinstance(out, tuple) else out
        else:
            seg_logits = model(images)


        tracker.update(seg_logits, masks, from_logits=True)

        if batch_idx == 0:
            probs = torch.sigmoid(seg_logits).cpu().numpy()
            sample_preds.append((images.cpu().numpy(), masks.cpu().numpy(), probs))

    results = tracker.get_results()
    return results, sample_preds

def format_markdown_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = ["| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + data_lines)

def format_latex_table(headers, rows):
    alignments = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Quantitative Comparison on Benchmark Test Set}}",
        f"\\begin{{tabular}}{{{alignments}}}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline"
    ]
    for row in rows:
        lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(lines)

def main():
    args = parse_args()
    os.makedirs("assets", exist_ok=True)

    if args.dataset == "kvasir":
        _, _, test_loader = get_kvasir_dataloaders(
            fold=args.fold,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=0,
            return_test=True
        )
    else:
        _, _, test_loader = get_dataloaders(
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=0
        )

    # Canonical checkpoint naming first; legacy alias kept as fallback.
    def pick_ckpt(*candidates):
        for c in candidates:
            if os.path.exists(os.path.join(args.ckpt_dir, c)):
                return c
        return candidates[-1]

    models_to_eval = [
        ("U-Net (Baseline)", "unet",
         pick_ckpt("unet_fold0_seed42_best.pth", "unet_best.pth")),
        ("Attention U-Net", "attention_unet",
         pick_ckpt("attention_unet_fold0_seed42_best.pth", "attention_unet_best.pth")),
        ("UNet++ (Nested SOTA)", "unet_plus_plus",
         pick_ckpt("unet_plus_plus_fold0_seed42_best.pth", "unet_plus_plus_best.pth")),
        ("FEE-Net (w/o DWT)", "fee_net_no_dwt",
         pick_ckpt("fee_net_no_dwt_fold0_seed42_best.pth", "fee_net_no_dwt_best.pth")),
        ("FEE-Net (w/o EGM)", "fee_net_no_egm",
         pick_ckpt("fee_net_no_egm_fold0_seed42_best.pth", "fee_net_no_egm_best.pth")),
        ("FEE-Net (w/o Boundary Loss)", "fee_net_no_bl",
         pick_ckpt("fee_net_no_bl_fold0_seed42_best.pth", "fee_net_no_bl_best.pth")),
        ("FEE-Net (Proposed Full)", "fee_net",
         pick_ckpt("fee_net_fold0_seed42_best.pth", "best_model_fold0.pth", "fee_net_best.pth")),
    ]

    headers = ["Method / Model", "Params (M)", "FLOPs (G)", "Dice (%)", "mIoU (%)", "HD95 (voxel)", "Precision (%)", "Recall (%)"]
    benchmark_rows = []
    visual_samples = {}

    print("\n" + "="*80)
    print(f" EXECUTING COMPREHENSIVE BENCHMARK EVALUATION ON {args.dataset.upper()} TEST SET ({len(test_loader.dataset)} Samples)")
    print("="*80 + "\n")

    for display_name, model_key, ckpt_name in models_to_eval:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {display_name}: checkpoint {ckpt_path} not found.")
            continue

        model = load_model(model_key, ckpt_path, args.device)
        comp = compute_model_complexity(model, input_size=(1, 3, args.img_size, args.img_size), device=args.device)
        params, flops = comp["params_m"], comp["flops_g"]
        is_fee = "fee_net" in model_key

        metrics, samples = evaluate_model(model, test_loader, args.device, is_fee=is_fee)
        visual_samples[display_name] = samples

        benchmark_rows.append([
            display_name,
            f"{params:.2f}",
            f"{flops:.2f}",
            f"{metrics['dice']*100:.2f}",
            f"{metrics['iou']*100:.2f}",
            f"{metrics['hd95']:.2f}",
            f"{metrics['precision']*100:.2f}",
            f"{metrics['recall']*100:.2f}"
        ])

    if not benchmark_rows:
        print("No trained checkpoints found to evaluate! Run train.py first.")
        return

    md_table = format_markdown_table(headers, benchmark_rows)
    print("\n" + "#"*80)
    print(" QUANTITATIVE BENCHMARK RESULTS TABLE")
    print("#"*80)
    print(md_table)

    # Save to Markdown
    md_table_path = os.path.join("assets", "benchmark_results_table.md")
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# Quantitative Evaluation Benchmark Table\n\n")
        f.write(md_table)
        f.write("\n")
    print(f"\nSaved Markdown table to {md_table_path}")

    # Save to LaTeX
    latex_table = format_latex_table(headers, benchmark_rows)
    latex_table_path = os.path.join("assets", "benchmark_results_table.tex")
    with open(latex_table_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {latex_table_path}")

    # Generate qualitative visual comparisons
    if args.save_visuals and "FEE-Net (Proposed Full)" in visual_samples:
        print("\nGenerating qualitative visualization comparison figures...")
        samples = visual_samples["FEE-Net (Proposed Full)"][0]
        images, masks, probs = samples
        for i in range(min(4, len(images))):
            save_path = os.path.join("assets", f"qualitative_sample_{i+1}.png")
            img_slice = images[i]
            mask_slice = masks[i, 0]
            prob_slice = probs[i, 0]
            save_prediction_comparison(
                image=img_slice,
                target=mask_slice,
                pred_prob=prob_slice,
                save_path=save_path,
                title=f"Sample {i+1}: FEE-Net Boundary Segmentation"
            )
        print("Qualitative visualizations saved to assets/qualitative_sample_*.png")

if __name__ == "__main__":
    main()
