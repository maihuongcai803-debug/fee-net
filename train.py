import os
import sys
import time
import json
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.fee_net import FEENet
from models.fee_net_early_exit import FEENetEarlyExit
from models.unet_plus_plus import UNetPlusPlus
from datasets.dataset_loader import get_dataloaders
from datasets.real_medical_loader import get_kvasir_dataloaders, get_kvasir_triage_dataloaders
from utils.losses import CompoundLoss, DiceLoss
from utils.metrics import MetricTracker, compute_model_complexity
from utils.visualizer import plot_training_curves, save_prediction_comparison

import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

TRIAGE_POSITIVE_PIXEL_THRESHOLD = 10  # min mask pixels for a case to count as lesion (at img_size=256)

def parse_args():
    # Pre-parse to pick up --config before defining defaults from it
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="configs/default_config.yaml")
    cfg_args, _ = pre.parse_known_args()
    cfg = {}
    if os.path.exists(cfg_args.config):
        with open(cfg_args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    cfg_train = cfg.get("training", {})
    cfg_data = cfg.get("dataset", {})
    cfg_loss = cfg.get("loss", {})

    parser = argparse.ArgumentParser(description="Train Medical Image Segmentation Models")
    parser.add_argument("--model", type=str, default="fee_net",
                        choices=["unet", "attention_unet", "unet_plus_plus", "nested_unet", "fee_net",
                                 "fee_net_no_dwt", "fee_net_no_egm", "fee_net_no_bl", "fee_net_early_exit"],
                        help="Model architecture to train")
    parser.add_argument("--dataset", type=str, default="kvasir", choices=["kvasir", "kvasir_triage", "synthetic"],
                        help="Dataset to train on (kvasir: Kvasir-SEG real endoscopy, kvasir_triage: genuine normal+polyp cohort, synthetic: synthetic medseg)")
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Fold index for Kvasir 5-Fold cross-validation (0-4)")
    parser.add_argument("--epochs", type=int, default=cfg_train.get("epochs", 50),
                        help="Number of training epochs (config default: 50)")
    parser.add_argument("--batch_size", type=int, default=cfg_data.get("batch_size", 16),
                        help="Batch size for training (config default: 16)")
    parser.add_argument("--lr", type=float, default=cfg_train.get("lr", 3e-4), help="Learning rate")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42), help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=cfg_train.get("early_stopping", 10),
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--img_size", type=int, default=cfg_data.get("img_size", 256), help="Input image dimension")
    parser.add_argument("--lambda_boundary", type=float, default=cfg_loss.get("lambda_boundary", 0.1),
                        help="Boundary-IoU loss weight for FEE-Net variants (config default: 0.1)")
    parser.add_argument("--num_workers", type=int, default=cfg_data.get("num_workers", 0),
                        help="DataLoader worker processes (0 = main process; raise on Linux/Kaggle)")
    parser.add_argument("--device", type=str, default=cfg.get("device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="YAML config providing defaults; CLI flags override")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_name", type=str, default=None, help="Custom checkpoint filename")
    parser.add_argument("--no_detach_triage", action="store_false", dest="detach_triage", default=True,
                        help="Disable stop-gradient into shared backbone for the triage head "
                             "(stop-gradient is ON by default)")
    return parser.parse_args()

def build_model(model_name, device, detach_triage=True):
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
    elif model_name == "fee_net_early_exit":
        model = FEENetEarlyExit(in_channels=3, num_classes=1, base_channels=48, detach_triage_feature=detach_triage)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model.to(device)

def train_one_epoch(model, dataloader, optimizer, criterion, device, is_fee_net, is_no_bl=False, is_early_exit=False, scaler=None):
    model.train()
    tracker = MetricTracker()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None and device == 'cuda')):
            if is_early_exit:
                outputs = model(images, return_aux=True)
                seg_logits, edge_logits, triage_logits, quality_logits, aux_preds = outputs
                loss, loss_dict = criterion(seg_logits, edge_logits, masks)
                if aux_preds is not None:
                    aux_loss = 0.0
                    for aux in aux_preds:
                        aux_up = F.interpolate(aux, size=masks.shape[2:], mode='bilinear', align_corners=True)
                        l_aux, _ = criterion(aux_up, None, masks)
                        aux_loss += l_aux
                    loss = loss + 0.3 * (aux_loss / len(aux_preds))
                triage_targets = (masks.sum(dim=(1, 2, 3)) > TRIAGE_POSITIVE_PIXEL_THRESHOLD).long()
                l_triage = F.cross_entropy(triage_logits, triage_targets)
                loss = loss + 0.5 * l_triage
            elif is_fee_net:
                outputs = model(images, return_aux=True)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    seg_logits, edge_logits, aux_preds = outputs
                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    seg_logits, edge_logits = outputs
                    aux_preds = None
                else:
                    seg_logits = outputs
                    edge_logits = None
                    aux_preds = None

                if is_no_bl:
                    # Ablation: without boundary loss
                    loss, loss_dict = criterion(seg_logits, None, masks)
                else:
                    loss, loss_dict = criterion(seg_logits, edge_logits, masks)

                # Multi-scale deep supervision auxiliary loss (0.3 weight factor)
                if aux_preds is not None:
                    aux_loss = 0.0
                    for aux in aux_preds:
                        aux_up = F.interpolate(aux, size=masks.shape[2:], mode='bilinear', align_corners=True)
                        l_aux, _ = criterion(aux_up, None, masks)
                        aux_loss += l_aux
                    loss = loss + 0.3 * (aux_loss / len(aux_preds))
            else:
                seg_logits = model(images)
                loss, loss_dict = criterion(seg_logits, None, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        tracker.update(seg_logits.detach(), masks.detach(), from_logits=True)

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.get_results()
    metrics["loss"] = epoch_loss
    return metrics

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, is_fee_net, is_no_bl=False, is_early_exit=False):
    model.eval()
    tracker = MetricTracker()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        if is_early_exit:
            out = model(images, force_full=True, return_raw_logits=True)
            seg_logits = out[0] if isinstance(out, tuple) else out
            loss, _ = criterion(seg_logits, None, masks)
        elif is_fee_net:
            out = model(images)
            seg_logits = out[0] if isinstance(out, tuple) else out
            loss, _ = criterion(seg_logits, None, masks)
        else:
            seg_logits = model(images)
            loss, _ = criterion(seg_logits, None, masks)

        running_loss += loss.item() * images.size(0)
        tracker.update(seg_logits, masks, from_logits=True)

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.get_results()
    metrics["loss"] = epoch_loss
    return metrics

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    print(f"\n{'='*65}")
    print(f" Starting Experiment: {args.model.upper()} on Device: {args.device}")
    print(f" Dataset: {args.dataset.upper()} (Fold: {args.fold}) | Seed: {args.seed}")
    print(f" Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f" Early Stopping Patience: {args.patience}")
    print(f"{'='*65}\n")

    # Dataloaders
    if args.dataset == "kvasir":
        train_loader, val_loader, test_loader = get_kvasir_dataloaders(
            fold=args.fold,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            return_test=True
        )
    elif args.dataset == "kvasir_triage":
        train_loader, val_loader, test_loader = get_kvasir_triage_dataloaders(
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            return_test=True
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers
        )

    model = build_model(args.model, args.device, detach_triage=args.detach_triage)
    
    # Safe model complexity calculation on a detached clone
    dummy_model = build_model(args.model, "cpu", detach_triage=args.detach_triage)
    comp = compute_model_complexity(dummy_model, input_size=(1, 3, args.img_size, args.img_size), device="cpu")
    params, flops = comp["params_m"], comp["flops_g"]
    del dummy_model

    print(f"Model Complexity -> Parameters: {params:.2f}M | FLOPs: {flops:.2f}G")

    is_fee = "fee_net" in args.model
    is_early_exit = args.model == "fee_net_early_exit"
    is_no_bl = args.model == "fee_net_no_bl"

    # Loss function
    if is_no_bl or not is_fee:
        # Standard BCE + Dice
        criterion = CompoundLoss(lambda_bce=1.0, lambda_dice=1.0, lambda_boundary=0.0)
    else:
        # Balanced Compound Loss with boundary prior regularization
        criterion = CompoundLoss(lambda_bce=1.0, lambda_dice=1.0, lambda_boundary=args.lambda_boundary)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if (args.device == "cuda" and torch.cuda.is_available()) else None

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "val_iou": [], "val_hd95": []}
    best_val_dice = -1.0
    patience_counter = 0
    
    if args.ckpt_name:
        best_ckpt_path = os.path.join(args.save_dir, args.ckpt_name)
    else:
        best_ckpt_path = os.path.join(args.save_dir, f"{args.model}_fold{args.fold}_seed{args.seed}_best.pth")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        train_res = train_one_epoch(model, train_loader, optimizer, criterion, args.device, is_fee, is_no_bl, is_early_exit, scaler)
        val_res = evaluate_epoch(model, val_loader, criterion, args.device, is_fee, is_no_bl, is_early_exit)
        scheduler.step()

        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["train_dice"].append(train_res["dice"])
        history["val_dice"].append(val_res["dice"])
        history["val_iou"].append(val_res["iou"])
        history["val_hd95"].append(val_res["hd95"])

        epoch_time = time.time() - t_epoch_start

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_res['loss']:.4f}, Dice: {train_res['dice']*100:.2f}% | "
              f"Val Loss: {val_res['loss']:.4f}, Dice: {val_res['dice']*100:.2f}%, "
              f"mIoU: {val_res['iou']*100:.2f}%, HD95: {val_res['hd95']:.2f}")

        # Checkpoint saving
        if val_res["dice"] > best_val_dice:
            best_val_dice = val_res["dice"]
            patience_counter = 0
            clean_state_dict = {k: v for k, v in model.state_dict().items() if not k.endswith("total_ops") and not k.endswith("total_params")}
            save_payload = {
                "epoch": epoch,
                "model_state_dict": clean_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "val_iou": val_res["iou"],
                "val_hd95": val_res["hd95"],
                "params_m": params,
                "flops_g": flops,
                "model": args.model,
                "fold": args.fold,
                "seed": args.seed,
                "history": history
            }
            torch.save(save_payload, best_ckpt_path)
            print(f"  >>> Best model saved to {best_ckpt_path} (Val Dice: {best_val_dice*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  >>> Early stopping triggered after {args.patience} epochs without improvement.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Val Dice: {best_val_dice*100:.2f}%")

    # Plot training curves
    run_tag = f"{args.model}_fold{args.fold}_seed{args.seed}"
    curve_path = os.path.join("assets", f"{run_tag}_training_curves.png")
    plot_training_curves(history, curve_path)

    # Save history json
    history_path = os.path.join(args.save_dir, f"{run_tag}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
