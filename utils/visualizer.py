"""
Visualization Utilities for Medical Image Segmentation.

Implements:
- save_prediction_comparison: Multi-panel comparison plot
  (a) Original Image, (b) Ground Truth Mask, (c) Predicted Probability Heatmap, (d) Overlay Contours (GT green, Pred red)
- plot_training_curves: Train/Val Loss and Train/Val Dice curves across training epochs
"""

import os
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless environments
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


def _to_2d_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts tensor/numpy mask to a 2D float32 numpy array (H, W)."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data, dtype=np.float32)

    # Squeeze channel and batch dims if single item
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    return arr


def _prepare_image_for_display(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts image tensor or array to [0, 1] range in shape (H, W) or (H, W, 3).
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    img = np.asarray(image, dtype=np.float32)

    # Squeeze single batch dimension if present
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # Handle channel dimension: (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # Normalize image to [0, 1] for display
    img_min, img_max = np.min(img), np.max(img)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    return np.clip(img, 0.0, 1.0)


def save_prediction_comparison(
    image: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    pred_prob: Union[torch.Tensor, np.ndarray],
    save_path: str,
    title: Optional[str] = None
) -> None:
    """
    Saves a 4-panel comparison plot for medical image segmentation evaluation:
    (a) Original Image
    (b) Ground Truth Mask
    (c) Predicted Probability Heatmap (continuous [0, 1] probability with colorbar)
    (d) Overlay Contours (Ground Truth in Green, Prediction in Red over Original Image)

    Args:
        image: Original input image (H, W) or (C, H, W) or (H, W, C).
        target: Ground truth binary mask (H, W) or (1, H, W).
        pred_prob: Model predicted probability map (or logits) (H, W) or (1, H, W).
        save_path: Target file path to save the comparison figure.
        title: Optional figure super-title.
    """
    # Prepare image
    disp_img = _prepare_image_for_display(image)

    # Prepare target
    tgt_2d = _to_2d_numpy(target)
    tgt_bin = (tgt_2d > 0.5).astype(np.uint8)

    # Prepare pred_prob
    pred_2d = _to_2d_numpy(pred_prob)
    # If logits, convert to probabilities
    if np.any(pred_2d < 0.0) or np.any(pred_2d > 1.0):
        pred_2d = 1.0 / (1.0 + np.exp(-np.clip(pred_2d, -30.0, 30.0)))
    pred_prob_clean = np.clip(pred_2d, 0.0, 1.0)
    pred_bin = (pred_prob_clean > 0.5).astype(np.uint8)

    # Create 1x4 figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    fig.subplots_adjust(wspace=0.2)

    is_rgb = disp_img.ndim == 3 and disp_img.shape[2] == 3

    # (a) Original Image
    if is_rgb:
        axes[0].imshow(disp_img)
    else:
        axes[0].imshow(disp_img, cmap="gray")
    axes[0].set_title("(a) Original Image", fontsize=12, fontweight="bold", pad=8)
    axes[0].axis("off")

    # (b) Ground Truth Mask
    axes[1].imshow(tgt_bin, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("(b) Ground Truth Mask", fontsize=12, fontweight="bold", pad=8)
    axes[1].axis("off")

    # (c) Predicted Probability Heatmap
    im_c = axes[2].imshow(pred_prob_clean, cmap="jet", vmin=0.0, vmax=1.0)
    axes[2].set_title("(c) Predicted Probability Heatmap", fontsize=12, fontweight="bold", pad=8)
    axes[2].axis("off")
    cbar = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Probability", rotation=270, labelpad=12, fontsize=10)

    # (d) Overlay Contours
    if is_rgb:
        axes[3].imshow(disp_img)
    else:
        axes[3].imshow(disp_img, cmap="gray")

    # Draw GT Contour in Green (lime)
    has_gt_contour = False
    if np.any(tgt_bin > 0):
        axes[3].contour(tgt_bin, levels=[0.5], colors=["#00FF00"], linewidths=2.0)
        has_gt_contour = True

    # Draw Pred Contour in Red
    has_pred_contour = False
    if np.any(pred_bin > 0):
        axes[3].contour(pred_bin, levels=[0.5], colors=["#FF0000"], linewidths=2.0)
        has_pred_contour = True

    # Build legend
    legend_elements = [
        Line2D([0], [0], color="#00FF00", lw=2, label="Ground Truth (GT)"),
        Line2D([0], [0], color="#FF0000", lw=2, label="Prediction (p > 0.5)")
    ]
    axes[3].legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
        fontsize=9
    )
    axes[3].set_title("(d) Boundary Overlay", fontsize=12, fontweight="bold", pad=8)
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    figsize: tuple = (14, 5)
) -> None:
    """
    Plots training and validation curves across training epochs:
    - Train / Val Loss Curve (with best minimum val loss marked)
    - Train / Val Dice Curve (with best maximum val dice marked)

    Args:
        history: Dictionary containing epoch metrics, e.g.:
                 {
                     'train_loss': [0.6, 0.4, 0.2],
                     'val_loss': [0.5, 0.35, 0.25],
                     'train_dice': [0.65, 0.80, 0.88],
                     'val_dice': [0.60, 0.78, 0.85]
                 }
        save_path: File path to save the generated plot image.
        figsize: Figure dimensions (width, height).
    """
    # Key extraction with fallback support
    train_loss = history.get("train_loss") or history.get("train_losses") or history.get("loss", [])
    val_loss = history.get("val_loss") or history.get("val_losses", [])
    train_dice = history.get("train_dice") or history.get("train_dices") or history.get("dice", [])
    val_dice = history.get("val_dice") or history.get("val_dices", [])

    num_epochs = max(len(train_loss), len(val_loss), len(train_dice), len(val_dice))
    if num_epochs == 0:
        raise ValueError("History dictionary contains no loss or dice records.")

    epochs = list(range(1, num_epochs + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=300)

    # 1. Loss Subplot
    if train_loss:
        ax1.plot(epochs[:len(train_loss)], train_loss, label="Train Loss", color="#1f77b4", lw=2, marker="o", markersize=4)
    if val_loss:
        ax1.plot(epochs[:len(val_loss)], val_loss, label="Val Loss", color="#ff7f0e", lw=2, marker="s", markersize=4)
        best_val_loss_idx = int(np.argmin(val_loss))
        best_val_loss = val_loss[best_val_loss_idx]
        best_epoch_loss = epochs[best_val_loss_idx]
        ax1.scatter(
            [best_epoch_loss], [best_val_loss],
            color="#d62728", s=100, zorder=5,
            edgecolors="black", linewidths=1.5,
            label=f"Min Val Loss ({best_val_loss:.4f} @ Ep {best_epoch_loss})"
        )

    ax1.set_title("Training and Validation Loss", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # 2. Dice Subplot
    if train_dice:
        ax1_dice_epochs = epochs[:len(train_dice)]
        ax2.plot(ax1_dice_epochs, train_dice, label="Train Dice", color="#2ca02c", lw=2, marker="o", markersize=4)
    if val_dice:
        ax2_dice_epochs = epochs[:len(val_dice)]
        ax2.plot(ax2_dice_epochs, val_dice, label="Val Dice", color="#9467bd", lw=2, marker="s", markersize=4)
        best_val_dice_idx = int(np.argmax(val_dice))
        best_val_dice = val_dice[best_val_dice_idx]
        best_epoch_dice = epochs[best_val_dice_idx]
        ax2.scatter(
            [best_epoch_dice], [best_val_dice],
            color="#e377c2", s=100, zorder=5,
            edgecolors="black", linewidths=1.5,
            label=f"Max Val Dice ({best_val_dice:.4f} @ Ep {best_epoch_dice})"
        )

    ax2.set_title("Training and Validation Dice Score", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Dice Score", fontsize=11)
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
