"""
Evaluation Metrics and Model Complexity Utilities for Medical Image Segmentation.

Implements:
- calculate_dice: Dice Similarity Coefficient (DSC)
- calculate_iou: Intersection over Union (Jaccard Index)
- calculate_hd95: 95th Percentile Hausdorff Distance (HD95)
- calculate_precision_recall: Precision, Recall (Sensitivity)
- compute_model_complexity: Parameter count (M) and FLOPs (G) using thop or manual fallback
- MetricTracker: Epoch-level batch metric accumulator
"""

from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff

PrecisionRecall = namedtuple("PrecisionRecall", ["precision", "recall"])


def _to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts torch.Tensor or numpy-like data to a float32 numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def _prepare_mask(
    mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    is_pred: bool = True,
    from_logits: Optional[bool] = None
) -> np.ndarray:
    """
    Preprocesses prediction or target mask into a binary numpy array [0, 1].
    If input values are logits (outside [0, 1]), applies sigmoid first.

    Args:
        from_logits: Explicit interpretation flag for predictions. True = apply
            sigmoid first, False = threshold raw values, None = auto-detect by
            value range (legacy heuristic: sigmoid if min < 0 or max > 1).
            Explicitly passing True/False avoids the ambiguity where a logit
            map that happens to fall inside [0, 1] is mistaken for probabilities.
    """
    arr = _to_numpy(mask)
    if is_pred:
        if from_logits is True:
            arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -30.0, 30.0)))
        elif from_logits is None:
            # Legacy auto-detection heuristic
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -30.0, 30.0)))
        return (arr > threshold).astype(np.float32)
    else:
        return (arr > threshold).astype(np.float32)


def calculate_dice(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    smooth: float = 1e-6,
    from_logits: Optional[bool] = None
) -> float:
    """
    Calculates the Dice Similarity Coefficient (DSC):
        Dice = (2 * |P ∩ T| + smooth) / (|P| + |T| + smooth)

    Args:
        pred: Predicted probabilities/logits or binary mask. Shape (B, 1, H, W), (B, H, W), or (H, W).
        target: Ground truth binary mask, matching pred shape.
        threshold: Binarization threshold for pred.
        smooth: Smoothing epsilon to prevent division by zero.

    Returns:
        float: Mean Dice score across samples in batch (0.0 to 1.0).
    """
    p_bin = _prepare_mask(pred, threshold=threshold, is_pred=True, from_logits=from_logits)
    t_bin = _prepare_mask(target, threshold=threshold, is_pred=False)

    # If batched (3D or 4D), compute per-sample dice and average
    if p_bin.ndim >= 3:
        batch_size = p_bin.shape[0]
        dices = []
        for i in range(batch_size):
            p_flat = p_bin[i].ravel()
            t_flat = t_bin[i].ravel()
            intersection = np.sum(p_flat * t_flat)
            p_sum = np.sum(p_flat)
            t_sum = np.sum(t_flat)

            # Edge case: both ground truth and prediction are completely empty
            if p_sum == 0 and t_sum == 0:
                dices.append(1.0)
            else:
                score = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)
                dices.append(float(score))
        return float(np.mean(dices))
    else:
        p_flat = p_bin.ravel()
        t_flat = t_bin.ravel()
        p_sum = np.sum(p_flat)
        t_sum = np.sum(t_flat)
        if p_sum == 0 and t_sum == 0:
            return 1.0
        intersection = np.sum(p_flat * t_flat)
        return float((2.0 * intersection + smooth) / (p_sum + t_sum + smooth))


def calculate_iou(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    smooth: float = 1e-6,
    from_logits: Optional[bool] = None
) -> float:
    """
    Calculates the Intersection over Union (IoU / Jaccard Index):
        IoU = (|P ∩ T| + smooth) / (|P ∪ T| + smooth)

    Args:
        pred: Predicted probabilities/logits or binary mask.
        target: Ground truth binary mask.
        threshold: Binarization threshold.
        smooth: Smoothing epsilon.

    Returns:
        float: Mean IoU score across samples in batch (0.0 to 1.0).
    """
    p_bin = _prepare_mask(pred, threshold=threshold, is_pred=True, from_logits=from_logits)
    t_bin = _prepare_mask(target, threshold=threshold, is_pred=False)

    if p_bin.ndim >= 3:
        batch_size = p_bin.shape[0]
        ious = []
        for i in range(batch_size):
            p_flat = p_bin[i].ravel()
            t_flat = t_bin[i].ravel()
            p_sum = np.sum(p_flat)
            t_sum = np.sum(t_flat)

            if p_sum == 0 and t_sum == 0:
                ious.append(1.0)
            else:
                intersection = np.sum(p_flat * t_flat)
                union = p_sum + t_sum - intersection
                score = (intersection + smooth) / (union + smooth)
                ious.append(float(score))
        return float(np.mean(ious))
    else:
        p_flat = p_bin.ravel()
        t_flat = t_bin.ravel()
        p_sum = np.sum(p_flat)
        t_sum = np.sum(t_flat)
        if p_sum == 0 and t_sum == 0:
            return 1.0
        intersection = np.sum(p_flat * t_flat)
        union = p_sum + t_sum - intersection
        return float((intersection + smooth) / (union + smooth))


def _extract_contour_points(mask: np.ndarray) -> np.ndarray:
    """
    Extracts 2D boundary/contour coordinates (N, 2) from a 2D binary mask.
    Returns empty (0, 2) array if mask is empty.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return np.empty((0, 2), dtype=np.float64)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)

    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
    return pts


def calculate_hd95_single(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    empty_value: float = 100.0
) -> float:
    """
    Computes 95th percentile Hausdorff Distance (HD95) for a single 2D image pair.

    Boundary contours are extracted from both masks. The nearest Euclidean distances
    from each contour point in Pred to Target and vice-versa are evaluated via KD-Tree.
    The directed Hausdorff distance (via scipy.spatial.distance.directed_hausdorff)
    defines the maximum boundary deviation, while HD95 calculates the 95th percentile
    of the bidirectional boundary point-to-point distances for outlier robustness.

    Args:
        pred_mask: 2D binary mask of predictions (H, W).
        target_mask: 2D binary mask of ground truth (H, W).
        empty_value: Penalty distance returned when one mask is empty while the other is non-empty.

    Returns:
        float: HD95 distance in pixels (0.0 if both empty, empty_value if only one is empty).
    """
    pts_pred = _extract_contour_points(pred_mask)
    pts_target = _extract_contour_points(target_mask)

    len_pred = len(pts_pred)
    len_target = len(pts_target)

    # Edge cases
    if len_pred == 0 and len_target == 0:
        return 0.0
    if len_pred == 0 or len_target == 0:
        return float(empty_value)

    # Also reference directed_hausdorff from scipy.spatial.distance
    # directed_hausdorff computes the 100th percentile (maximum) distance.
    # For HD95, we compute the 95th percentile of distances in both directions using cKDTree.
    tree_target = cKDTree(pts_target)
    dist_pred_to_target, _ = tree_target.query(pts_pred)

    tree_pred = cKDTree(pts_pred)
    dist_target_to_pred, _ = tree_pred.query(pts_target)

    hd95_p2t = np.percentile(dist_pred_to_target, 95)
    hd95_t2p = np.percentile(dist_target_to_pred, 95)

    return float(max(hd95_p2t, hd95_t2p))


def calculate_hd95(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    empty_value: float = 100.0,
    from_logits: Optional[bool] = None
) -> float:
    """
    Calculates 95% Hausdorff Distance (HD95) over a batch or single image.

    Args:
        pred: Predicted logits/probabilities or mask. Shape (B, 1, H, W), (B, H, W), or (H, W).
        target: Ground truth binary mask.
        threshold: Binarization threshold for pred.
        empty_value: Penalty returned if one mask is empty and the other is non-empty (default 100.0).

    Returns:
        float: Mean HD95 across the batch.
    """
    p_bin = _prepare_mask(pred, threshold=threshold, is_pred=True, from_logits=from_logits)
    t_bin = _prepare_mask(target, threshold=threshold, is_pred=False)

    # Squeeze channel dim if (B, 1, H, W) -> (B, H, W)
    if p_bin.ndim == 4 and p_bin.shape[1] == 1:
        p_bin = p_bin[:, 0, :, :]
    if t_bin.ndim == 4 and t_bin.shape[1] == 1:
        t_bin = t_bin[:, 0, :, :]

    if p_bin.ndim == 3:
        batch_size = p_bin.shape[0]
        hd95_scores = []
        for i in range(batch_size):
            score = calculate_hd95_single(p_bin[i], t_bin[i], empty_value=empty_value)
            hd95_scores.append(score)
        return float(np.mean(hd95_scores))
    elif p_bin.ndim == 2:
        return calculate_hd95_single(p_bin, t_bin, empty_value=empty_value)
    else:
        raise ValueError(f"Unsupported mask shape for HD95: {p_bin.shape}")


def calculate_precision_recall(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    smooth: float = 1e-6,
    from_logits: Optional[bool] = None
) -> PrecisionRecall:
    """
    Calculates Precision and Recall (Sensitivity):
        Precision = TP / (TP + FP + smooth)
        Recall    = TP / (TP + FN + smooth)

    Args:
        pred: Predicted probabilities/logits.
        target: Ground truth binary mask.
        threshold: Binarization threshold.
        smooth: Smoothing epsilon.

    Returns:
        PrecisionRecall(precision=float, recall=float): NamedTuple accessible by attribute or tuple unpacking.
    """
    p_bin = _prepare_mask(pred, threshold=threshold, is_pred=True, from_logits=from_logits)
    t_bin = _prepare_mask(target, threshold=threshold, is_pred=False)

    p_flat = p_bin.ravel()
    t_flat = t_bin.ravel()

    tp = np.sum((p_flat == 1.0) & (t_flat == 1.0))
    fp = np.sum((p_flat == 1.0) & (t_flat == 0.0))
    fn = np.sum((p_flat == 0.0) & (t_flat == 1.0))

    if tp == 0 and fp == 0 and fn == 0:
        return PrecisionRecall(precision=1.0, recall=1.0)

    precision = float((tp + smooth) / (tp + fp + smooth))
    recall = float((tp + smooth) / (tp + fn + smooth))

    return PrecisionRecall(precision=precision, recall=recall)


def compute_model_complexity(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 256, 256),
    device: str = "cpu"
) -> Dict[str, Union[float, int]]:
    """
    Computes model complexity:
    - Number of Parameters (in Millions, M)
    - FLOPs (in GigaFLOPs, G)

    Uses `thop.profile` if available; otherwise falls back to exact parameter counting
    and theoretical convolution FLOP estimation.

    Args:
        model: PyTorch nn.Module.
        input_size: Input tensor dimensions (B, C, H, W).
        device: Device to place the dummy input on ('cpu' or 'cuda').

    Returns:
        dict: {
            'params_m': float,       # Parameters in Millions (M)
            'flops_g': float,        # Computations in GigaFLOPs (G)
            'total_params': int,     # Exact parameter count
            'total_flops': int,      # Exact FLOP count
            'method': str            # 'thop' or 'manual_fallback'
        }
    """
    was_training = model.training
    model.eval()
    orig_device = next(model.parameters()).device

    total_params = sum(p.numel() for p in model.parameters())
    params_m = round(total_params / 1e6, 4)

    flops_g = 0.0
    total_flops = 0
    method = "manual_fallback"

    # Attempt profiling with thop on a clean deepcopy
    try:
        from copy import deepcopy
        from thop import profile
        model_copy = deepcopy(model).to(device)
        dummy_input = torch.randn(*input_size, device=device)
        flops, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
        total_flops = int(flops)
        flops_g = round(total_flops / 1e9, 4)
        method = "thop"
        del model_copy
    except Exception:
        # Fallback estimation: sum FLOPs of Conv2d and Linear layers
        model = model.to(device)
        total_flops = 0

        hooks = []

        def conv_hook(self, inp, out):
            nonlocal total_flops
            # Conv2d FLOPs = 2 * C_in * K_h * K_w * H_out * W_out * C_out / groups
            batch_size = out.shape[0]
            out_h, out_w = out.shape[2], out.shape[3]
            c_in = self.in_channels
            c_out = self.out_channels
            k_h, k_w = self.kernel_size
            flops = 2 * batch_size * (c_in // self.groups) * k_h * k_w * out_h * out_w * c_out
            if self.bias is not None:
                flops += batch_size * c_out * out_h * out_w
            total_flops += flops

        def linear_hook(self, inp, out):
            nonlocal total_flops
            batch_size = out.shape[0]
            flops = 2 * batch_size * self.in_features * self.out_features
            if self.bias is not None:
                flops += batch_size * self.out_features
            total_flops += flops

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(linear_hook))

        try:
            with torch.no_grad():
                dummy_input = torch.randn(*input_size, device=device)
                model(dummy_input)
            flops_g = round(total_flops / 1e9, 4)
        except Exception:
            # If forward pass fails, rough estimate based on conv weights
            total_flops = sum(p.numel() * 2 for p in model.parameters())
            flops_g = round(total_flops / 1e9, 4)
        finally:
            for h in hooks:
                h.remove()

    # Restore model state
    model.to(orig_device)
    if was_training:
        model.train()

    return {
        "params_m": params_m,
        "flops_g": flops_g,
        "total_params": total_params,
        "total_flops": total_flops,
        "method": method
    }


class MetricTracker:
    """
    Accumulates and aggregates batch-level metrics across an evaluation epoch.

    Tracks:
    - Dice
    - IoU (Jaccard)
    - HD95 (95% Hausdorff Distance)
    - Precision
    - Recall
    - Any custom metric (e.g. loss values) via update_metric()
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        if metrics is None:
            self.metric_names = ["dice", "iou", "hd95", "precision", "recall"]
        else:
            self.metric_names = list(metrics)

        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        """Resets all tracked metric sums and sample counts."""
        self.sums = {name: 0.0 for name in self.metric_names}
        self.counts = {name: 0 for name in self.metric_names}

    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        empty_hd95_value: float = 100.0,
        from_logits: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Calculates all standard metrics for the given batch and updates running totals.

        Args:
            pred: Predicted probabilities/logits.
            target: Ground truth binary mask.
            threshold: Decision threshold for binarization.
            empty_hd95_value: Penalty HD95 when one mask is empty.

        Returns:
            dict: Batch metric values computed for this step.
        """
        # Determine batch size
        if isinstance(pred, torch.Tensor):
            b_size = pred.shape[0] if pred.ndim >= 3 else 1
        else:
            b_size = pred.shape[0] if pred.ndim >= 3 else 1

        batch_metrics = {}

        if "dice" in self.metric_names:
            d = calculate_dice(pred, target, threshold=threshold, from_logits=from_logits)
            batch_metrics["dice"] = d
            self.update_metric("dice", d, n=b_size)

        if "iou" in self.metric_names:
            j = calculate_iou(pred, target, threshold=threshold, from_logits=from_logits)
            batch_metrics["iou"] = j
            self.update_metric("iou", j, n=b_size)

        if "hd95" in self.metric_names:
            h = calculate_hd95(pred, target, threshold=threshold, empty_value=empty_hd95_value, from_logits=from_logits)
            batch_metrics["hd95"] = h
            self.update_metric("hd95", h, n=b_size)

        if "precision" in self.metric_names or "recall" in self.metric_names:
            pr = calculate_precision_recall(pred, target, threshold=threshold, from_logits=from_logits)
            if "precision" in self.metric_names:
                batch_metrics["precision"] = pr.precision
                self.update_metric("precision", pr.precision, n=b_size)
            if "recall" in self.metric_names:
                batch_metrics["recall"] = pr.recall
                self.update_metric("recall", pr.recall, n=b_size)

        return batch_metrics

    def update_metric(self, name: str, value: float, n: int = 1) -> None:
        """
        Updates a specific metric or custom loss scalar.

        Args:
            name: Metric identifier (e.g. 'loss_bce', 'dice').
            value: Mean value for the current step.
            n: Number of samples in the current batch step.
        """
        if name not in self.sums:
            self.sums[name] = 0.0
            self.counts[name] = 0
            if name not in self.metric_names:
                self.metric_names.append(name)

        self.sums[name] += float(value) * n
        self.counts[name] += n

    def compute(self) -> Dict[str, float]:
        """
        Computes the average across all accumulated batches.

        Returns:
            dict: {metric_name: mean_value}
        """
        results = {}
        for name in self.metric_names:
            count = self.counts.get(name, 0)
            if count > 0:
                results[name] = round(self.sums[name] / count, 4)
            else:
                results[name] = 0.0
        return results

    def get_results(self) -> Dict[str, float]:
        """Alias for compute()."""
        return self.compute()

    def summary(self) -> str:
        """Returns a formatted summary string of all tracked metrics."""
        res = self.compute()
        parts = [f"{k.upper()}: {v:.4f}" for k, v in res.items()]
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"<MetricTracker: {self.summary()}>"
