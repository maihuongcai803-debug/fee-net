# -*- coding: utf-8 -*-
"""
Official Kvasir-SEG Real Clinical Dataset Loader
================================================
Exclusively loads the official Simula Kvasir-SEG endoscopic polyp dataset.
Includes:
- Absolute data integrity check (Strictly 1000 official clinical cases)
- 5-Fold Stratified Cross-Validation partitioning based on true lesion area ratio
- Standard PyTorch Dataset and DataLoader with medical augmentations
- NO synthetic fallback generators allowed (Ensures 100% genuine scientific data)
"""

import os
import sys
import glob
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional

# Augmentations
try:
    from datasets.transforms import get_train_transforms, get_val_transforms
except ImportError:
    from transforms import get_train_transforms, get_val_transforms


def compute_mask_lesion_ratio(mask_path: str) -> float:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0
    return float(np.count_nonzero(mask > 127)) / float(mask.size)


def _default_kvasir_dir() -> str:
    """Project-relative default (portable). Resolved from repo root two levels up."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kvasir_seg")


def generate_official_stratified_5fold(
    data_dir: str = None,
    seed: int = 42,
    force: bool = False
) -> Dict[str, Dict[str, List[str]]]:
    """
    P0 fix (v2 protocol): partitions the 1000 genuine Kvasir-SEG cases into
    a **global independent test set** plus 5 stratified folds.

    v2 scheme:
    1. Stratify all 1000 cases into 5 strata by lesion-area-ratio quantiles.
    2. Hold out 30 cases per stratum (150 total, 15%) as a *global* test set
       that is excluded from all cross-validation training/validation.
    3. 5-fold stratified CV on the remaining 850 cases (170 val per fold).

    This guarantees TEST ∩ (any fold's train ∪ val) = ∅, eliminating the
    selection bias of the v1 protocol where `test == fold val` (see
    04_Experiment_Logs/Exp_027 and 06_Code_Docs/Code_Review_2026-10-01.md).

    Overwrite protection: if splits.json already exists, the existing splits
    are returned unchanged unless ``force=True``. When force=True overwrites a
    legacy v1 file, a backup is written to splits_v1_backup.json first.
    """
    if data_dir is None:
        data_dir = _default_kvasir_dir()
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    split_path = os.path.join(data_dir, "splits.json")
    if os.path.exists(split_path):
        with open(split_path, "r") as fp:
            existing = json.load(fp)
        if "test" in existing:
            print(f"[Splits] {split_path} already exists (v2, with independent test); "
                  f"returning existing splits (pass force=True to regenerate).")
            return existing
        if not force:
            print(f"[Splits] {split_path} exists but is legacy v1 (no 'test' key). "
                  f"Re-run with force=True to migrate to the v2 protocol "
                  f"(a backup will be written first).")
            return existing
        import shutil
        shutil.copyfile(split_path, os.path.join(data_dir, "splits_v1_backup.json"))
        print(f"[Splits] Legacy v1 splits backed up to splits_v1_backup.json")

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Official Kvasir-SEG directories not found at {data_dir}. "
                                f"Please run tools/download_official_kvasir.py first.")
        
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    if len(img_files) != 1000:
        raise ValueError(f"Expected exactly 1000 official Kvasir-SEG images, but found {len(img_files)}!")

    stems = []
    ratios = []
    
    for img_p in img_files:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        # Locate corresponding mask
        mask_candidates = glob.glob(os.path.join(mask_dir, f"{stem}.*"))
        if not mask_candidates:
            raise FileNotFoundError(f"Missing ground truth mask for {stem}")
        ratio = compute_mask_lesion_ratio(mask_candidates[0])
        stems.append(stem)
        ratios.append(ratio)
        
    stems = np.array(stems)
    ratios = np.array(ratios)
    
    # 5 Strata based on quantiles
    bins = np.percentile(ratios, [20, 40, 60, 80])
    strata_labels = np.digitize(ratios, bins)
    
    rng = np.random.RandomState(seed)

    # ---- v2: global stratified test holdout (30 per stratum = 150) ----
    test_idx: List[int] = []
    for s_idx in range(5):
        s_samples = np.where(strata_labels == s_idx)[0]
        rng.shuffle(s_samples)
        test_idx.extend(s_samples[:30].tolist())
    test_idx = set(test_idx)

    # ---- 5-fold stratified CV on the remaining 850 ----
    fold_indices = [[] for _ in range(5)]
    for s_idx in range(5):
        s_samples = np.where(strata_labels == s_idx)[0]
        s_cv = [i for i in s_samples if i not in test_idx]
        rng.shuffle(s_cv)
        sub_folds = np.array_split(np.array(s_cv), 5)
        for f in range(5):
            fold_indices[f].extend(sub_folds[f].tolist())

    splits = {"test": sorted(stems[sorted(test_idx)].tolist())}
    for f in range(5):
        val_idx = fold_indices[f]
        train_idx = [idx for k in range(5) if k != f for idx in fold_indices[k]]
        splits[f"fold_{f}"] = {
            "train": sorted(stems[train_idx].tolist()),
            "val": sorted(stems[val_idx].tolist())
        }

    # Safety checks: global test must be disjoint from every fold
    test_set = set(splits["test"])
    assert len(test_set) == 150, f"Expected 150 test cases, got {len(test_set)}"
    for f in range(5):
        assert not (test_set & set(splits[f"fold_{f}"]["train"])), f"test leaked into fold {f} train"
        assert not (test_set & set(splits[f"fold_{f}"]["val"])), f"test leaked into fold {f} val"

    with open(split_path, "w") as fp:
        json.dump(splits, fp, indent=2)

    print(f"[Verified] v2 splits saved to {split_path}: "
          f"150 independent test + 5 stratified folds on 850 cases")
    return splits


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        fold: int = 0,
        img_size: int = 256,
        transform = None
    ):
        super().__init__()
        if data_dir is None:
            data_dir = _default_kvasir_dir()
        self.data_dir = data_dir
        self.split = split
        self.fold = fold
        self.img_size = img_size

        split_json = os.path.join(data_dir, "splits.json")
        if not os.path.exists(split_json):
            generate_official_stratified_5fold(data_dir)

        with open(split_json, "r") as fp:
            all_splits = json.load(fp)

        # Shared independent test set (P0 fix). Fallback to fold-val keeps
        # legacy / Kaggle environments runnable with the old behaviour.
        if split == "test":
            if "test" in all_splits:
                self.stems = list(all_splits["test"])
                self.is_shared_test = True
            else:
                print("[KvasirSegDataset] WARNING: legacy v1 splits.json has no independent "
                      "'test' key; falling back to fold-val (old behaviour). Regenerate "
                      "with generate_official_stratified_5fold(force=True) to migrate to v2.")
                self.stems = list(all_splits[f"fold_{fold}"]["val"])
                self.is_shared_test = False
        elif split == "train":
            self.stems = list(all_splits[f"fold_{fold}"]["train"])
            self.is_shared_test = False
        else:
            self.stems = list(all_splits[f"fold_{fold}"]["val"])
            self.is_shared_test = False

        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")

        # Cache stem -> absolute path mapping once (was per-access globbing)
        self.img_paths = {s: glob.glob(os.path.join(self.img_dir, f"{s}.*"))[0] for s in self.stems}
        self.mask_paths = {s: glob.glob(os.path.join(self.mask_dir, f"{s}.*"))[0] for s in self.stems}

        if transform is None:
            if split == "train":
                self.transform = get_train_transforms(img_size)
            else:
                self.transform = get_val_transforms(img_size)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img_p = self.img_paths[stem]
        mask_p = self.mask_paths[stem]
        
        # Read RGB image
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read Grayscale Mask
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            try:
                res = self.transform(image=img, mask=mask)
            except TypeError:
                res = self.transform(img, mask)
            
            if isinstance(res, dict):
                img = res["image"]
                mask = res["mask"]
            elif isinstance(res, (tuple, list)):
                img, mask = res[0], res[1]
            
        # Ensure mask is (1, H, W)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
        return img.float(), mask.float()


def get_kvasir_dataloaders(
    data_dir: str = None,
    fold: int = 0,
    batch_size: int = 16,
    img_size: int = 256,
    num_workers: int = 0,
    return_test: bool = False
):
    train_ds = KvasirSegDataset(data_dir=data_dir, split="train", fold=fold, img_size=img_size)
    val_ds = KvasirSegDataset(data_dir=data_dir, split="val", fold=fold, img_size=img_size)
    # P0 fix: the test loader now draws from the shared independent test set
    # (falls back to fold-val with a warning on legacy splits).
    test_ds = KvasirSegDataset(data_dir=data_dir, split="test", fold=fold, img_size=img_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if return_test:
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class KvasirTriagePairedDataset(Dataset):
    """
    Genuine paired triage dataset (data/kvasir_triage/train and val)
    containing authentic normal mucosa (mask=0, class=0) and genuine polyp cases (mask>0, class=1).
    """
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        img_size: int = 256,
        transform = None
    ):
        super().__init__()
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kvasir_triage")
        self.split_dir = os.path.join(data_dir, split)
        self.img_files = sorted(glob.glob(os.path.join(self.split_dir, "images", "*.png")))
        self.mask_dir = os.path.join(self.split_dir, "masks")
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p = self.img_files[idx]
        stem = os.path.splitext(os.path.basename(img_p))[0]
        mask_p = os.path.join(self.mask_dir, f"{stem}.png")
        
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(mask_p):
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            
        if self.transform is not None:
            try:
                res = self.transform(image=img, mask=mask)
                img, mask = res['image'], res['mask']
            except (TypeError, KeyError):
                res = self.transform(img, mask)
                img, mask = res[0], res[1]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            return img.float(), mask.float()
        else:
            # Default Kaggle-aligned [0, 1] preprocessing
            img_res = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask_res = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            t_img = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
            t_mask = torch.from_numpy(mask_res).unsqueeze(0).float()
            return t_img, t_mask


def get_kvasir_triage_dataloaders(
    data_dir: str = None,
    batch_size: int = 16,
    img_size: int = 256,
    num_workers: int = 0,
    return_test: bool = False
):
    train_ds = KvasirTriagePairedDataset(data_dir=data_dir, split="train", img_size=img_size)
    val_ds = KvasirTriagePairedDataset(data_dir=data_dir, split="val", img_size=img_size)
    # NOTE (limitation): the triage cohort has no independent test partition yet;
    # test currently mirrors val. Known issue tracked in 06_Code_Docs/Code_Review_2026-10-01.md.
    test_ds = KvasirTriagePairedDataset(data_dir=data_dir, split="val", img_size=img_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if return_test:
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


# Backward compatibility aliases
create_kvasir_stratified_splits = generate_official_stratified_5fold
def download_or_generate_kvasir_seg(data_dir: str = None, download: bool = True):
    return data_dir or _default_kvasir_dir()
KvasirDataLoaderBundle = tuple


if __name__ == "__main__":
    print("="*65)
    print(" Verifying Official Kvasir-SEG Real Dataset Pipeline")
    print("="*65)
    # Idempotent: returns existing splits.json unchanged unless force=True
    splits = generate_official_stratified_5fold()
    print("Folds summary:")
    for f in range(5):
        t_len = len(splits[f"fold_{f}"]["train"])
        v_len = len(splits[f"fold_{f}"]["val"])
        print(f"  Fold {f}: {t_len} Train cases, {v_len} Validation cases")

    if "test" not in splits:
        raise SystemExit("Legacy v1 splits detected. Re-run with "
                         "generate_official_stratified_5fold(force=True) to migrate to v2.")
    print(f"Shared independent test set: {len(splits['test'])} cases")

    tl, vl, testl = get_kvasir_dataloaders(fold=0, batch_size=16, return_test=True)
    imgs, msks = next(iter(tl))
    print(f"Sample Batch Check: Images {imgs.shape}, Masks {msks.shape}")
    print(f"Test loader size: {len(testl.dataset)} cases "
          f"(independent: {getattr(testl.dataset, 'is_shared_test', 'n/a')})")
    print("Official Kvasir-SEG Verified!")
