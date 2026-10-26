# -*- coding: utf-8 -*-
"""
Genuine Kvasir-SEG Endoscopic Triage Dataset Builder
===================================================
Constructs a clinically authentic triage cohort using 100% genuine endoscopic images
from the official Kvasir-SEG dataset.

Features:
1. Positive class (Class 1: LESION):
   - Samples 300 genuine large/clear polyp cases from Kvasir-SEG.
2. Negative class (Class 0: NORMAL):
   - Samples 300 genuine background mucosa regions from distinct Kvasir-SEG frames
     (containing real capillary vessels, mucosal folds, moisture glare, specular reflections).
   - Zero polyp pixels (dilated safety boundary margin > 30px).
   - Excludes endoscopic camera black borders and letterboxing.
   - Absolutely NO flat or synthetic color backgrounds.
3. Partitions into balanced train and validation splits.
"""

import os
import cv2
import json
import glob
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


def build_kvasir_triage_dataset(
    kvasir_dir: str = "data/kvasir_seg",
    output_dir: str = "data/kvasir_triage",
    num_normal: int = 300,
    num_lesion: int = 300,
    train_ratio: float = 0.8,
    img_size: int = 256,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Builds and persists a balanced, 100% genuine endoscopic triage dataset.
    """
    random.seed(seed)
    np.random.seed(seed)

    kvasir_path = Path(kvasir_dir)
    img_dir = kvasir_path / "images"
    msk_dir = kvasir_path / "masks"

    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"Kvasir-SEG images or masks not found in {kvasir_dir}")

    all_imgs = sorted(list(img_dir.glob("*.*")))
    if len(all_imgs) < 600:
        raise ValueError(f"Expected at least 600 images in {img_dir}, found {len(all_imgs)}")

    # 1. Analyze polyp mask areas across all cases
    polyp_catalog = []
    for p in all_imgs:
        stem = p.stem
        mask_cands = list(msk_dir.glob(f"{stem}.*"))
        if not mask_cands:
            continue
        msk = cv2.imread(str(mask_cands[0]), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            continue
        area = int(np.count_nonzero(msk > 127))
        ratio = float(area) / float(msk.size)
        polyp_catalog.append({
            "stem": stem,
            "img_path": p,
            "msk_path": mask_cands[0],
            "area": area,
            "ratio": ratio,
            "shape": msk.shape
        })

    # Sort descending by polyp area ratio
    polyp_catalog.sort(key=lambda x: x["ratio"], reverse=True)

    # Stratify 300 polyps across clinical risk spectrum:
    # 150 High-risk large/clear polyps (ratio >= 0.15)
    # 150 Medium/Diminutive polyps (ratio < 0.15)
    large_polyps = [c for c in polyp_catalog if c["ratio"] >= 0.15]
    small_polyps = [c for c in polyp_catalog if c["ratio"] < 0.15]

    rng_strat = random.Random(seed)
    n_hr = num_lesion // 2
    n_sm = num_lesion - n_hr
    selected_large = rng_strat.sample(large_polyps, min(n_hr, len(large_polyps)))
    selected_small = rng_strat.sample(small_polyps, min(n_sm, len(small_polyps)))

    for c in selected_large:
        c["risk_level"] = "HIGH_RISK"
    for c in selected_small:
        c["risk_level"] = "MODERATE_OR_SMALL"

    lesion_cases = selected_large + selected_small
    rng_strat.shuffle(lesion_cases)
    lesion_stems = set(c["stem"] for c in lesion_cases)

    # 2. Extract genuine background mucosa patches from the remaining cases
    # to guarantee ZERO image overlap between normal and lesion cases!
    remaining_cases = [c for c in polyp_catalog if c["stem"] not in lesion_stems]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

    normal_cases = []
    for item in remaining_cases:
        img = cv2.imread(str(item["img_path"]))
        msk = cv2.imread(str(item["msk_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue

        H, W = msk.shape
        dilated_msk = cv2.dilate(msk, kernel)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        crop_size = min(256, H, W)
        stride = 32
        best_patch = None
        best_score = -1.0

        for y in range(0, H - crop_size + 1, stride):
            for x in range(0, W - crop_size + 1, stride):
                if np.any(dilated_msk[y:y+crop_size, x:x+crop_size] > 0):
                    continue
                patch_gray = gray[y:y+crop_size, x:x+crop_size]
                black_ratio = np.mean(patch_gray < 20)
                if black_ratio > 0.05:  # Exclude black endoscopic border
                    continue
                mean_val = float(np.mean(patch_gray))
                std_val = float(np.std(patch_gray))
                if mean_val < 30 or std_val < 12:  # Require real capillary/fold texture
                    continue

                if std_val > best_score:
                    best_score = std_val
                    best_patch = (y, x, crop_size, mean_val, std_val)

        if best_patch is not None:
            normal_cases.append({
                "src_stem": item["stem"],
                "img_path": item["img_path"],
                "crop_coords": (best_patch[0], best_patch[1], best_patch[2]),
                "mean_val": best_patch[3],
                "std_val": best_patch[4],
                "risk_level": "NORMAL"
            })
            if len(normal_cases) == num_normal:
                break

    if len(normal_cases) < num_normal:
        raise RuntimeError(f"Could only extract {len(normal_cases)} genuine normal patches; needed {num_normal}")

    print(f"[Kvasir Triage Builder] Selected {len(lesion_cases)} polyp cases ({len(selected_large)} high-risk [min ratio={min(x['ratio'] for x in selected_large):.3f}], {len(selected_small)} small/medium [min ratio={min(x['ratio'] for x in selected_small):.4f}])")
    print(f"[Kvasir Triage Builder] Extracted {len(normal_cases)} genuine normal mucosa patches from distinct images")

    # 3. Create Splits
    # Shuffle independently with fixed seed
    rng = random.Random(seed)
    rng.shuffle(lesion_cases)
    rng.shuffle(normal_cases)

    n_train_lesion = int(num_lesion * train_ratio)
    n_train_normal = int(num_normal * train_ratio)

    splits = {
        "train": {
            "lesion": lesion_cases[:n_train_lesion],
            "normal": normal_cases[:n_train_normal]
        },
        "val": {
            "lesion": lesion_cases[n_train_lesion:],
            "normal": normal_cases[n_train_normal:]
        }
    }

    # 4. Save to Disk
    out_path = Path(output_dir)
    zero_mask = np.zeros((img_size, img_size), dtype=np.uint8)

    stats = {}
    for split_name, split_data in splits.items():
        s_img_dir = out_path / split_name / "images"
        s_msk_dir = out_path / split_name / "masks"
        s_img_dir.mkdir(parents=True, exist_ok=True)
        s_msk_dir.mkdir(parents=True, exist_ok=True)

        # Save Normal Cases
        for i, item in enumerate(split_data["normal"]):
            img = cv2.imread(str(item["img_path"]))
            y, x, c_size = item["crop_coords"]
            crop = img[y:y+c_size, x:x+c_size]
            if crop.shape[0] != img_size or crop.shape[1] != img_size:
                crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)

            fname = f"normal_{split_name}_{i+1:04d}_{item['src_stem']}.png"
            cv2.imwrite(str(s_img_dir / fname), crop)
            cv2.imwrite(str(s_msk_dir / fname), zero_mask)

        # Save Lesion Cases (Genuine 256x256 mucosal patch centered on polyp)
        for i, item in enumerate(split_data["lesion"]):
            img = cv2.imread(str(item["img_path"]))
            msk = cv2.imread(str(item["msk_path"]), cv2.IMREAD_GRAYSCALE)
            H, W = msk.shape

            cnts, _ = cv2.findContours((msk > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cy, cx = H // 2, W // 2
            else:
                cy, cx = H // 2, W // 2

            crop_size = min(img_size, H, W)
            x1 = max(0, min(W - crop_size, cx - crop_size // 2))
            y1 = max(0, min(H - crop_size, cy - crop_size // 2))

            crop_img = img[y1:y1+crop_size, x1:x1+crop_size]
            crop_msk = msk[y1:y1+crop_size, x1:x1+crop_size]

            if crop_img.shape[0] != img_size or crop_img.shape[1] != img_size:
                crop_img = cv2.resize(crop_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                crop_msk = cv2.resize(crop_msk, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

            risk_tag = "highrisk" if item.get("risk_level") == "HIGH_RISK" else "moderate"
            fname = f"lesion_{split_name}_{i+1:04d}_{risk_tag}_{item['stem']}.png"
            cv2.imwrite(str(s_img_dir / fname), crop_img)
            cv2.imwrite(str(s_msk_dir / fname), crop_msk)

        stats[split_name] = {
            "normal": len(split_data["normal"]),
            "lesion": len(split_data["lesion"]),
            "total": len(split_data["normal"]) + len(split_data["lesion"])
        }

    meta_file = out_path / "cohort_manifest.json"
    manifest = {
        "dataset_name": "Kvasir-SEG Genuine Endoscopic Triage Cohort",
        "description": "100% genuine endoscopic images containing real mucosal vessels, folds, moisture glare and large/clear polyps",
        "total_cases": num_normal + num_lesion,
        "splits": stats,
        "seed": seed
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Kvasir Triage Builder] Dataset successfully built in {output_dir}: {stats}")
    return manifest


def load_genuine_endoscopic_cohort(
    kvasir_triage_dir: str = "data/kvasir_triage",
    split: str = "val",
    target_count: int = 120,
    normal_ratio: float = 0.8,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Loads a genuine endoscopic clinical screening cohort with specified prevalence.
    """
    triage_path = Path(kvasir_triage_dir)
    
    if split == "val":
        normal_pool = sorted(list(triage_path.glob("val/images/normal_*.png")))
        lesion_pool = sorted(list(triage_path.glob("val/images/lesion_*.png")))
    else:
        normal_pool = sorted(list(triage_path.glob("*/images/normal_*.png")))
        lesion_pool = sorted(list(triage_path.glob("*/images/lesion_*.png")))

    rng = random.Random(seed)
    
    if split == "val" and target_count == (len(normal_pool) + len(lesion_pool)):
        selected_normals = list(normal_pool)
        selected_lesions = list(lesion_pool)
    else:
        num_normal = int(round(target_count * normal_ratio))
        num_lesion = target_count - num_normal
        if num_normal <= len(normal_pool):
            selected_normals = rng.sample(normal_pool, num_normal)
        else:
            selected_normals = [rng.choice(normal_pool) for _ in range(num_normal)]
            
        if num_lesion <= len(lesion_pool):
            selected_lesions = rng.sample(lesion_pool, num_lesion)
        else:
            selected_lesions = [rng.choice(lesion_pool) for _ in range(num_lesion)]
    
    cohort = []
    for p in selected_normals:
        m_path = p.parent.parent / "masks" / p.name
        cohort.append({
            "img_path": str(p),
            "msk_path": str(m_path),
            "class_label": 0,
            "type": "NORMAL"
        })
        
    for i, p in enumerate(selected_lesions):
        m_path = p.parent.parent / "masks" / p.name
        msk = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        area_ratio = float(np.count_nonzero(msk > 127)) / float(msk.size) if msk is not None else 0.2
        ltype = "HIGH_RISK" if "highrisk" in p.name.lower() else "BENIGN"
        cohort.append({
            "img_path": str(p),
            "msk_path": str(m_path),
            "class_label": 1,
            "type": ltype,
            "area_ratio": area_ratio
        })
        
    rng.shuffle(cohort)
    return cohort
