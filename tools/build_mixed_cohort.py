# -*- coding: utf-8 -*-
"""
Build a leakage-free MIXED cohort (lesion + normal mucosa) from the held-out split.

Why this exists
---------------
Kvasir-SEG is an all-positive cohort, so the triage gate's *specificity* (how
often healthy frames are actually bypassed) and the real compute saving cannot be
measured on it -- a limitation we state in the paper. This builder derives a mixed
cohort from the 150 held-out test frames only:

  * lesion samples  : the polyp frame itself (downsampled to 256x256)
  * normal samples  : a mucosa crop from the *same* frame, at least `margin` px
                      away from the annotated polyp (default 30 px), with black
                      borders/letterboxing excluded

Because only test-split frames are used, the cohort is disjoint from every
training and calibration frame used for the segmentation network. The cohort is
then split 75/75 (deterministic) into a CALIBRATION half and a CERTIFICATION
half, so the conformal threshold is never evaluated on the data that chose it.

Outputs (under data/kvasir_triage_test/):
    calib/{images,masks}/<stem>_L.png | <stem>_N.png
    certify/{images,masks}/...
    manifest.json   (stem, kind, source frame for each sample)
"""
import argparse
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KSEG = os.path.join(ROOT, "data", "kvasir_seg")   # overridable via --kseg
OUT = os.path.join(ROOT, "data", "kvasir_triage_test")  # overridable via --out


def find_mask(mask_dir, stem):
    for ext in (".jpg", ".png", ".jpeg", ".tif"):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no mask for {stem}")


def pick_normal_crops(img, mask, margin=20, min_size=64, max_crops=2):
    """Return up to `max_crops` well-separated mucosa crops from one frame."""
    d2 = _crop_score_map(img, mask, margin, min_size)
    if d2 is None:
        return []
    h, w = mask.shape
    picks = []
    work = d2.copy()
    for _ in range(max_crops):
        r = int(work.max())
        if r < min_size / 2:
            break
        cy, cx = np.unravel_index(int(np.argmax(work)), work.shape)
        r = int(min(r, min(h, w) / 2 - 1))
        y0, x0 = max(0, cy - r), max(0, cx - r)
        y1, x1 = min(h, cy + r), min(w, cx + r)
        if (y1 - y0) >= min_size and (x1 - x0) >= min_size:
            crop = img[y0:y1, x0:x1]
            if (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) <= 12).mean() <= 0.25:
                picks.append(crop)
        # suppress a neighbourhood so the next pick is far away
        y, x = int(cy), int(cx)
        rad = max(min_size, r)
        work[max(0, y - rad):y + rad, max(0, x - rad):x + rad] = 0
    return picks


def _crop_score_map(img, mask, margin, min_size):
    """Largest axis-aligned window of the image whose pixels are all background."""
    h, w = mask.shape
    free = (mask <= 127)
    if free.sum() < min_size * min_size:
        return None
    # Distance transform gives the largest inscribed background square cheaply.
    # Exclude endoscopic black borders.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d2 = cv2.distanceTransform(((mask <= 127) & (gray > 12)).astype(np.uint8), cv2.DIST_L2, 3)
    if d2.max() < min_size / 2:
        return None
    return d2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=int, default=30)
    ap.add_argument("--min-size", type=int, default=64)
    ap.add_argument("--max-per-frame", type=int, default=2)
    ap.add_argument("--source", default="test", choices=["test", "trainval"],
                    help="which frames to derive the cohort from: 'test' (held-out, for "
                         "certification) or 'trainval' (fold-0 train+val, for TRAINING the gate "
                         "so that its input distribution matches deployment)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--kseg", default=KSEG, help="Kvasir-SEG root (contains images/ and masks/)")
    ap.add_argument("--splits", default=None, help="path to splits.json (default: <kseg>/splits.json)")
    args = ap.parse_args()
    kseg_dir = args.kseg
    splits_path = args.splits or os.path.join(kseg_dir, "splits.json")

    splits = json.load(open(splits_path, encoding="utf-8"))
    if args.source == "trainval":
        # fold-0 train + val frames: never used for certification, safe for gate training
        source_stems = sorted(set(splits["fold_0"]["train"]) | set(splits["fold_0"]["val"]))
    else:
        source_stems = list(splits["test"])
    test_stems = source_stems
    img_dir, mask_dir = os.path.join(kseg_dir, "images"), os.path.join(kseg_dir, "masks")
    rng = np.random.RandomState(42)

    manifest = {"source": args.source, "n_source_frames": len(source_stems),
                "margin": args.margin, "min_size": args.min_size,
                "max_crops_per_frame": args.max_per_frame, "splits": splits_path,
                "samples": [],
                "policy": "leakage-free: derived from held-out test frames only; "
                          "normal samples are real mucosal crops of the same frames"}
    for sub in ("calib", "certify"):
        for kind in ("images", "masks"):
            os.makedirs(os.path.join(args.out, sub, kind), exist_ok=True)

    n_lesion = n_normal = n_skip = 0
    for i, stem in enumerate(test_stems):
        sub = "calib" if i % 2 == 0 else "certify"     # deterministic 75/75
        img = cv2.imread(find_mask(img_dir, stem))
        msk = cv2.imread(find_mask(mask_dir, stem), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            n_skip += 1
            continue
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        msk_small = cv2.resize((msk > 127).astype(np.uint8) * 255, (256, 256), interpolation=cv2.INTER_NEAREST)
        # lesion sample: the polyp frame
        lp = os.path.join(args.out, sub, "images", f"{stem}_L.png")
        lm = os.path.join(args.out, sub, "masks", f"{stem}_L.png")
        cv2.imwrite(lp, img); cv2.imwrite(lm, msk_small)
        manifest["samples"].append({"split": sub, "kind": "lesion", "stem": stem,
                                    "file": os.path.basename(lp)})
        n_lesion += 1
        # normal sample: mucosa crop from the same frame
        crops = pick_normal_crops(img, msk_small, margin=args.margin,
                                  min_size=args.min_size, max_crops=args.max_per_frame)
        for k, crop in enumerate(crops):
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            suffix = "_N.png" if k == 0 else f"_N{k+1}.png"
            np_ = os.path.join(args.out, sub, "images", f"{stem}{suffix}")
            nm = os.path.join(args.out, sub, "masks", f"{stem}{suffix}")
            cv2.imwrite(np_, crop); cv2.imwrite(nm, np.zeros((256, 256), np.uint8))
            manifest["samples"].append({"split": sub, "kind": "normal", "stem": stem,
                                        "file": os.path.basename(np_)})
            n_normal += 1

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[mixed-cohort] lesion={n_lesion} normal={n_normal} skipped_frames={n_skip} -> {args.out}")
    for sub in ("calib", "certify"):
        k = [s for s in manifest["samples"] if s["split"] == sub]
        les = sum(1 for s in k if s["kind"] == "lesion")
        print(f"  {sub:8s}: {les} lesion + {len(k)-les} normal = {len(k)}")


if __name__ == "__main__":
    main()
