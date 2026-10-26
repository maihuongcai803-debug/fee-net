r"""
Medical Image Segmentation Dataset Loader & Synthetic Benchmark Generator

Features:
1. MedicalSegDataset:
   - Robust PyTorch Dataset supporting arbitrary image/mask directories (PNG, JPG, TIFF, BMP).
   - If directory is missing or empty, automatically triggers realistic synthetic medical lesion generation.
2. Realistic Synthetic Lesion Generator:
   - Synthesizes diverse, clinically inspired medical images mimicking Ultrasound (US),
     Dermoscopy (Skin), and Computed Tomography (CT).
   - Generates organic, non-convex lesion contours with multi-harmonic polar perturbation.
   - Accurately models fuzzy partial-volume borders, tissue speckle/Gaussian noise,
     and low-contrast infiltrative boundaries.
   - Automatically populates 600 train, 150 val, 150 test samples into `d:\machine_learning\data\synthetic_medseg`.
3. get_dataloaders:
   - High-level factory function returning (train_loader, val_loader, test_loader)
     with matched medical data augmentations.
"""

import os
import glob
import random
from pathlib import Path
from typing import Optional, Tuple, List, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    from datasets.transforms import get_train_transforms, get_val_transforms, MedicalCompose
except ImportError:
    from transforms import get_train_transforms, get_val_transforms, MedicalCompose


# =====================================================================
# 1. Realistic Synthetic Medical Lesion Generator
# =====================================================================

def generate_organic_lesion_mask(img_size: int = 256,
                                 min_radius: float = 28.0,
                                 max_radius: float = 68.0) -> np.ndarray:
    """
    Generates an organic, clinically realistic non-convex lesion contour
    using multi-harmonic Fourier radial perturbation.
    """
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Random center with margin
    margin = int(img_size * 0.22)
    cx = random.randint(margin, img_size - margin)
    cy = random.randint(margin, img_size - margin)

    # Base radius and eccentricity
    base_r = random.uniform(min_radius, max_radius)
    eccentricity = random.uniform(0.70, 1.30)

    # Multi-harmonic polar modulation
    num_points = 180
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    harmonics = [
        (2, random.uniform(-0.15, 0.15)),
        (3, random.uniform(-0.12, 0.12)),
        (4, random.uniform(-0.08, 0.08)),
        (5, random.uniform(-0.05, 0.05)),
        (6, random.uniform(-0.03, 0.03))
    ]

    r = np.full_like(theta, base_r)
    for k, amp in harmonics:
        phase = random.uniform(0, 2 * np.pi)
        r += base_r * amp * np.cos(k * theta + phase)

    # Apply aspect ratio stretching
    xs = cx + r * np.cos(theta) * eccentricity
    ys = cy + r * np.sin(theta)

    # Clip coordinates to image boundary
    xs = np.clip(xs, 1, img_size - 2)
    ys = np.clip(ys, 1, img_size - 2)

    pts = np.stack([xs, ys], axis=-1).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Subtle morphological roughness
    if random.random() > 0.5:
        k_size = random.choice([3, 5])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        if random.random() > 0.5:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def synthesize_medical_sample(img_size: int = 256,
                              modality: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes a realistic single medical image and corresponding binary mask.

    Supported modalities:
    - 'ultrasound': Hypoechoic dark core, acoustic speckle texture, posterior acoustic shadowing.
    - 'dermoscopy': Irregular melanin pigmentation, brownish skin gradient, fine skin surface texture.
    - 'ct': Soft-tissue density, infiltrative ground-glass fuzzy margin, quantum Gaussian noise.
    """
    if modality is None:
        modality = random.choice(['ultrasound', 'dermoscopy', 'ct'])

    # 1. Generate exact ground truth mask
    mask = generate_organic_lesion_mask(img_size=img_size)

    # 2. Compute soft fuzzy transition boundary (partial volume effect & tissue infiltration)
    blur_k = random.choice([7, 11, 15])
    blur_sigma = random.uniform(2.0, 4.2)
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (blur_k, blur_k), blur_sigma)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    # Coordinate grids for anatomical gradients
    y_coords, x_coords = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    center_y, center_x = img_size / 2.0, img_size / 2.0

    if modality == 'ultrasound':
        # Ultrasound tissue background: acoustic line attenuation + speckle noise
        bg_base = 80.0 + 35.0 * np.sin(x_coords / 14.0) * np.cos(y_coords / 22.0)
        # Vertical attenuation gradient (deeper tissue has acoustic loss)
        attenuation = np.exp(-0.0015 * y_coords)
        bg = bg_base * attenuation

        # Multi-scale ultrasound speckle noise
        speckle = np.random.gamma(shape=4.0, scale=0.25, size=(img_size, img_size)).astype(np.float32)
        bg = bg * speckle

        # Lesion is hypoechoic (darker, lower reflection)
        lesion_base = 25.0 + 15.0 * np.random.randn(img_size, img_size).astype(np.float32)
        lesion_speckle = np.random.gamma(shape=2.5, scale=0.3, size=(img_size, img_size)).astype(np.float32)
        lesion = np.clip(lesion_base * lesion_speckle, 10.0, 70.0)

        # Composite image
        image_gray = bg * (1.0 - soft_mask) + lesion * soft_mask

        # Add posterior acoustic enhancement / shadow below lesion
        lesion_y_sum = np.sum(soft_mask, axis=0, keepdims=True)
        shadow_map = np.tile(lesion_y_sum, (img_size, 1)) * (y_coords > center_y) * 0.08
        image_gray = np.clip(image_gray - shadow_map + np.random.normal(0, 3.5, (img_size, img_size)), 0, 255)

        # Convert to 3-channel RGB representation
        image_rgb = cv2.merge([image_gray, image_gray, image_gray]).astype(np.uint8)

    elif modality == 'dermoscopy':
        # Dermoscopic background: pinkish/tan skin tones with illumination vignette
        dist_from_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        vignette = 1.0 - 0.28 * (dist_from_center / (img_size * 0.70)) ** 2
        vignette = np.clip(vignette, 0.45, 1.0)

        # Skin tone base
        skin_r = 210.0 + 15.0 * np.random.randn(img_size, img_size)
        skin_g = 165.0 + 12.0 * np.random.randn(img_size, img_size)
        skin_b = 145.0 + 12.0 * np.random.randn(img_size, img_size)

        # Lesion pigmentation: heterogeneous deep brown / dark melanin
        lesion_r = 75.0 + 25.0 * np.sin(x_coords / 8.0) * np.cos(y_coords / 8.0) + np.random.normal(0, 10, (img_size, img_size))
        lesion_g = 45.0 + 18.0 * np.cos(x_coords / 10.0) + np.random.normal(0, 8, (img_size, img_size))
        lesion_b = 35.0 + 15.0 * np.sin(y_coords / 10.0) + np.random.normal(0, 8, (img_size, img_size))

        # Composite channels
        img_r = (skin_r * (1.0 - soft_mask) + lesion_r * soft_mask) * vignette
        img_g = (skin_g * (1.0 - soft_mask) + lesion_g * soft_mask) * vignette
        img_b = (skin_b * (1.0 - soft_mask) + lesion_b * soft_mask) * vignette

        image_rgb = np.stack([img_r, img_g, img_b], axis=-1)
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    else:  # CT / Soft Tissue Radiography
        # Soft tissue parenchymal density (40 - 70 HU equivalent mapped to 110 - 150)
        bg = 135.0 + 20.0 * np.sin(x_coords / 30.0) + 15.0 * np.cos(y_coords / 35.0)

        # Infiltrative tumor core: either slightly hyperdense or hypodense with necrotic center
        tumor_core = 85.0 + 35.0 * np.random.randn(img_size, img_size).astype(np.float32)

        # Simulated ground-glass opacity halo
        halo_mask = cv2.GaussianBlur(soft_mask, (25, 25), 6.0)
        bg_modulated = bg * (1.0 - halo_mask) + (bg * 0.75 + 30.0) * halo_mask

        # Composite CT image
        image_gray = bg_modulated * (1.0 - soft_mask) + tumor_core * soft_mask

        # Quantum CT Poisson-Gaussian noise
        noise = np.random.normal(0, 6.0, (img_size, img_size)).astype(np.float32)
        image_gray = np.clip(image_gray + noise, 0, 255)

        image_rgb = cv2.merge([image_gray, image_gray, image_gray]).astype(np.uint8)

    return image_rgb, mask


def generate_synthetic_benchmark(root_dir: str = "d:/machine_learning/data/synthetic_medseg",
                                 num_train: int = 600,
                                 num_val: int = 150,
                                 num_test: int = 150,
                                 img_size: int = 256,
                                 seed: int = 42) -> None:
    """
    Generates a complete reproducible synthetic medical segmentation benchmark dataset
    with realistic lesion images and ground truth masks.
    """
    random.seed(seed)
    np.random.seed(seed)

    splits = [
        ('train', num_train),
        ('val', num_val),
        ('test', num_test)
    ]

    base_path = Path(root_dir)
    print(f"[Dataset Generator] Generating synthetic benchmark at: {base_path.resolve()} ...")

    total_samples = num_train + num_val + num_test
    generated_count = 0

    for split_name, count in splits:
        img_dir = base_path / split_name / "images"
        msk_dir = base_path / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img_rgb, mask = synthesize_medical_sample(img_size=img_size)

            sample_id = f"{i + 1:04d}"
            img_path = img_dir / f"case_{sample_id}.png"
            msk_path = msk_dir / f"case_{sample_id}.png"

            # Save as standard PNG
            # OpenCV writes BGR, convert RGB to BGR before saving
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), img_bgr)
            cv2.imwrite(str(msk_path), mask)

            generated_count += 1

        print(f"  -> Generated {count} {split_name} samples in {img_dir.parent}")

    print(f"[Dataset Generator] Successfully generated all {total_samples} synthetic samples!")


# =====================================================================
# 2. PyTorch MedicalSegDataset
# =====================================================================

class MedicalSegDataset(Dataset):
    """
    Robust PyTorch Dataset for medical image segmentation.

    Args:
        img_dir: Directory containing image files. If None or empty,
                 automatically loads/generates synthetic benchmark data.
        mask_dir: Directory containing ground truth mask files.
        split: One of 'train', 'val', or 'test'.
        img_size: Expected spatial resolution (H, W).
        transform: Augmentation pipeline callable (image, mask) -> (image, mask).
        in_channels: 3 for RGB, 1 for single-channel grayscale.
    """
    def __init__(self,
                 img_dir: Optional[Union[str, Path]] = None,
                 mask_dir: Optional[Union[str, Path]] = None,
                 split: str = "train",
                 img_size: int = 256,
                 transform: Optional[MedicalCompose] = None,
                 in_channels: int = 3):
        super().__init__()
        self.split = split.lower()
        self.img_size = img_size
        self.transform = transform
        self.in_channels = in_channels

        default_synthetic_root = Path("d:/machine_learning/data/synthetic_medseg")

        # Resolve paths
        if img_dir is None:
            img_dir = default_synthetic_root / self.split / "images"
            mask_dir = default_synthetic_root / self.split / "masks"
        else:
            img_dir = Path(img_dir)
            if mask_dir is None:
                mask_dir = img_dir.parent / "masks"
            else:
                mask_dir = Path(mask_dir)

        # Check if directory exists and has files; if not, trigger synthetic generator
        valid_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        found_files = []
        if img_dir.exists():
            for ext in valid_extensions:
                found_files.extend(glob.glob(str(img_dir / ext)))

        if len(found_files) == 0:
            print(f"[MedicalSegDataset] Image directory {img_dir} is empty or missing. Triggering synthetic benchmark generation...")
            generate_synthetic_benchmark(root_dir=str(default_synthetic_root), img_size=self.img_size)
            # Re-point to generated directory
            img_dir = default_synthetic_root / self.split / "images"
            mask_dir = default_synthetic_root / self.split / "masks"
            for ext in valid_extensions:
                found_files.extend(glob.glob(str(img_dir / ext)))

        found_files.sort()
        self.image_paths: List[Path] = [Path(p) for p in found_files]
        self.mask_paths: List[Path] = []

        # Match masks
        for img_path in self.image_paths:
            stem = img_path.stem
            candidates = [
                mask_dir / f"{stem}{img_path.suffix}",
                mask_dir / f"{stem}.png",
                mask_dir / f"{stem}_mask{img_path.suffix}",
                mask_dir / f"{stem}_mask.png",
                mask_dir / f"{stem}_seg{img_path.suffix}",
                mask_dir / f"{stem}_seg.png",
            ]
            matched_mask = None
            for c in candidates:
                if c.exists():
                    matched_mask = c
                    break

            if matched_mask is None:
                raise FileNotFoundError(f"Could not find matching mask for image: {img_path} in {mask_dir}")
            self.mask_paths.append(matched_mask)

        print(f"[MedicalSegDataset] Initialized '{self.split}' dataset: {len(self.image_paths)} image-mask pairs loaded.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.image_paths[idx])
        msk_path = str(self.mask_paths[idx])

        # Load image
        if self.in_channels == 3:
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Failed to read image at: {img_path}")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to read image at: {img_path}")

        # Load mask
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask at: {msk_path}")

        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # Fallback default conversion if no transform provided
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            if image.ndim == 2:
                image = np.expand_dims(image, 0)
            elif image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))

            image = torch.from_numpy(image.astype(np.float32) / 255.0).float()
            mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0).float()

        return image, mask


# =====================================================================
# 3. DataLoaders Factory
# =====================================================================

def get_dataloaders(root_dir: Optional[str] = None,
                    batch_size: int = 8,
                    img_size: int = 256,
                    num_workers: int = 0,
                    in_channels: int = 3) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns (train_loader, val_loader, test_loader)
    with standard medical image augmentations.

    Args:
        root_dir: Root directory of dataset containing train/, val/, test/ folders.
                  If None, defaults to `d:/machine_learning/data/synthetic_medseg`.
        batch_size: Mini-batch size for training and evaluation.
        img_size: Image height and width.
        num_workers: Number of subprocesses for PyTorch DataLoader (0 for Windows compatibility).
        in_channels: 3 for RGB, 1 for grayscale.
    """
    train_transform = get_train_transforms(img_size=img_size)
    val_transform = get_val_transforms(img_size=img_size)

    train_dataset = MedicalSegDataset(
        img_dir=Path(root_dir) / "train" / "images" if root_dir else None,
        split="train",
        img_size=img_size,
        transform=train_transform,
        in_channels=in_channels
    )

    val_dataset = MedicalSegDataset(
        img_dir=Path(root_dir) / "val" / "images" if root_dir else None,
        split="val",
        img_size=img_size,
        transform=val_transform,
        in_channels=in_channels
    )

    test_dataset = MedicalSegDataset(
        img_dir=Path(root_dir) / "test" / "images" if root_dir else None,
        split="test",
        img_size=img_size,
        transform=val_transform,
        in_channels=in_channels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("=" * 60)
    print("Self-test: MedicalSegDataset & Synthetic Benchmark")
    print("=" * 60)

    # 1. Instantiate DataLoaders (triggers synthetic generation if not present)
    batch_size = 8
    img_size = 256
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=None,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=0,
        in_channels=3
    )

    print(f"\nDataLoader Summary:")
    print(f"  Train batches: {len(train_loader)} (Total: {len(train_loader.dataset)} samples)")
    print(f"  Val batches:   {len(val_loader)} (Total: {len(val_loader.dataset)} samples)")
    print(f"  Test batches:  {len(test_loader)} (Total: {len(test_loader.dataset)} samples)")

    # Assert dataset counts
    assert len(train_loader.dataset) == 600, f"Expected 600 train samples, got {len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 150, f"Expected 150 val samples, got {len(val_loader.dataset)}"
    assert len(test_loader.dataset) == 150, f"Expected 150 test samples, got {len(test_loader.dataset)}"

    # 2. Fetch one batch from train_loader and verify shapes and dtypes
    images, masks = next(iter(train_loader))
    print(f"\nBatch inspection:")
    print(f"  Images batch shape: {images.shape}, dtype: {images.dtype}, Min: {images.min():.2f}, Max: {images.max():.2f}")
    print(f"  Masks batch shape:  {masks.shape}, dtype: {masks.dtype}, Unique values: {torch.unique(masks).tolist()}")

    assert images.shape == (batch_size, 3, img_size, img_size), f"Image batch shape mismatch: {images.shape}"
    assert masks.shape == (batch_size, 1, img_size, img_size), f"Mask batch shape mismatch: {masks.shape}"
    assert set(torch.unique(masks).tolist()).issubset({0.0, 1.0}), "Mask tensor contains non-binary values!"

    # 3. Fetch one batch from val_loader and verify
    val_images, val_masks = next(iter(val_loader))
    assert val_images.shape == (batch_size, 3, img_size, img_size)
    assert val_masks.shape == (batch_size, 1, img_size, img_size)
    print("Validation batch inspection passed.")

    print("=" * 60)
    print("ALL DATASET & DATALOADER SELF-TESTS PASSED!")
    print("=" * 60)
