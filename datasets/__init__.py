"""
Datasets package for medical image segmentation.
Exposes MedicalSegDataset, get_dataloaders, and transform utilities.
"""

from .dataset_loader import MedicalSegDataset, get_dataloaders, generate_synthetic_benchmark
from .real_medical_loader import (
    KvasirSegDataset,
    get_kvasir_dataloaders,
    download_or_generate_kvasir_seg,
    create_kvasir_stratified_splits,
    KvasirDataLoaderBundle
)
from .transforms import (
    MedicalCompose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    ContrastAdjust,
    Resize,
    ToTensor,
    Normalize,
    get_train_transforms,
    get_val_transforms
)

__all__ = [
    "MedicalSegDataset",
    "get_dataloaders",
    "generate_synthetic_benchmark",
    "KvasirSegDataset",
    "get_kvasir_dataloaders",
    "download_or_generate_kvasir_seg",
    "create_kvasir_stratified_splits",
    "KvasirDataLoaderBundle",
    "MedicalCompose",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "ColorJitter",
    "ContrastAdjust",
    "Resize",
    "ToTensor",
    "Normalize",
    "get_train_transforms",
    "get_val_transforms"
]

