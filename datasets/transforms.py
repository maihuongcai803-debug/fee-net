"""
Medical Image Segmentation Data Augmentation Pipeline
Supports synchronized spatial transformations on (image, mask) pairs and
intensity transformations on images only.

Works seamlessly with:
- PyTorch Tensors (C, H, W) or (B, C, H, W)
- NumPy arrays (H, W, C), (H, W), or (C, H, W)
- PIL Images
"""

import random
from typing import Sequence, Tuple, Union, Optional
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image


def _ensure_tensor_or_pil(data, is_mask: bool = False):
    """Convert numpy array to tensor if needed for functional transforms."""
    if isinstance(data, np.ndarray):
        arr = data.copy()
        if is_mask:
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            tensor = torch.from_numpy(arr)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 3 and tensor.shape[2] == 1:
                tensor = tensor.permute(2, 0, 1)
            return tensor, "numpy"
        else:
            if arr.ndim == 2:
                tensor = torch.from_numpy(arr).unsqueeze(0)
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                tensor = torch.from_numpy(arr)
            return tensor, "numpy"
    elif isinstance(data, torch.Tensor):
        return data, "tensor"
    elif isinstance(data, Image.Image):
        return data, "pil"
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _restore_type(transformed, orig_type: str, is_mask: bool = False):
    """Restore transformed object back to original format if it was numpy."""
    if orig_type == "numpy":
        if isinstance(transformed, torch.Tensor):
            arr = transformed.cpu().numpy()
            if is_mask:
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                return arr
            else:
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                return arr
        elif isinstance(transformed, Image.Image):
            return np.array(transformed)
    return transformed


class RandomHorizontalFlip:
    """Synchronized random horizontal flip for medical image and segmentation mask."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            img_obj, img_type = _ensure_tensor_or_pil(image, is_mask=False)
            msk_obj, msk_type = _ensure_tensor_or_pil(mask, is_mask=True)

            img_flipped = TF.hflip(img_obj)
            msk_flipped = TF.hflip(msk_obj)

            return (
                _restore_type(img_flipped, img_type, is_mask=False),
                _restore_type(msk_flipped, msk_type, is_mask=True)
            )
        return image, mask


class RandomVerticalFlip:
    """Synchronized random vertical flip for medical image and segmentation mask."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            img_obj, img_type = _ensure_tensor_or_pil(image, is_mask=False)
            msk_obj, msk_type = _ensure_tensor_or_pil(mask, is_mask=True)

            img_flipped = TF.vflip(img_obj)
            msk_flipped = TF.vflip(msk_obj)

            return (
                _restore_type(img_flipped, img_type, is_mask=False),
                _restore_type(msk_flipped, msk_type, is_mask=True)
            )
        return image, mask


class RandomRotation:
    """
    Synchronized random rotation.
    Mask uses NEAREST NEIGHBOR interpolation to preserve discrete class labels.
    Image uses BILINEAR interpolation.
    """
    def __init__(self, degrees: Union[float, Sequence[float]] = 15.0, p: float = 0.5):
        if isinstance(degrees, (int, float)):
            self.degrees = (-float(degrees), float(degrees))
        else:
            self.degrees = (float(degrees[0]), float(degrees[1]))
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])

            img_obj, img_type = _ensure_tensor_or_pil(image, is_mask=False)
            msk_obj, msk_type = _ensure_tensor_or_pil(mask, is_mask=True)

            img_rot = TF.rotate(img_obj, angle, interpolation=InterpolationMode.BILINEAR)
            msk_rot = TF.rotate(msk_obj, angle, interpolation=InterpolationMode.NEAREST)

            return (
                _restore_type(img_rot, img_type, is_mask=False),
                _restore_type(msk_rot, msk_type, is_mask=True)
            )
        return image, mask


class ColorJitter:
    """
    Randomly alters brightness, contrast, and saturation of the image.
    Mask remains completely untouched.
    """
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.1, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            img_obj, img_type = _ensure_tensor_or_pil(image, is_mask=False)

            # Apply brightness
            if self.brightness > 0:
                b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                img_obj = TF.adjust_brightness(img_obj, max(0.0, b_factor))

            # Apply contrast
            if self.contrast > 0:
                c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                img_obj = TF.adjust_contrast(img_obj, max(0.0, c_factor))

            # Apply saturation if 3 channels
            if self.saturation > 0:
                s_factor = 1.0 + random.uniform(-self.saturation, self.saturation)
                # Saturation works on color images
                try:
                    img_obj = TF.adjust_saturation(img_obj, max(0.0, s_factor))
                except Exception:
                    pass

            image = _restore_type(img_obj, img_type, is_mask=False)
        return image, mask


class ContrastAdjust:
    """
    Explicit medical contrast adjustment (useful for CT windowing / ultrasound gain shifts).
    Mask remains untouched.
    """
    def __init__(self, contrast_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            img_obj, img_type = _ensure_tensor_or_pil(image, is_mask=False)
            factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            img_obj = TF.adjust_contrast(img_obj, factor)
            image = _restore_type(img_obj, img_type, is_mask=False)
        return image, mask


class RandomEndoscopicGlare:
    """
    Simulates optical specular reflections and moisture glare on mucosal surfaces,
    mimicking endoscope xenon/LED light reflections on wet gastrointestinal tissue.
    Generates bright saturated highlight spots with Gaussian halo falloff.

    Transforms the image while keeping the ground-truth segmentation mask intact,
    training the network to be illumination- and glare-invariant.
    """
    def __init__(self, p: float = 0.3, num_spots_range: Tuple[int, int] = (1, 4)):
        self.p = p
        self.num_spots_range = num_spots_range

    def __call__(self, image, mask):
        if random.random() < self.p:
            img_obj, img_type = _ensure_tensor_or_pil(image, is_mask=False)

            # Convert to numpy uint8 (H, W, C)
            if isinstance(img_obj, torch.Tensor):
                img_np = (img_obj.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                was_tensor_float = img_obj.is_floating_point()
            elif isinstance(img_obj, Image.Image):
                img_np = np.array(img_obj)
                was_tensor_float = False
            else:
                img_np = np.array(img_obj, dtype=np.uint8)
                was_tensor_float = False

            h, w = img_np.shape[:2]
            glare_layer = np.zeros((h, w), dtype=np.float32)
            num_spots = random.randint(*self.num_spots_range)
            for _ in range(num_spots):
                cx = random.randint(int(w * 0.15), int(w * 0.85))
                cy = random.randint(int(h * 0.15), int(h * 0.85))
                rx = random.randint(max(4, int(w * 0.04)), max(8, int(w * 0.14)))
                ry = random.randint(max(4, int(h * 0.03)), max(8, int(h * 0.10)))
                angle = random.randint(0, 180)

                spot_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(spot_mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
                k_size = max(5, int(min(rx, ry) * 0.8) | 1)
                spot_blurred = cv2.GaussianBlur(spot_mask.astype(np.float32), (k_size, k_size), 0) / 255.0
                glare_layer = np.maximum(glare_layer, spot_blurred)

            out_img = img_np.astype(np.float32).copy()
            if out_img.ndim == 3:
                glare_layer_3c = np.repeat(glare_layer[:, :, np.newaxis], out_img.shape[2], axis=2)
            else:
                glare_layer_3c = glare_layer
            out_img = out_img * (1.0 - 0.75 * glare_layer_3c) + 255.0 * (0.85 * glare_layer_3c)
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)

            if isinstance(img_obj, torch.Tensor):
                res_tensor = torch.from_numpy(out_img).permute(2, 0, 1)
                if was_tensor_float:
                    res_tensor = res_tensor.float() / 255.0
                res_img = res_tensor
            elif isinstance(img_obj, Image.Image):
                res_img = Image.fromarray(out_img)
            else:
                res_img = out_img

            return _restore_type(res_img, img_type, is_mask=False), mask
        return image, mask


class Resize:
    """
    Resize image (bilinear) and mask (nearest-neighbor) to target dimensions.
    """
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, image, mask):
        img_obj, img_type = _ensure_tensor_or_pil(image, is_mask=False)
        msk_obj, msk_type = _ensure_tensor_or_pil(mask, is_mask=True)

        img_res = TF.resize(img_obj, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        msk_res = TF.resize(msk_obj, self.size, interpolation=InterpolationMode.NEAREST)

        return (
            _restore_type(img_res, img_type, is_mask=False),
            _restore_type(msk_res, msk_type, is_mask=True)
        )


class ToTensor:
    """
    Converts (image, mask) into standard PyTorch Tensors.
    - Image -> torch.FloatTensor of shape (C, H, W) scaled to [0.0, 1.0].
    - Mask  -> torch.FloatTensor of shape (1, H, W) binarized to {0.0, 1.0}.
    """
    def __call__(self, image, mask):
        # Image conversion
        if isinstance(image, Image.Image):
            img_t = TF.to_tensor(image)  # (C, H, W), [0, 1]
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 0)
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))
            if arr.max() > 1.0:
                arr /= 255.0
            img_t = torch.from_numpy(arr).float()
        elif isinstance(image, torch.Tensor):
            img_t = image.float()
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)
            elif img_t.dim() == 3 and img_t.shape[-1] in (1, 3, 4) and img_t.shape[0] not in (1, 3, 4):
                img_t = img_t.permute(2, 0, 1)
            if img_t.max() > 1.0:
                img_t /= 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Mask conversion
        if isinstance(mask, Image.Image):
            mask_arr = np.array(mask, dtype=np.float32)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            if mask_arr.max() > 1.0:
                mask_arr /= 255.0
            mask_t = torch.from_numpy((mask_arr > 0.5).astype(np.float32)).unsqueeze(0)
        elif isinstance(mask, np.ndarray):
            mask_arr = mask.astype(np.float32)
            if mask_arr.ndim == 3:
                if mask_arr.shape[-1] in (1, 3, 4):
                    mask_arr = mask_arr[:, :, 0]
                elif mask_arr.shape[0] == 1:
                    mask_arr = mask_arr[0]
            if mask_arr.max() > 1.0:
                mask_arr /= 255.0
            mask_t = torch.from_numpy((mask_arr > 0.5).astype(np.float32)).unsqueeze(0)
        elif isinstance(mask, torch.Tensor):
            mask_t = mask.float()
            if mask_t.dim() == 2:
                mask_t = mask_t.unsqueeze(0)
            elif mask_t.dim() == 3 and mask_t.shape[-1] == 1:
                mask_t = mask_t.permute(2, 0, 1)
            if mask_t.max() > 1.0:
                mask_t /= 255.0
            mask_t = (mask_t > 0.5).float()
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        return img_t, mask_t


class Normalize:
    """
    Standard channel-wise normalization on Image tensor: (img - mean) / std.
    Mask remains untouched.
    """
    def __init__(self, mean: Sequence[float] = (0.485, 0.456, 0.406), std: Sequence[float] = (0.229, 0.224, 0.225)):
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Normalize requires image to be a torch.Tensor. Place ToTensor() before Normalize().")

        num_channels = image.shape[0]
        if num_channels == 1:
            mean = [self.mean[0]]
            std = [self.std[0]]
        else:
            mean = self.mean[:num_channels]
            std = self.std[:num_channels]

        image_norm = TF.normalize(image, mean=mean, std=std)
        return image_norm, mask


class MedicalCompose:
    """
    Chains multiple transformations sequentially on (image, mask).
    """
    def __init__(self, transforms: Sequence):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_train_transforms(img_size: int = 256,
                        mean: Sequence[float] = (0.485, 0.456, 0.406),
                        std: Sequence[float] = (0.229, 0.224, 0.225),
                        use_glare_aug: bool = True) -> MedicalCompose:
    """
    Standard training transformation pipeline for medical image segmentation.
    Includes spatial resizing, flipping, rotation, physics-informed endoscopic glare,
    contrast/color perturbations, tensor conversion, and normalization.
    """
    transforms_list = [
        Resize((img_size, img_size)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=20.0, p=0.5),
    ]
    if use_glare_aug:
        transforms_list.append(RandomEndoscopicGlare(p=0.3))
    transforms_list.extend([
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
        ContrastAdjust(contrast_range=(0.85, 1.15), p=0.5),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return MedicalCompose(transforms_list)


def get_val_transforms(img_size: int = 256,
                      mean: Sequence[float] = (0.485, 0.456, 0.406),
                      std: Sequence[float] = (0.229, 0.224, 0.225)) -> MedicalCompose:
    """
    Deterministic validation/test pipeline: Resize -> ToTensor -> Normalize.
    """
    return MedicalCompose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])


if __name__ == '__main__':
    print("=" * 60)
    print("Self-test: Medical Transforms Pipeline")
    print("=" * 60)

    # 1. Test with NumPy arrays
    np_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    np_mask = np.zeros((300, 300), dtype=np.uint8)
    np_mask[100:200, 100:200] = 255

    pipeline_train = get_train_transforms(img_size=256)
    out_img, out_mask = pipeline_train(np_img, np_mask)

    print(f"NumPy Input Pipeline Output:")
    print(f"  Image Type: {type(out_img)}, Shape: {out_img.shape}, Min: {out_img.min():.2f}, Max: {out_img.max():.2f}")
    print(f"  Mask Type:  {type(out_mask)}, Shape: {out_mask.shape}, Unique Values: {torch.unique(out_mask).tolist()}")

    assert out_img.shape == (3, 256, 256), f"Wrong output image shape: {out_img.shape}"
    assert out_mask.shape == (1, 256, 256), f"Wrong output mask shape: {out_mask.shape}"
    assert set(torch.unique(out_mask).tolist()).issubset({0.0, 1.0}), "Mask contains non-binary values!"

    # 2. Test with PIL Images
    pil_img = Image.fromarray(np_img)
    pil_mask = Image.fromarray(np_mask)
    out_img_pil, out_mask_pil = pipeline_train(pil_img, pil_mask)
    assert out_img_pil.shape == (3, 256, 256)
    assert out_mask_pil.shape == (1, 256, 256)
    print("PIL Input Pipeline passed successfully.")

    # 3. Test with PyTorch Tensors
    t_img = torch.rand(3, 256, 256)
    t_mask = (torch.rand(1, 256, 256) > 0.5).float()
    rot = RandomRotation(degrees=30, p=1.0)
    rot_img, rot_mask = rot(t_img, t_mask)
    assert rot_img.shape == (3, 256, 256)
    assert rot_mask.shape == (1, 256, 256)
    print("PyTorch Tensor Rotation passed successfully.")

    # 4. Test Validation Pipeline
    pipeline_val = get_val_transforms(img_size=256)
    val_img, val_mask = pipeline_val(np_img, np_mask)
    assert val_img.shape == (3, 256, 256)
    assert val_mask.shape == (1, 256, 256)
    print("Validation Pipeline passed successfully.")

    print("=" * 60)
    print("ALL TRANSFORMS SELF-TESTS PASSED!")
    print("=" * 60)
