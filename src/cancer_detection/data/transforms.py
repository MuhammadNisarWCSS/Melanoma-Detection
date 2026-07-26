from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — appropriate since backbone is pretrained on ImageNet
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 384) -> A.Compose:
    """Dermoscopy-aware augmentation pipeline for training.

    Design choices:
    - RandomResizedCrop: simulates varying zoom levels in clinical imaging
    - Flips + RandomRotate90: lesions have no canonical orientation
    - HueSaturationValue: models color constancy variation across imaging devices
    - CoarseDropout: simulates hair and ruler artifacts common in dermoscopy
    - GaussNoise: regularizes against sensor noise
    """
    return A.Compose(
        [
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=image_size // 16,
                max_width=image_size // 16,
                min_holes=1,
                fill_value=0,
                p=0.3,
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 384) -> A.Compose:
    """Deterministic transforms for validation and test — no stochastic ops."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(image_size: int = 384) -> list[A.Compose]:
    """Eight deterministic TTA variants (flips × rotations)."""
    base = [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]
    augment_sets: list[list[A.BasicTransform]] = [
        [],
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        [A.RandomRotate90(p=1.0)],
        [A.RandomRotate90(p=1.0), A.RandomRotate90(p=1.0)],
        [A.RandomRotate90(p=1.0), A.RandomRotate90(p=1.0), A.RandomRotate90(p=1.0)],
        [A.HorizontalFlip(p=1.0), A.RandomRotate90(p=1.0)],
        [A.VerticalFlip(p=1.0), A.RandomRotate90(p=1.0)],
    ]
    return [A.Compose(augs + base) for augs in augment_sets]
