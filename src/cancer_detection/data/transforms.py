from __future__ import annotations

from collections.abc import Callable
from typing import Any

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — appropriate since backbone is pretrained on ImageNet
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _rot90(k: int) -> Callable[..., np.ndarray]:
    """Build a fixed k*90-degree rotation for use inside A.Lambda.

    A.RandomRotate90 draws k at random even at p=1.0, which makes "deterministic"
    TTA non-reproducible. np.rot90 returns a negative-stride view, so the result is
    copied before it reaches ToTensorV2.
    """

    def apply(image: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(image, k=k))

    return apply


def _to_square(image_size: int) -> list[A.BasicTransform]:
    """Scale the shorter edge to image_size, then centre-crop to a square.

    Deliberately not A.Resize(image_size, image_size): ISIC images are almost all
    3:2, so a direct resize squashes every one of them horizontally by 1.5x. The
    training pipeline uses RandomResizedCrop, which preserves aspect ratio within
    its ratio bounds, so a squashing validation transform puts evaluation and
    serving in a different geometry than training.
    """
    return [
        A.SmallestMaxSize(max_size=image_size, interpolation=cv2.INTER_AREA),
        A.CenterCrop(height=image_size, width=image_size),
    ]


def _normalize() -> list[A.BasicTransform]:
    return [A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()]


def get_train_transforms(image_size: int = 384) -> A.Compose:
    """Dermoscopy-aware augmentation pipeline for training.

    Design choices:
    - RandomResizedCrop: simulates varying zoom levels in clinical imaging
    - Flips + rotations: lesions have no canonical orientation
    - HueSaturationValue / RandomBrightnessContrast: colour constancy varies by device
    - CoarseDropout: simulates hair and ruler artifacts common in dermoscopy
    - ImageCompression / Downscale / blur: the training cache is a single BICUBIC
      downsample re-encoded at JPEG quality 92, so every training image carries an
      identical resampling fingerprint. Varying it is regularisation against a cue
      that has nothing to do with the lesion.
    - GaussNoise: regularizes against sensor noise
    """
    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(image_size // 32, image_size // 16),
                hole_width_range=(image_size // 32, image_size // 16),
                fill=0,
                p=0.3,
            ),
            A.ImageCompression(quality_range=(40, 95), p=0.4),
            A.Downscale(scale_range=(0.35, 0.9), p=0.25),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ],
                p=0.2,
            ),
            A.GaussNoise(p=0.2),
            *_normalize(),
        ]
    )


def get_val_transforms(image_size: int = 384) -> A.Compose:
    """Deterministic transforms for validation, test and serving — no stochastic ops."""
    return A.Compose([*_to_square(image_size), *_normalize()])


def get_tta_transforms(image_size: int = 384) -> list[A.Compose]:
    """The eight dihedral symmetries of a square, applied deterministically.

    Index 0 is the identity and is therefore equivalent to get_val_transforms, so
    callers can reuse a single base pass for both TTA and GradCAM.
    """
    geometry = _to_square(image_size)
    augment_sets: list[list[A.BasicTransform]] = [
        [],
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        [A.Lambda(image=_rot90(1), name="rot90")],
        [A.Lambda(image=_rot90(2), name="rot180")],
        [A.Lambda(image=_rot90(3), name="rot270")],
        [A.HorizontalFlip(p=1.0), A.Lambda(image=_rot90(1), name="rot90")],
        [A.VerticalFlip(p=1.0), A.Lambda(image=_rot90(1), name="rot90")],
    ]
    return [A.Compose(augs + geometry + _normalize()) for augs in augment_sets]
