"""Unit tests for Albumentations transform pipelines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cancer_detection.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_tta_transforms,
)


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (384, 384, 3), dtype=np.uint8)


def test_val_transforms_output_shape(image: np.ndarray) -> None:
    transform = get_val_transforms(384)
    result = transform(image=image)["image"]
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 384, 384)


def test_val_transforms_output_dtype(image: np.ndarray) -> None:
    transform = get_val_transforms(384)
    result = transform(image=image)["image"]
    assert result.dtype == torch.float32


def test_val_transforms_is_deterministic(image: np.ndarray) -> None:
    transform = get_val_transforms(384)
    r1 = transform(image=image)["image"]
    r2 = transform(image=image)["image"]
    assert torch.allclose(r1, r2)


def test_train_transforms_output_shape(image: np.ndarray) -> None:
    transform = get_train_transforms(384)
    result = transform(image=image)["image"]
    assert result.shape == (3, 384, 384)


def test_train_transforms_normalized(image: np.ndarray) -> None:
    """After Normalize, values should be in roughly [-3, 3] (ImageNet stats)."""
    transform = get_val_transforms(384)
    result = transform(image=image)["image"]
    assert result.min() > -5.0
    assert result.max() < 5.0


def test_tta_transforms_count() -> None:
    tta = get_tta_transforms(384)
    assert len(tta) == 8


def test_tta_transforms_all_produce_correct_shape(image: np.ndarray) -> None:
    for i, transform in enumerate(get_tta_transforms(384)):
        result = transform(image=image)["image"]
        assert result.shape == (3, 384, 384), f"TTA transform {i} produced wrong shape"


def test_custom_image_size(image: np.ndarray) -> None:
    small_image = image[:224, :224]
    transform = get_val_transforms(224)
    result = transform(image=small_image)["image"]
    assert result.shape == (3, 224, 224)
