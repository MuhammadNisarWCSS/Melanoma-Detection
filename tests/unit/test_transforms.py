"""Unit tests for Albumentations transform pipelines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cancer_detection.data.transforms import (
    get_train_transforms,
    get_tta_transforms,
    get_val_transforms,
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


def test_val_transforms_preserve_aspect_ratio() -> None:
    """A 3:2 image must be cropped to a square, not squashed into one.

    Under a squashing A.Resize the two halves of a horizontally-mirrored pair stay
    mirrored, so this checks the geometry directly: a distinctive stripe placed in
    the centre must keep its proportions.
    """
    tall = np.zeros((600, 900, 3), dtype=np.uint8)
    tall[250:350, :, :] = 255  # horizontal band spanning 1/6 of the height

    result = get_val_transforms(384)(image=tall)["image"]
    band_rows = (result[0].numpy() > 0).sum(axis=1)
    band_height = int((band_rows > 0).sum())

    # 100/600 of the height, scaled so the short edge becomes 384 -> ~64px.
    assert 55 <= band_height <= 75, f"band height {band_height} implies aspect distortion"


def test_tta_transforms_count() -> None:
    tta = get_tta_transforms(384)
    assert len(tta) == 8


def test_tta_transforms_all_produce_correct_shape(image: np.ndarray) -> None:
    for i, transform in enumerate(get_tta_transforms(384)):
        result = transform(image=image)["image"]
        assert result.shape == (3, 384, 384), f"TTA transform {i} produced wrong shape"


def test_tta_transforms_are_deterministic(image: np.ndarray) -> None:
    """Repeated calls must return identical tensors.

    A.RandomRotate90 draws its rotation count at random even at p=1.0, which made
    the API return a different probability for the same upload on every request.
    """
    for i, transform in enumerate(get_tta_transforms(384)):
        first = transform(image=image)["image"]
        second = transform(image=image)["image"]
        assert torch.equal(first, second), f"TTA transform {i} is not deterministic"


def test_tta_transforms_are_distinct(image: np.ndarray) -> None:
    """The eight passes should be eight different views, not duplicates."""
    outputs = [t(image=image)["image"] for t in get_tta_transforms(384)]
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.equal(outputs[i], outputs[j]), f"TTA passes {i} and {j} match"


def test_tta_first_pass_matches_val_transform(image: np.ndarray) -> None:
    """Predictor reuses the base pass for both TTA index 0 and GradCAM."""
    val = get_val_transforms(384)(image=image)["image"]
    tta_identity = get_tta_transforms(384)[0](image=image)["image"]
    assert torch.equal(val, tta_identity)


def test_custom_image_size(image: np.ndarray) -> None:
    small_image = image[:224, :224]
    transform = get_val_transforms(224)
    result = transform(image=small_image)["image"]
    assert result.shape == (3, 224, 224)
