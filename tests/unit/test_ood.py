"""Unit tests for the Mahalanobis OOD detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from cancer_detection.data.transforms import get_val_transforms
from cancer_detection.serving.ood import EmbeddingOODDetector, _mahalanobis_batch


class _TinyBackbone(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_features = dim
        self.proj = nn.Linear(3 * 4 * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()


def _identity_detector(dim: int, threshold: float, n_reference: int = 100) -> EmbeddingOODDetector:
    """Build a detector whose PCA projection is the identity, for simple tests."""
    return EmbeddingOODDetector(
        pca_mean=np.zeros(dim),
        pca_components=np.eye(dim),
        mean=np.zeros(dim),
        precision=np.eye(dim),
        threshold=threshold,
        n_reference=n_reference,
    )


def test_mahalanobis_zero_at_mean() -> None:
    mean = np.zeros(4)
    precision = np.eye(4)
    feats = np.zeros((3, 4))
    dists = _mahalanobis_batch(feats, mean, precision)
    assert np.allclose(dists, 0.0)


def test_mahalanobis_is_actual_distance_not_squared() -> None:
    """A unit step along one axis under identity precision should score 1.0, not 1.0**2."""
    mean = np.zeros(3)
    precision = np.eye(3)
    feats = np.array([[3.0, 4.0, 0.0]])  # distance 5 under identity metric
    dists = _mahalanobis_batch(feats, mean, precision)
    assert dists[0] == pytest.approx(5.0)


def test_detector_flags_far_points() -> None:
    rng = np.random.default_rng(0)
    # 99th percentile of chi^2(8) is ~20, so distance ~= sqrt(20) ~= 4.47; set
    # threshold well below that so a near point stays under and a far point trips it.
    detector = _identity_detector(dim=8, threshold=3.0)
    near = rng.normal(0, 0.1, size=8)
    far = np.ones(8) * 10.0
    d_near, ood_near = detector.score(near)
    d_far, ood_far = detector.score(far)
    assert d_near < detector.threshold
    assert not ood_near
    assert d_far > detector.threshold
    assert ood_far


def test_detector_save_load_roundtrip(tmp_path) -> None:
    detector = _identity_detector(dim=4, threshold=3.5, n_reference=42)
    path = tmp_path / "ood.npz"
    detector.save(path)
    loaded = EmbeddingOODDetector.load(path)
    assert loaded.threshold == pytest.approx(3.5)
    assert loaded.n_reference == 42
    assert np.allclose(loaded.mean, detector.mean)
    assert loaded.feature_dim == detector.feature_dim


def test_load_rejects_mismatched_feature_dim(tmp_path) -> None:
    """A cache fit against a different backbone must not silently mis-score."""
    detector = _identity_detector(dim=8, threshold=3.5)
    path = tmp_path / "ood.npz"
    detector.save(path)
    with pytest.raises(ValueError, match="feature_dim"):
        EmbeddingOODDetector.load(path, expected_feature_dim=1792)


def test_fit_returns_none_without_data(tmp_path) -> None:
    model = _TinyModel()
    result = EmbeddingOODDetector.fit(
        model,
        tmp_path / "missing.csv",
        tmp_path / "missing_images",
        torch.device("cpu"),
        n_samples=10,
    )
    assert result is None


def _write_synthetic_images(
    img_dir, rng: np.random.Generator, n: int, offset: int = 0
) -> list[str]:
    img_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n):
        name = f"img_{offset + i:05d}"
        arr = rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{name}.jpg")
        names.append(name)
    return names


def test_fit_held_out_flag_rate_is_near_percentile(tmp_path) -> None:
    """Regression test for in-sample calibration: a held-out in-distribution sample
    should trip the OOD flag at close to the fitted percentile's complement, not far
    more often — which is what happens if the threshold is calibrated on the same
    data used to fit mean/precision.
    """
    img_dir = tmp_path / "images"
    rng = np.random.default_rng(0)

    # Fit on a larger in-distribution sample.
    fit_names = _write_synthetic_images(img_dir, rng, n=400)
    fit_df = pd.DataFrame({"image_name": fit_names})
    csv_path = tmp_path / "train.csv"
    fit_df.to_csv(csv_path, index=False)

    model = _TinyModel()
    detector = EmbeddingOODDetector.fit(
        model,
        csv_path,
        img_dir,
        torch.device("cpu"),
        n_samples=400,
        image_size=4,
        batch_size=32,
        percentile=99.0,
        n_components=4,
    )
    assert detector is not None

    # Score a *fresh* in-distribution sample never seen during fit or calibration,
    # through the same val transform pipeline fit() uses internally.
    held_out_names = _write_synthetic_images(img_dir, rng, n=300, offset=1000)
    transform = get_val_transforms(4)
    flagged = 0
    with torch.no_grad():
        for name in held_out_names:
            arr = np.array(Image.open(img_dir / f"{name}.jpg").convert("RGB"))
            tensor = transform(image=arr)["image"].unsqueeze(0)
            embedding = model.backbone(tensor).squeeze(0)
            _, is_ood = detector.score(embedding)
            flagged += int(is_ood)

    flag_rate = flagged / len(held_out_names)
    # Fit at the 99th percentile — allow generous slack for a small sample, but this
    # must not blow up to 10-20%+, which is the symptom of in-sample calibration.
    assert flag_rate <= 0.08
