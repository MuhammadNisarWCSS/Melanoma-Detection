"""Shared pytest fixtures — synthetic data only, no real ISIC images required.

All fixtures use temporary directories and randomly-generated data so the
entire test suite runs in CI without a Kaggle download.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_image_array() -> np.ndarray:
    """Random 384×384 uint8 RGB numpy array (mimics a dermoscopy image)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (384, 384, 3), dtype=np.uint8)


@pytest.fixture
def small_image_array() -> np.ndarray:
    """Tiny 64×64 image for fast unit tests."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_bytes(sample_image_array: np.ndarray) -> bytes:
    """JPEG-encoded bytes of sample_image_array."""
    buf = io.BytesIO()
    Image.fromarray(sample_image_array).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Metadata / DataFrame fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic ISIC-format DataFrame with 20 rows (2 positives)."""
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame(
        {
            "image_name": [f"img_{i:04d}" for i in range(n)],
            "target": [0] * (n - 2) + [1, 1],
            "age_approx": list(rng.integers(20, 80, n - 1).astype(float)) + [np.nan],
            "sex": (["male", "female"] * (n // 2))[:n],
            "anatom_site_general_challenge": (
                ["torso", "head/neck", "upper extremity", "lower extremity"] * 5
            )[:n],
        }
    )


@pytest.fixture
def sample_metadata_row(sample_df: pd.DataFrame) -> pd.Series:
    return sample_df.iloc[0]


@pytest.fixture
def sample_image_dir(tmp_path: "Path", sample_df: pd.DataFrame) -> "Path":
    """Temp directory populated with synthetic JPEG files for each row in sample_df."""
    from pathlib import Path

    img_dir: Path = tmp_path  # type: ignore[assignment]
    rng = np.random.default_rng(99)
    for name in sample_df["image_name"]:
        arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{name}.jpg")
    return img_dir
