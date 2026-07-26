"""Unit tests for MetadataEncoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cancer_detection.data.metadata import MetadataEncoder


@pytest.fixture
def encoder() -> MetadataEncoder:
    return MetadataEncoder()


def test_output_shape(encoder: MetadataEncoder, sample_metadata_row: pd.Series) -> None:
    tensor = encoder.encode(sample_metadata_row)
    assert tensor.shape == (3,)
    assert tensor.dtype == torch.float32


def test_known_values(encoder: MetadataEncoder) -> None:
    row = pd.Series({"age_approx": 50.0, "sex": "male", "anatom_site_general_challenge": "torso"})
    t = encoder.encode(row)
    # age 50 → (50-50)/15 = 0.0
    assert abs(t[0].item()) < 1e-5
    # male → 1.0
    assert abs(t[1].item() - 1.0) < 1e-5
    # torso index 3, N_SITES=6 → 3/5 = 0.6
    assert abs(t[2].item() - 0.6) < 1e-5


def test_missing_age(encoder: MetadataEncoder) -> None:
    row = pd.Series({"age_approx": np.nan, "sex": "female", "anatom_site_general_challenge": "torso"})
    t = encoder.encode(row)
    # NaN age → standardised mean → 0.0
    assert abs(t[0].item()) < 1e-5


def test_unknown_sex(encoder: MetadataEncoder) -> None:
    row = pd.Series({"age_approx": 40.0, "sex": None, "anatom_site_general_challenge": "torso"})
    t = encoder.encode(row)
    # Unknown sex → 0.5 sentinel
    assert abs(t[1].item() - 0.5) < 1e-5


def test_unknown_site(encoder: MetadataEncoder) -> None:
    row = pd.Series({"age_approx": 40.0, "sex": "male", "anatom_site_general_challenge": "unknown_region"})
    t = encoder.encode(row)
    # Unknown site → 0.5 sentinel
    assert abs(t[2].item() - 0.5) < 1e-5


def test_values_in_reasonable_range(encoder: MetadataEncoder, sample_metadata_row: pd.Series) -> None:
    t = encoder.encode(sample_metadata_row)
    # age (standardised): typical range ~ [-3, 3]
    assert -5 < t[0].item() < 5
    # sex: 0, 0.5, or 1
    assert t[1].item() in (0.0, 0.5, 1.0)
    # site: [0, 1]
    assert 0.0 <= t[2].item() <= 1.0


def test_encode_batch(encoder: MetadataEncoder, sample_df: pd.DataFrame) -> None:
    batch = encoder.encode_batch(sample_df)
    assert batch.shape == (len(sample_df), 3)
    assert batch.dtype == torch.float32
