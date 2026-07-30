"""Unit tests for ISICDataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from cancer_detection.data.dataset import ISICDataset
from cancer_detection.data.metadata import MetadataEncoder
from cancer_detection.data.transforms import get_val_transforms


@pytest.fixture
def dataset(sample_df: pd.DataFrame, sample_image_dir: Path) -> ISICDataset:
    return ISICDataset(
        df=sample_df,
        image_dir=sample_image_dir,
        transform=get_val_transforms(64),
        encoder=MetadataEncoder(),
        is_test=False,
    )


def test_length(dataset: ISICDataset, sample_df: pd.DataFrame) -> None:
    assert len(dataset) == len(sample_df)


def test_item_types(dataset: ISICDataset) -> None:
    image, meta, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(meta, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def test_image_shape(dataset: ISICDataset) -> None:
    image, _, _ = dataset[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32


def test_metadata_shape(dataset: ISICDataset) -> None:
    _, meta, _ = dataset[0]
    assert meta.shape == (3,)
    assert meta.dtype == torch.float32


def test_label_dtype(dataset: ISICDataset) -> None:
    _, _, label = dataset[0]
    assert label.dtype == torch.float32
    assert label.item() in (0.0, 1.0)


def test_positive_label(dataset: ISICDataset, sample_df: pd.DataFrame) -> None:
    pos_idx = sample_df[sample_df["target"] == 1].index[0]
    _, _, label = dataset[pos_idx]
    assert label.item() == 1.0


def test_labels_property(dataset: ISICDataset, sample_df: pd.DataFrame) -> None:
    labels = dataset.labels
    assert labels.shape == (len(sample_df),)
    assert labels.sum() == sample_df["target"].sum()


def test_test_mode_no_label(sample_df: pd.DataFrame, sample_image_dir: Path) -> None:
    ds = ISICDataset(
        df=sample_df,
        image_dir=sample_image_dir,
        transform=get_val_transforms(64),
        encoder=MetadataEncoder(),
        is_test=True,
    )
    item = ds[0]
    assert len(item) == 2  # (image, meta) — no label


def test_no_encoder_returns_zeros(sample_df: pd.DataFrame, sample_image_dir: Path) -> None:
    ds = ISICDataset(
        df=sample_df,
        image_dir=sample_image_dir,
        transform=get_val_transforms(64),
        encoder=None,
    )
    _, meta, _ = ds[0]
    assert torch.all(meta == 0.0)
