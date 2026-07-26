from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler

from cancer_detection.data.dataset import ISICDataset
from cancer_detection.data.metadata import MetadataEncoder
from cancer_detection.data.transforms import get_train_transforms, get_val_transforms


class ISICDataModule(LightningDataModule):
    """LightningDataModule for the ISIC 2020 melanoma dataset.

    Handles three-layer class imbalance mitigation:
    1. WeightedRandomSampler over-samples the minority (malignant) class
       so each batch sees ~positive_sample_rate positives.
    2. The caller supplies Focal Loss (layer 2) and threshold calibration (layer 3).

    Expects processed CSVs produced by scripts/prepare_data.py:
        data/processed/train.csv, val.csv, test.csv
    Each CSV must have: image_name, target, age_approx, sex,
                        anatom_site_general_challenge
    """

    def __init__(self, data_cfg: DictConfig, training_cfg: DictConfig) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.training_cfg = training_cfg
        self.encoder = MetadataEncoder()
        self._train_ds: ISICDataset | None = None
        self._val_ds: ISICDataset | None = None
        self._test_ds: ISICDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        processed = Path(self.data_cfg.processed_dir)
        image_dir = Path(self.data_cfg.image_dir)
        image_size: int = self.data_cfg.image_size

        if stage in ("fit", None):
            train_df = pd.read_csv(processed / "train.csv")
            val_df = pd.read_csv(processed / "val.csv")
            self._train_ds = ISICDataset(
                train_df, image_dir, get_train_transforms(image_size), self.encoder
            )
            self._val_ds = ISICDataset(
                val_df, image_dir, get_val_transforms(image_size), self.encoder
            )

        if stage in ("test", None):
            test_df = pd.read_csv(processed / "test.csv")
            self._test_ds = ISICDataset(
                test_df, image_dir, get_val_transforms(image_size), self.encoder
            )

    def _make_weighted_sampler(self, dataset: ISICDataset) -> WeightedRandomSampler:
        labels = dataset.labels
        pos_count = labels.sum()
        neg_count = len(labels) - pos_count
        # Guard against edge case (e.g., tiny synthetic dataset in tests)
        if pos_count == 0 or neg_count == 0:
            weights = np.ones(len(labels), dtype=np.float64)
        else:
            class_weights = np.array([1.0 / neg_count, 1.0 / pos_count])
            weights = class_weights[labels]
        sample_weights = torch.from_numpy(weights).double()
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _loader_kwargs(self) -> dict[str, Any]:
        """DataLoader settings shared by all three splits.

        Workers are spawned rather than forked on Windows, so respawning them every
        epoch is expensive enough to erase most of the benefit of using them.
        """
        num_workers: int = self.data_cfg.num_workers
        kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": True}
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            # Each prefetched batch is num_workers * this many decoded image tensors held
            # in IPC queues. Raising it past 2 has deadlocked worker spawn on Windows.
            kwargs["prefetch_factor"] = 2
        return kwargs

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._train_ds is not None
        sampler = self._make_weighted_sampler(self._train_ds)
        return DataLoader(
            self._train_ds,
            batch_size=self.training_cfg.batch_size,
            sampler=sampler,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.training_cfg.batch_size * 2,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self.training_cfg.batch_size * 2,
            shuffle=False,
            **self._loader_kwargs(),
        )
