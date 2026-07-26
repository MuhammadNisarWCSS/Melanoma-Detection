from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from albumentations.core.composition import Compose
from PIL import Image
from torch.utils.data import Dataset

from cancer_detection.data.metadata import MetadataEncoder


class ISICDataset(Dataset):
    """PyTorch Dataset for ISIC 2020 dermoscopy images.

    Returns a 3-tuple (image_tensor, metadata_tensor, label) for train/val splits,
    or a 2-tuple (image_tensor, metadata_tensor) for test splits where labels
    are unavailable.

    Args:
        df: DataFrame with columns [image_name, target, age_approx, sex,
            anatom_site_general_challenge]. 'target' is optional when is_test=True.
        image_dir: Directory containing <image_name>.jpg files.
        transform: Albumentations Compose pipeline. Applied to the raw numpy array.
        encoder: MetadataEncoder instance. If None, metadata tensor is all-zeros.
        is_test: When True, __getitem__ omits the label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path | str,
        transform: Compose | None = None,
        encoder: MetadataEncoder | None = None,
        is_test: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.encoder = encoder
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        row = self.df.iloc[idx]
        img_path = self.image_dir / f"{row['image_name']}.jpg"

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        meta: torch.Tensor = (
            self.encoder.encode(row)
            if self.encoder is not None
            else torch.zeros(3, dtype=torch.float32)
        )

        if self.is_test:
            return image, meta

        label = torch.tensor(float(row["target"]), dtype=torch.float32)
        return image, meta, label

    @property
    def labels(self) -> np.ndarray:
        """Return label array for computing class weights / samplers."""
        return self.df["target"].to_numpy(dtype=np.int64)
