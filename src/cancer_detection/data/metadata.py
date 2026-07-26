from __future__ import annotations

import numpy as np
import pandas as pd
import torch

# Fixed encoding maps — learned from train set statistics to avoid leakage
SITE_MAP: dict[str, int] = {
    "head/neck": 0,
    "upper extremity": 1,
    "lower extremity": 2,
    "torso": 3,
    "palms/soles": 4,
    "oral/genital": 5,
}
SEX_MAP: dict[str, float] = {"male": 1.0, "female": 0.0}

# ISIC 2020 train set age statistics (used for standardization without fitting on val/test)
_AGE_MEAN: float = 50.0
_AGE_STD: float = 15.0
_N_SITES: int = len(SITE_MAP)

# Sentinel value for missing categoricals (mid-point = 0.5 signals unknown)
_UNKNOWN_FLOAT: float = 0.5


class MetadataEncoder:
    """Stateless encoder for ISIC patient metadata.

    Encodes three features into a float32 tensor:
      [0] age_approx   — standardized with fixed (mean=50, std=15)
      [1] sex          — binary (male=1, female=0, unknown=0.5)
      [2] anatom_site  — ordinal normalized to [0, 1] (unknown=0.5)
    """

    def encode(self, row: pd.Series) -> torch.Tensor:
        age_raw = row.get("age_approx", np.nan)
        if age_raw is None or (isinstance(age_raw, float) and np.isnan(age_raw)):
            age_norm = 0.0  # mean-impute maps to 0 after standardization
        else:
            age_norm = (float(age_raw) - _AGE_MEAN) / _AGE_STD

        sex_raw = str(row.get("sex", "")).strip().lower()
        sex = SEX_MAP.get(sex_raw, _UNKNOWN_FLOAT)

        site_raw = str(row.get("anatom_site_general_challenge", "")).strip().lower()
        site_idx = SITE_MAP.get(site_raw, -1)
        site_norm = site_idx / (_N_SITES - 1) if site_idx >= 0 else _UNKNOWN_FLOAT

        return torch.tensor([age_norm, sex, site_norm], dtype=torch.float32)

    def encode_batch(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.stack([self.encode(row) for _, row in df.iterrows()])
