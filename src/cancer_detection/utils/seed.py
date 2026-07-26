from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for full reproducibility.

    Covers Python, NumPy, PyTorch (CPU + CUDA). Also sets the PYTHONHASHSEED
    environment variable for hash-based operations and makes cuDNN deterministic.

    Note: deterministic cuDNN may reduce throughput. This is the right trade-off
    for a portfolio project where reproducible results are more important than
    maximum training speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
