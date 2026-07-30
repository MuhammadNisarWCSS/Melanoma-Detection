from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all relevant random seeds for reproducibility.

    Covers Python, NumPy, PyTorch (CPU + CUDA), and sets PYTHONHASHSEED for
    hash-based operations.

    ``deterministic`` also owns the two cuDNN backend flags, since they are
    mutually exclusive: ``cudnn.benchmark`` autotunes the fastest convolution
    algorithm per input shape, but that choice is nondeterministic, so it can
    only be enabled when ``cudnn.deterministic`` is off. Pass the same value
    used for ``Trainer(deterministic=...)`` so the two never disagree.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
