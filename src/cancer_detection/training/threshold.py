"""Decision-threshold calibration that matches how the API actually scores images.

Two properties matter here and both were previously violated:

1. The threshold must be derived from the *same weights* that get deployed. Fitting
   it against whatever happens to be in memory when training stops silently pairs a
   threshold with a different model than the one served.
2. It must be derived from the *same inference pipeline*. The API averages eight TTA
   passes; averaging shrinks the spread of the probability distribution, so a
   threshold fitted on single-pass probabilities lands at a different operating
   point than intended once TTA is switched on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from cancer_detection.data.dataset import ISICDataset
from cancer_detection.data.metadata import MetadataEncoder
from cancer_detection.data.transforms import get_tta_transforms, get_val_transforms
from cancer_detection.evaluation.metrics import find_optimal_threshold
from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)


def predict_split(
    model: nn.Module,
    df: pd.DataFrame,
    image_dir: Path | str,
    device: torch.device,
    image_size: int = 384,
    batch_size: int = 32,
    num_workers: int = 0,
    tta_passes: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a labelled split and return (y_true, y_prob).

    With tta_passes > 1 the probabilities are the mean over that many deterministic
    dihedral views, which is exactly what Predictor.predict does per request.
    """
    encoder = MetadataEncoder()
    transforms = (
        get_tta_transforms(image_size)[:tta_passes]
        if tta_passes > 1
        else [get_val_transforms(image_size)]
    )

    model.eval().to(device)
    per_pass: list[np.ndarray] = []
    labels: np.ndarray | None = None

    for pass_idx, transform in enumerate(transforms, start=1):
        dataset = ISICDataset(df, image_dir, transform, encoder)
        # shuffle=False keeps every pass in the same row order so the means line up.
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        probs: list[float] = []
        targets: list[float] = []
        with torch.no_grad():
            for images, metadata, batch_labels in loader:
                logits = model(images.to(device), metadata.to(device))
                probs.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
                targets.extend(batch_labels.numpy().tolist())

        per_pass.append(np.array(probs))
        labels = np.array(targets, dtype=int)
        logger.info("Calibration pass complete", pass_index=pass_idx, of=len(transforms))

    assert labels is not None
    return labels, np.mean(per_pass, axis=0)


def calibrate_threshold(
    model: nn.Module,
    val_df: pd.DataFrame,
    image_dir: Path | str,
    device: torch.device,
    target_sensitivity: float = 0.80,
    output_path: Path | str | None = None,
    image_size: int = 384,
    batch_size: int = 32,
    num_workers: int = 0,
    tta_passes: int = 8,
    run_id: str | None = None,
    checkpoint: str | None = None,
) -> dict:
    """Pick the highest threshold on the validation split that still hits target sensitivity.

    run_id and checkpoint are recorded in the payload so a threshold can always be
    traced back to the exact weights it was fitted against.
    """
    y_true, y_prob = predict_split(
        model,
        val_df,
        image_dir,
        device,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        tta_passes=tta_passes,
    )

    threshold = find_optimal_threshold(y_true, y_prob, target_sensitivity)
    payload = {
        "threshold": float(threshold),
        "target_sensitivity": float(target_sensitivity),
        "tta_passes": int(tta_passes),
        "val_auroc": float(roc_auc_score(y_true, y_prob)),
        "n_val": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "run_id": run_id,
        "checkpoint": checkpoint,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Threshold written", path=str(path), **payload)

    return payload
