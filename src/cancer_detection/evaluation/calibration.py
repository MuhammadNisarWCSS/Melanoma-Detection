from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) — weighted average of |accuracy - confidence|.

    A calibrated model has ECE ≈ 0: when it outputs probability 0.7, it should
    be correct ~70% of the time. This is a hard requirement for clinical decision
    support tools where the output probability informs treatment decisions.

    Args:
        y_true: Ground-truth binary labels
        y_prob: Predicted probabilities in [0, 1]
        n_bins: Number of equal-width confidence bins

    Returns:
        ECE in [0, 1]. Lower is better. Well-calibrated models typically < 0.05.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute binned data for a reliability (calibration) diagram.

    Returns:
        Dict with keys:
            mean_pred:  Mean predicted probability per bin
            mean_true:  Fraction of positives per bin (empirical probability)
            counts:     Number of samples per bin
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred: list[float] = []
    mean_true: list[float] = []
    counts: list[float] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        mean_true.append(float(y_true[mask].mean()))
        counts.append(float(mask.sum()))

    return {"mean_pred": mean_pred, "mean_true": mean_true, "counts": counts}
