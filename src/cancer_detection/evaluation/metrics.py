from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# numpy renamed trapz -> trapezoid in 2.0 and dropped the old spelling. pyproject
# allows numpy>=1.26, so resolve whichever name this install provides.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a full suite of clinical classification metrics.

    Args:
        y_true: Ground-truth binary labels {0, 1}
        y_prob: Predicted probabilities in [0, 1]
        threshold: Decision threshold for binary predictions

    Returns:
        Dict with keys: auroc, f1, sensitivity, specificity, pauc, threshold
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)   # recall for positives
    specificity = tn / (tn + fp + 1e-8)   # recall for negatives

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "pauc": float(partial_auc(y_true, y_prob, min_tpr=0.80)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def partial_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_tpr: float = 0.80,
) -> float:
    """Compute the partial AUC above a minimum TPR (sensitivity) threshold.

    This is the official ISIC 2020 competition metric. Clinically, it measures
    discriminative ability specifically where the model is sensitive enough to
    be useful — false negatives (missed cancers) are far more costly than
    false positives in screening contexts.

    Returns a value in [0, 1] normalized by the maximum achievable pAUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    mask = tpr >= min_tpr
    if not mask.any():
        return 0.0

    # Integrate only the portion of the ROC curve where TPR >= min_tpr, measuring
    # height above the min_tpr line rather than above zero — the band below the line
    # is available to every classifier and would otherwise dominate the score.
    selected_fpr = fpr[mask]
    selected_tpr = tpr[mask]
    raw_pauc = float(_trapezoid(selected_tpr - min_tpr, selected_fpr))

    # Normalize so perfect classifier → 1.0 (random → 0.5)
    max_pauc = (1.0 - float(selected_fpr[0])) * (1.0 - min_tpr)
    return raw_pauc / (max_pauc + 1e-8)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.80,
) -> float:
    """Find the highest threshold that maintains at least target_sensitivity.

    Strategy: among all thresholds where sensitivity >= target, pick the
    largest one. This maximises specificity (minimises false positives) while
    respecting the clinical sensitivity floor.

    Falls back to 0.5 if no threshold achieves the target.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    mask = tpr >= target_sensitivity
    if not mask.any():
        return 0.5
    valid_thresholds = thresholds[mask]
    return float(np.max(valid_thresholds))
