from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)

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
        Dict with keys: auroc, f1, sensitivity, specificity, ppv, npv, pauc, threshold
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)  # recall for positives
    specificity = tn / (tn + fp + 1e-8)  # recall for negatives
    ppv = tp / (tp + fp + 1e-8)  # precision / positive predictive value
    npv = tn / (tn + fn + 1e-8)  # negative predictive value

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
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

    This is the pAUC-above-min-TPR formulation used by the ISIC **2024** melanoma
    competition (ISIC 2020 itself was scored on plain AUROC). Clinically, it measures
    discriminative ability specifically where the model is sensitive enough to
    be useful — false negatives (missed cancers) are far more costly than
    false positives in screening contexts.

    Returns a value in [0, 1] normalized against the full (1 - min_tpr) x 1.0 band,
    so a perfect classifier scores 1.0 and a random classifier scores
    0.5 * (1 - min_tpr) — e.g. ~0.1 at the default min_tpr=0.80.
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

    # Normalize against the full achievable band, not the classifier's own ROC —
    # a denominator that depends on selected_fpr[0] (this classifier's FPR at the
    # min_tpr knee) is not monotone: a model that only reaches min_tpr at a bad FPR
    # gets a *smaller* denominator and can score higher than a genuinely better model.
    max_pauc = 1.0 - min_tpr
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


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: object,
    n_boot: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Stratified bootstrap confidence interval for a scalar metric.

    Resamples positives and negatives separately to avoid degenerate splits —
    with a small positive count, unstratified bootstrap frequently yields samples
    with zero positive cases where AUROC / pAUC are undefined.

    Args:
        y_true: Ground-truth binary labels
        y_prob: Predicted probabilities
        metric_fn: Callable(y_true, y_prob) -> float
        n_boot: Number of bootstrap replicates
        seed: Random seed for reproducibility
        confidence: Width of the interval (default 0.95)

    Returns:
        (lower_bound, upper_bound) at the requested confidence level
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    scores: list[float] = []
    n_failed = 0
    for _ in range(n_boot):
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([boot_pos, boot_neg])
        try:
            val = float(metric_fn(y_true[idx], y_prob[idx]))  # type: ignore[operator]
            if np.isfinite(val):
                scores.append(val)
        except Exception as exc:
            n_failed += 1
            if n_failed <= 3:
                logger.debug("bootstrap replicate failed", error=str(exc))

    if n_failed:
        logger.debug(
            "bootstrap_ci: replicates failed or were non-finite",
            n_failed=n_failed,
            n_boot=n_boot,
        )
    if not scores:
        return (float("nan"), float("nan"))

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1.0 - alpha)))
    return (lo, hi)


def roc_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 200,
) -> dict[str, list[float]]:
    """Compute ROC curve coordinates, downsampled for charting.

    Returns:
        Dict with keys 'fpr' and 'tpr' (lists of floats).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    if len(fpr) > max_points:
        indices = np.round(np.linspace(0, len(fpr) - 1, max_points)).astype(int)
        fpr = fpr[indices]
        tpr = tpr[indices]

    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 100,
) -> list[dict[str, float]]:
    """Compute sensitivity/specificity/PPV at n_points evenly spaced thresholds.

    Used to power the threshold slider on the frontend.

    Returns:
        List of dicts with keys: threshold, sensitivity, specificity, ppv, tp, fp, tn, fn
    """
    thresholds = np.linspace(float(y_prob.min()), float(y_prob.max()), n_points)
    result: list[dict[str, float]] = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        ppv = tp / (tp + fp + 1e-8)
        result.append(
            {
                "threshold": float(t),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "ppv": float(ppv),
                "tp": float(tp),
                "fp": float(fp),
                "tn": float(tn),
                "fn": float(fn),
            }
        )

    return result
