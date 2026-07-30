"""Unit tests for src/cancer_detection/evaluation/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from cancer_detection.evaluation.metrics import find_optimal_threshold, partial_auc


def _labels_and_scores(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_prob = np.concatenate([pos, neg])
    return y_true.astype(int), y_prob


def test_partial_auc_perfect_classifier_scores_near_one() -> None:
    y_true, y_prob = _labels_and_scores(
        pos=np.linspace(0.9, 1.0, 50), neg=np.linspace(0.0, 0.1, 50)
    )
    assert partial_auc(y_true, y_prob, min_tpr=0.80) == pytest.approx(1.0, abs=1e-6)


def test_partial_auc_random_classifier_scores_below_perfect() -> None:
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.ones(200), np.zeros(200)]).astype(int)
    y_prob = rng.uniform(0, 1, size=400)
    pauc = partial_auc(y_true, y_prob, min_tpr=0.80)
    # Constant-denominator normalization: perfect -> 1.0, random -> ~0.5*(1-min_tpr).
    assert 0.0 <= pauc < 0.5


def test_partial_auc_is_monotone_in_auroc() -> None:
    """A worse classifier (by AUROC) must not score a higher pAUC.

    Regression test for a normalization bug where the denominator depended on the
    classifier's own ROC curve: a model that only reached the min_tpr knee at a bad
    FPR got a *smaller* denominator and could score pAUC=1.0 while its true AUROC
    was ~0.10 — worse than random.
    """
    # Constructed so 81% of negatives outrank every positive, and the remaining 90
    # negatives outrank nothing — true AUROC = (100*90) / (100*900) = 0.10.
    n_pos, n_neg = 100, 900
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
    y_prob = np.concatenate(
        [
            np.full(n_pos, 0.5),  # positives: mid-range scores
            np.full(810, 0.9),  # 810 negatives score highest
            np.full(90, 0.1),  # remaining 90 negatives score lowest
        ]
    )
    bad_pauc = partial_auc(y_true, y_prob, min_tpr=0.80)

    # A genuinely strong classifier: positives score high, negatives score low.
    good_true = np.concatenate([np.ones(100), np.zeros(900)]).astype(int)
    good_prob = np.concatenate([np.full(100, 0.95), np.full(900, 0.05)])
    good_pauc = partial_auc(good_true, good_prob, min_tpr=0.80)

    assert good_pauc > bad_pauc


def test_find_optimal_threshold_meets_target_sensitivity() -> None:
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)
    y_prob = np.concatenate([rng.normal(0.7, 0.15, 50), rng.normal(0.3, 0.15, 50)]).clip(0, 1)

    threshold = find_optimal_threshold(y_true, y_prob, target_sensitivity=0.80)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn)
    assert sensitivity >= 0.80 - 1e-9


def test_find_optimal_threshold_falls_back_when_unreachable() -> None:
    # sklearn's roc_curve always includes the point where every sample is predicted
    # positive (tpr=1.0), so any target_sensitivity <= 1.0 is technically reachable.
    # A target above 1.0 is the only way to force the "no threshold works" branch.
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    threshold = find_optimal_threshold(y_true, y_prob, target_sensitivity=1.5)
    assert threshold == 0.5
