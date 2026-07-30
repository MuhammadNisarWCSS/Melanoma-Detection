"""Unit tests for src/cancer_detection/data/datamodule.py."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from cancer_detection.data.datamodule import ISICDataModule


class _FakeDataset:
    """Minimal stand-in exposing only what _make_weighted_sampler reads."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels


def _module(positive_sample_rate: float | None = None) -> ISICDataModule:
    training_cfg = OmegaConf.create(
        {} if positive_sample_rate is None else {"positive_sample_rate": positive_sample_rate}
    )
    return ISICDataModule(OmegaConf.create({}), training_cfg)


def _realized_positive_rate(
    datamodule: ISICDataModule, labels: np.ndarray, n_draws: int = 20000
) -> float:
    sampler = datamodule._make_weighted_sampler(_FakeDataset(labels))
    draws = list(sampler)[:n_draws]
    drawn_labels = labels[draws]
    return float(drawn_labels.mean())


def _imbalanced_labels(seed: int, n: int = 2000, n_positive: int = 35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[:n_positive] = 1  # ~1.75% positive, matching the real ISIC prevalence
    rng.shuffle(labels)
    return labels


def test_default_positive_sample_rate_is_015() -> None:
    """No explicit config key should fall back to 0.15, not the old 50/50 default."""
    labels = _imbalanced_labels(seed=0)
    datamodule = _module(positive_sample_rate=None)
    realized = _realized_positive_rate(datamodule, labels)
    assert realized == pytest.approx(0.15, abs=0.02)


@pytest.mark.parametrize("target", [0.10, 0.30, 0.5])
def test_realized_positive_rate_matches_configured_target(target: float) -> None:
    labels = _imbalanced_labels(seed=1)
    datamodule = _module(positive_sample_rate=target)
    realized = _realized_positive_rate(datamodule, labels)
    assert realized == pytest.approx(target, abs=0.02)


def test_negative_coverage_improves_at_lower_positive_rate() -> None:
    """Lower positive_sample_rate should cover a larger fraction of unique negatives."""
    labels = _imbalanced_labels(seed=2)
    neg_idx = np.where(labels == 0)[0]

    def _unique_negative_coverage(target: float) -> float:
        datamodule = _module(positive_sample_rate=target)
        sampler = datamodule._make_weighted_sampler(_FakeDataset(labels))
        draws = np.array(list(sampler))
        seen_negatives = set(draws[np.isin(draws, neg_idx)].tolist())
        return len(seen_negatives) / len(neg_idx)

    coverage_015 = _unique_negative_coverage(0.15)
    coverage_050 = _unique_negative_coverage(0.5)
    assert coverage_015 > coverage_050
