"""Out-of-distribution detection via backbone embedding distance.

The classifier was trained exclusively on contact-dermoscopy images. A clinical
photo, a screenshot, or a heavily recompressed web download lives outside that
distribution; the model still emits a probability, but it is not trustworthy.

Approach
--------
Cache a sample of training-image backbone embeddings at startup, PCA-project them
to a lower dimension, and fit a Mahalanobis distance in that reduced space. At
request time, project the upload's embedding the same way and score its distance
to the training cloud. Distances past the 99th percentile of a held-out slice of
the training sample are flagged ``out_of_distribution``.

The PCA step matters at this sample size: EfficientNet-B4 embeddings are 1792-d,
and a full-dimensional covariance estimated from ~2048 samples (N/D ~= 1.14) is
numerically singular — inverting it produces a precision matrix dominated by
noise directions rather than the geometry of the training cloud. Projecting to
~64 dimensions first (N/D ~= 32) makes the covariance well-conditioned and keeps
the persisted artifact small enough to ship in the repo or as an MLflow artifact,
rather than needing the training dataset to be present at serve time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from cancer_detection.data.transforms import get_val_transforms
from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_N_COMPONENTS = 64
_SHRINKAGE = 0.1


class EmbeddingOODDetector:
    """Mahalanobis OOD detector over PCA-reduced MelanomaClassifier backbone features."""

    def __init__(
        self,
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
        mean: np.ndarray,
        precision: np.ndarray,
        threshold: float,
        n_reference: int,
    ) -> None:
        self.pca_mean = pca_mean.astype(np.float64)
        self.pca_components = pca_components.astype(np.float64)  # (n_components, feature_dim)
        self.mean = mean.astype(np.float64)  # (n_components,)
        self.precision = precision.astype(np.float64)  # (n_components, n_components)
        self.threshold = float(threshold)
        self.n_reference = int(n_reference)
        self.feature_dim = int(pca_components.shape[1])
        self.n_components = int(pca_components.shape[0])

    def _project(self, embeddings: np.ndarray) -> np.ndarray:
        return (embeddings - self.pca_mean) @ self.pca_components.T

    @classmethod
    def fit(
        cls,
        model: nn.Module,
        train_csv: Path | str,
        image_dir: Path | str,
        device: torch.device,
        n_samples: int = 2048,
        seed: int = 42,
        image_size: int = 384,
        batch_size: int = 32,
        percentile: float = 99.0,
        n_components: int = _DEFAULT_N_COMPONENTS,
    ) -> EmbeddingOODDetector | None:
        """Build a detector from a random sample of the training split.

        Returns None when the CSV/image dir is missing (e.g. CI without data) so
        the API can still start; predictions then omit the OOD flag.
        """
        csv_path = Path(train_csv)
        img_dir = Path(image_dir)
        if not csv_path.exists() or not img_dir.exists():
            logger.warning(
                "OOD detector skipped — training data not available",
                csv=str(csv_path),
                image_dir=str(img_dir),
            )
            return None

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None

        sample = df.sample(min(n_samples, len(df)), random_state=seed)
        transform = get_val_transforms(image_size)
        model.eval().to(device)

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            batch_images: list[torch.Tensor] = []
            for name in sample["image_name"]:
                path = img_dir / f"{name}.jpg"
                if not path.exists():
                    continue
                array = np.array(Image.open(path).convert("RGB"))
                batch_images.append(transform(image=array)["image"])
                if len(batch_images) >= batch_size:
                    stacked = torch.stack(batch_images).to(device)
                    feats = model.backbone(stacked).float().cpu().numpy()
                    embeddings.append(feats)
                    batch_images = []
            if batch_images:
                stacked = torch.stack(batch_images).to(device)
                feats = model.backbone(stacked).float().cpu().numpy()
                embeddings.append(feats)

        if not embeddings:
            logger.warning("OOD detector skipped — no images could be embedded")
            return None

        feats = np.concatenate(embeddings, axis=0).astype(np.float64)
        if feats.shape[0] < 32:
            logger.warning(
                "OOD detector skipped — too few embeddings to fit reliably",
                n_embedded=feats.shape[0],
            )
            return None

        # Hold out a slice for the percentile threshold: fitting mean/precision AND
        # the threshold on the same sample is in-sample calibration — Mahalanobis
        # distances within the fitting sample are systematically compressed (their
        # mean is exactly n_components * (n-1)/n by construction), so a held-out
        # in-distribution image would trip the resulting threshold far more than
        # the intended 1% rate.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(feats.shape[0])
        n_fit = max(1, int(0.75 * feats.shape[0]))
        fit_feats, cal_feats = feats[perm[:n_fit]], feats[perm[n_fit:]]
        if cal_feats.shape[0] == 0:
            cal_feats = fit_feats

        pca_mean = fit_feats.mean(axis=0)
        centered = fit_feats - pca_mean
        k = min(n_components, centered.shape[0] - 1, centered.shape[1])
        # SVD-based PCA: components are the top-k right singular vectors of the
        # centered data, i.e. directions of maximum variance.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:k]

        z_fit = centered @ components.T
        mean = z_fit.mean(axis=0)
        cov = np.cov(z_fit, rowvar=False)
        cov = np.atleast_2d(cov)
        # Shrinkage toward a scaled identity — proportional to the data's own scale,
        # unlike a fixed additive ridge — keeps the precision matrix well-behaved
        # even though PCA whitening already makes cov close to diagonal.
        trace_avg = float(np.trace(cov)) / cov.shape[0]
        cov = (1.0 - _SHRINKAGE) * cov + _SHRINKAGE * trace_avg * np.eye(cov.shape[0])
        # Symmetric PSD matrix, so the general SVD-based pseudo-inverse gives the same
        # result a symmetric-specific routine would — at n_components<=64 the cost
        # difference is negligible, and this keeps the module numpy-only (no scipy dep).
        precision = np.linalg.pinv(cov)

        cal_z = (cal_feats - pca_mean) @ components.T
        cal_distances = _mahalanobis_batch(cal_z, mean, precision)
        threshold = float(np.percentile(cal_distances, percentile))

        logger.info(
            "OOD detector fitted",
            n_reference=int(feats.shape[0]),
            n_components=k,
            feature_dim=int(feats.shape[1]),
            threshold=threshold,
            percentile=percentile,
            median_calibration_distance=float(np.median(cal_distances)),
        )
        return cls(
            pca_mean=pca_mean,
            pca_components=components,
            mean=mean,
            precision=precision,
            threshold=threshold,
            n_reference=int(feats.shape[0]),
        )

    def score(self, embedding: np.ndarray | torch.Tensor) -> tuple[float, bool]:
        """Return (mahalanobis_distance, is_ood) for a single embedding."""
        if isinstance(embedding, torch.Tensor):
            vec = embedding.detach().float().cpu().numpy().reshape(-1)
        else:
            vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
        z = self._project(vec[None, :])
        distance = float(_mahalanobis_batch(z, self.mean, self.precision)[0])
        return distance, distance > self.threshold

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            mean=self.mean,
            precision=self.precision,
            threshold=np.array([self.threshold]),
            n_reference=np.array([self.n_reference]),
        )
        meta = {
            "threshold": self.threshold,
            "n_reference": self.n_reference,
            "feature_dim": self.feature_dim,
            "n_components": self.n_components,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(
        cls, path: Path | str, expected_feature_dim: int | None = None
    ) -> EmbeddingOODDetector:
        """Load a persisted detector.

        Raises ValueError if ``expected_feature_dim`` is given and does not match
        the cache — e.g. a detector fitted against a different backbone. Without
        this check a stale cache silently propagates a broadcast error out of
        ``score()`` on every single request.
        """
        data = np.load(path)
        detector = cls(
            pca_mean=data["pca_mean"],
            pca_components=data["pca_components"],
            mean=data["mean"],
            precision=data["precision"],
            threshold=float(data["threshold"][0]),
            n_reference=int(data["n_reference"][0]),
        )
        if expected_feature_dim is not None and detector.feature_dim != expected_feature_dim:
            raise ValueError(
                f"OOD cache feature_dim={detector.feature_dim} does not match the live "
                f"backbone's feature_dim={expected_feature_dim} — likely fit against a "
                f"different model. Refit with EmbeddingOODDetector.fit()."
            )
        return detector


def _mahalanobis_batch(feats: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """sqrt((x - mean)^T precision (x - mean)) for each row of feats."""
    delta = feats - mean
    # BLAS-backed matrix multiply instead of a 3-operand einsum, which numpy runs
    # through an unoptimized C loop rather than GEMM — for D=64 this is the
    # difference between microseconds and (at full 1792-d) tens of seconds.
    sq_distance = ((delta @ precision) * delta).sum(axis=1)
    return np.asarray(np.sqrt(np.clip(sq_distance, 0.0, None)))
