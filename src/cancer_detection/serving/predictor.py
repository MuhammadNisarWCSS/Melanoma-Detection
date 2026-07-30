from __future__ import annotations

import json
import os
import re
from pathlib import Path

import mlflow.pytorch
import numpy as np
import pandas as pd
import torch

from cancer_detection.data.metadata import MetadataEncoder
from cancer_detection.data.transforms import get_tta_transforms, get_val_transforms
from cancer_detection.explainability.gradcam import GradCAMWrapper
from cancer_detection.models.classifier import MelanomaClassifier
from cancer_detection.serving.ood import EmbeddingOODDetector
from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)

_RUN_URI_RE = re.compile(r"^runs:/([^/]+)/")


class Predictor:
    """Production inference engine for MelanomaClassifier.

    Responsibilities:
    - Load model + threshold artifact from MLflow model registry at startup.
    - Run 8-pass TTA inference and return mean probability + uncertainty (std).
    - Flag uploads whose backbone embedding is far from the training cloud (OOD).
    - Optionally generate GradCAM heatmap on the base (non-augmented) pass.
    - Never re-load the model between requests (single load at startup).
    """

    def __init__(
        self,
        model_uri: str,
        threshold_path: str | None = None,
        tta_n_passes: int = 8,
        device: str = "cpu",
        ood_cache_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.tta_n_passes = min(tta_n_passes, 8)
        self.encoder = MetadataEncoder()
        self.val_transform = get_val_transforms()
        self.tta_transforms = get_tta_transforms()[: self.tta_n_passes]

        self.model_uri = model_uri
        logger.info("Loading model from MLflow", uri=model_uri)
        self.model: MelanomaClassifier = mlflow.pytorch.load_model(model_uri)
        self.model.eval().to(self.device)
        logger.info("Model loaded", device=str(self.device))

        self.threshold = self._load_threshold(threshold_path)
        self.grad_cam = GradCAMWrapper(self.model)
        self.ood = self._load_or_fit_ood(ood_cache_path)

    @property
    def ood_enabled(self) -> bool:
        return self.ood is not None

    @staticmethod
    def _load_threshold(threshold_path: str | None) -> float:
        if threshold_path is None:
            return 0.5
        p = Path(threshold_path)
        if not p.exists():
            logger.warning("Threshold file not found; using default 0.5", path=str(p))
            return 0.5
        data = json.loads(p.read_text())
        threshold: float = data["threshold"]
        logger.info("Calibrated threshold loaded", threshold=threshold)
        return threshold

    def _load_or_fit_ood(self, cache_path: str | None) -> EmbeddingOODDetector | None:
        path = Path(cache_path or os.environ.get("OOD_CACHE_PATH", "artifacts/ood_detector.npz"))
        # getattr, not direct attribute access: nn.Module's __getattr__ fallback makes
        # mypy infer arbitrary submodule/parameter attributes as "Tensor | Module".
        expected_dim = int(getattr(self.model.backbone, "num_features"))
        if path.exists():
            try:
                detector = EmbeddingOODDetector.load(path, expected_feature_dim=expected_dim)
                logger.info(
                    "OOD detector loaded from cache",
                    path=str(path),
                    threshold=detector.threshold,
                )
                return detector
            except Exception as exc:
                logger.warning("Failed to load OOD cache; will redownload or refit", error=str(exc))

        downloaded = self._download_ood_from_run(expected_dim)
        if downloaded is not None:
            try:
                downloaded.save(path)
            except Exception as exc:
                logger.warning("Could not persist downloaded OOD cache", error=str(exc))
            return downloaded

        train_csv = Path(os.environ.get("OOD_TRAIN_CSV", "data/processed/train.csv"))
        image_dir = Path(os.environ.get("OOD_IMAGE_DIR", "data/processed/jpeg_448"))
        n_samples = int(os.environ.get("OOD_N_SAMPLES", "2048"))
        fitted_detector = EmbeddingOODDetector.fit(
            self.model,
            train_csv,
            image_dir,
            self.device,
            n_samples=n_samples,
        )
        if fitted_detector is not None:
            try:
                fitted_detector.save(path)
            except Exception as exc:
                logger.warning("Could not persist OOD cache", error=str(exc))
        return fitted_detector

    def _download_ood_from_run(self, expected_dim: int) -> EmbeddingOODDetector | None:
        """Try to fetch a detector logged by scripts/evaluate.py on the model's run.

        Lets a bare API container — no training dataset on disk, so it cannot fit
        its own detector — still get OOD detection instead of silently running
        without it, as long as evaluate.py was run at least once for this model.
        """
        match = _RUN_URI_RE.match(self.model_uri or "")
        if not match:
            return None
        run_id = match.group(1)
        try:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/ood_detector.npz"
            )
            detector = EmbeddingOODDetector.load(local_path, expected_feature_dim=expected_dim)
            logger.info("OOD detector downloaded from MLflow run", run_id=run_id)
            return detector
        except Exception as exc:
            logger.debug("No OOD artifact available on model run", run_id=run_id, error=str(exc))
            return None

    def predict(
        self,
        image_array: np.ndarray,
        metadata_row: pd.Series,
        return_gradcam: bool = True,
    ) -> dict:
        """Run inference on a single dermoscopy image."""
        meta_tensor = self.encoder.encode(metadata_row).unsqueeze(0).to(self.device)

        base_image = self.val_transform(image=image_array)["image"]
        base_tensor = base_image.unsqueeze(0).to(self.device)

        ood_distance: float | None = None
        # None (not False) when detection is disabled — indistinguishable from
        # "checked, in-distribution" otherwise, which matters on a host with no
        # training data available to fit the detector (e.g. a bare EC2 API image).
        out_of_distribution: bool | None = None
        with torch.no_grad():
            if self.ood is not None:
                embedding = self.model.backbone(base_tensor).squeeze(0)
                ood_distance, out_of_distribution = self.ood.score(embedding)

            base_logit = self.model(base_tensor, meta_tensor)
            tta_probs = [torch.sigmoid(base_logit).item()]

            for transform in self.tta_transforms[1:]:
                aug_image = transform(image=image_array)["image"]
                aug_tensor = aug_image.unsqueeze(0).to(self.device)
                logit = self.model(aug_tensor, meta_tensor)
                tta_probs.append(torch.sigmoid(logit).item())

        mean_prob = float(np.mean(tta_probs))
        std_prob = float(np.std(tta_probs))
        label = int(mean_prob >= self.threshold)
        # Normalise distance-from-boundary against the calibrated threshold, not a
        # fixed 0.5 — the decision boundary is self.threshold (often well below 0.5
        # for a sensitivity-first operating point), so a prediction just past it
        # should read as low-confidence, not as an arbitrary distance from the middle.
        t = self.threshold
        confidence = abs(mean_prob - t) / max(t, 1.0 - t)

        result: dict = {
            "probability": round(mean_prob, 4),
            "label": label,
            "label_str": "malignant" if label == 1 else "benign",
            "confidence": round(confidence, 4),
            "tta_std": round(std_prob, 4),
            "threshold_used": self.threshold,
            "out_of_distribution": out_of_distribution,
            "ood_distance": None if ood_distance is None else round(float(ood_distance), 4),
            "gradcam_heatmap_b64": None,
        }

        if return_gradcam:
            heatmap = self.grad_cam.generate_heatmap(
                image_tensor=base_image.to(self.device),
                metadata_tensor=meta_tensor.squeeze(0),
                original_image=image_array,
                target_category=label,
            )
            result["gradcam_heatmap_b64"] = GradCAMWrapper.heatmap_to_base64(heatmap)

        return result
