from __future__ import annotations

import json
from pathlib import Path

import mlflow.pytorch
import numpy as np
import pandas as pd
import torch

from cancer_detection.data.metadata import MetadataEncoder
from cancer_detection.data.transforms import get_val_transforms, get_tta_transforms
from cancer_detection.explainability.gradcam import GradCAMWrapper
from cancer_detection.models.classifier import MelanomaClassifier
from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Production inference engine for MelanomaClassifier.

    Responsibilities:
    - Load model + threshold artifact from MLflow model registry at startup.
    - Run 8-pass TTA inference and return mean probability + uncertainty (std).
    - Optionally generate GradCAM heatmap on the base (non-augmented) pass.
    - Never re-load the model between requests (single load at startup).

    Args:
        model_uri:      MLflow model URI, e.g. 'runs:/<run_id>/model'.
        threshold_path: Path to JSON file produced by ThresholdCalibrationCallback.
        tta_n_passes:   Number of TTA augmentation variants. Must be ≤ 8.
        device:         'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_uri: str,
        threshold_path: str | None = None,
        tta_n_passes: int = 8,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.tta_n_passes = min(tta_n_passes, 8)
        self.encoder = MetadataEncoder()
        self.val_transform = get_val_transforms()
        self.tta_transforms = get_tta_transforms()[: self.tta_n_passes]

        logger.info("Loading model from MLflow", uri=model_uri)
        self.model: MelanomaClassifier = mlflow.pytorch.load_model(model_uri)
        self.model.eval().to(self.device)
        logger.info("Model loaded", device=str(self.device))

        self.threshold = self._load_threshold(threshold_path)
        self.grad_cam = GradCAMWrapper(self.model)

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

    def predict(
        self,
        image_array: np.ndarray,
        metadata_row: pd.Series,
        return_gradcam: bool = True,
    ) -> dict:
        """Run inference on a single dermoscopy image.

        Args:
            image_array:   Raw uint8 RGB numpy array (any spatial resolution).
            metadata_row:  pd.Series with keys age_approx, sex,
                           anatom_site_general_challenge.
            return_gradcam: Whether to include GradCAM heatmap in response.

        Returns:
            Dict compatible with PredictResponse schema.
        """
        meta_tensor = (
            self.encoder.encode(metadata_row).unsqueeze(0).to(self.device)
        )

        # Base (non-augmented) pass — used for GradCAM and as TTA pass 0
        base_image = self.val_transform(image=image_array)["image"]
        base_tensor = base_image.unsqueeze(0).to(self.device)

        # TTA does not need gradients; GradCAM below does, so do not wrap
        # the whole method in @torch.no_grad().
        with torch.no_grad():
            base_logit = self.model(base_tensor, meta_tensor)
            tta_probs = [torch.sigmoid(base_logit).item()]

            # Additional TTA passes (transforms[0] is the same as val_transform)
            for transform in self.tta_transforms[1:]:
                aug_image = transform(image=image_array)["image"]
                aug_tensor = aug_image.unsqueeze(0).to(self.device)
                logit = self.model(aug_tensor, meta_tensor)
                tta_probs.append(torch.sigmoid(logit).item())

        mean_prob = float(np.mean(tta_probs))
        std_prob = float(np.std(tta_probs))
        label = int(mean_prob >= self.threshold)

        result: dict = {
            "probability": round(mean_prob, 4),
            "label": label,
            "label_str": "malignant" if label == 1 else "benign",
            "confidence": round(abs(mean_prob - 0.5) * 2.0, 4),
            "tta_std": round(std_prob, 4),
            "threshold_used": self.threshold,
            "gradcam_heatmap_b64": None,
        }

        if return_gradcam:
            heatmap = self.grad_cam.generate_heatmap(
                image_tensor=base_image.to(self.device),
                metadata_tensor=meta_tensor.squeeze(0),
                original_image=image_array,
            )
            result["gradcam_heatmap_b64"] = GradCAMWrapper.heatmap_to_base64(heatmap)

        return result
