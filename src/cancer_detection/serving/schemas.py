from __future__ import annotations

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """API response for a single dermoscopy image prediction."""

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted malignancy probability after TTA averaging",
    )
    label: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary decision (0 = benign, 1 = malignant) at calibrated threshold",
    )
    label_str: str = Field(..., description="Human-readable label: 'benign' or 'malignant'")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Distance from decision boundary, normalised to [0, 1]",
    )
    tta_std: float = Field(
        ...,
        ge=0.0,
        description=(
            "Standard deviation of probabilities across TTA passes. "
            "High values signal prediction uncertainty — a clinical safety flag."
        ),
    )
    threshold_used: float = Field(
        ...,
        description="Calibrated decision threshold applied to produce label",
    )
    gradcam_heatmap_b64: str | None = Field(
        None,
        description="Base64-encoded PNG with GradCAM overlay. Null when return_gradcam=False.",
    )


class BatchPredictResponse(BaseModel):
    """API response for batch prediction."""

    predictions: list[PredictResponse]
    count: int
