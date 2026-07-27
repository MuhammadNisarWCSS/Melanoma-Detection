from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from cancer_detection.serving.model_uri import resolve_model_uri
from cancer_detection.serving.predictor import Predictor
from cancer_detection.serving.schemas import PredictResponse
from cancer_detection.utils.logger import configure_logging, get_logger

# File tee stays off in Docker; enable locally with LOG_TO_FILE=1 if desired.
configure_logging(name="api")
logger = get_logger(__name__)

predictor: Predictor | None = None


def _load_predictor() -> None:
    """Resolve and load the model (runs in a worker thread at startup)."""
    global predictor

    threshold_path = os.environ.get("THRESHOLD_PATH", "artifacts/threshold.json")
    tta_passes = int(os.environ.get("TTA_N_PASSES", "8"))
    device = os.environ.get("DEVICE", "cpu")

    try:
        model_uri = resolve_model_uri()
    except Exception as exc:
        model_uri = None
        logger.warning("Could not resolve model URI", error=str(exc))

    logger.info("Starting up", model_uri=model_uri, device=device)
    try:
        if model_uri is None:
            raise RuntimeError("No model URI available")
        predictor = Predictor(
            model_uri=model_uri,
            threshold_path=threshold_path,
            tta_n_passes=tta_passes,
            device=device,
        )
        logger.info("API ready")
    except Exception as exc:
        # Graceful degradation: API stays up; predict endpoints return 503.
        logger.warning("Model not loaded — predict endpoints will return 503", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start serving immediately; load the model in the background.

    Docker healthchecks hit /health during start-period. Blocking on a large
    MLflow/torch load here made the API look unhealthy and blocked frontend.
    """
    global predictor

    load_task = asyncio.create_task(asyncio.to_thread(_load_predictor))
    try:
        yield
    finally:
        if not load_task.done():
            load_task.cancel()
            try:
                await load_task
            except asyncio.CancelledError:
                pass
        predictor = None
        logger.info("API shutdown")


app = FastAPI(
    title="Melanoma Detection API",
    description=(
        "Multimodal dermoscopy image classifier with GradCAM explainability. "
        "Built on ISIC 2020 dataset with EfficientNet-B4 + patient metadata fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health() -> dict:
    """Liveness probe for Docker / Kubernetes health checks."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
    }


@app.get("/metadata", tags=["System"])
async def metadata() -> dict:
    """Return model and threshold metadata."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "threshold": predictor.threshold,
        "tta_passes": predictor.tta_n_passes,
        "device": str(predictor.device),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    image: UploadFile = File(..., description="Dermoscopy image (JPEG or PNG)"),
    age_approx: float = Form(50.0, description="Patient age (years)"),
    sex: str = Form("unknown", description="'male', 'female', or 'unknown'"),
    anatom_site: str = Form(
        "unknown",
        description="Anatomical site: torso, head/neck, upper extremity, lower extremity, etc.",
    ),
    return_gradcam: bool = Form(True, description="Include GradCAM heatmap in response"),
) -> PredictResponse:
    """Classify a single dermoscopy image as benign or malignant.

    Accepts multipart/form-data with both the image file and patient metadata.
    Returns probability, binary label, uncertainty estimate, and optional GradCAM.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc

    image_array = np.array(pil_img)
    metadata_row = pd.Series(
        {
            "age_approx": age_approx,
            "sex": sex.strip().lower(),
            "anatom_site_general_challenge": anatom_site.strip().lower(),
        }
    )

    try:
        result = predictor.predict(image_array, metadata_row, return_gradcam)
    except Exception as exc:
        logger.error("Prediction failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    return PredictResponse(**result)


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    images: list[UploadFile] = File(...),
    age_approx: float = Form(50.0),
    sex: str = Form("unknown"),
    anatom_site: str = Form("unknown"),
) -> dict:
    """Classify a list of dermoscopy images with shared metadata.

    Note: In production, each image would carry its own metadata. This endpoint
    demonstrates batch throughput for the same patient / imaging session.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata_row = pd.Series(
        {
            "age_approx": age_approx,
            "sex": sex.strip().lower(),
            "anatom_site_general_challenge": anatom_site.strip().lower(),
        }
    )
    results = []
    for img_file in images:
        raw = await img_file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        image_array = np.array(pil_img)
        result = predictor.predict(image_array, metadata_row, return_gradcam=False)
        results.append(result)

    return {"predictions": results, "count": len(results)}
