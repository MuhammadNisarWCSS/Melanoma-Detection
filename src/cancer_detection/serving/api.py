from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from cancer_detection.serving.model_uri import ensure_tracking_uri, resolve_model_uri
from cancer_detection.serving.predictor import Predictor
from cancer_detection.serving.schemas import PredictResponse
from cancer_detection.utils.logger import configure_logging, get_logger

# File tee stays off in Docker; enable locally with LOG_TO_FILE=1 if desired.
configure_logging(name="api")
logger = get_logger(__name__)

predictor: Predictor | None = None

# Serializes reloads so two concurrent /reload-model calls can't build two
# Predictors (each downloading + loading a full torch model) at once.
_reload_lock = asyncio.Lock()

# In-memory cache for test metrics — populated lazily on first request.
_test_metrics_cache: dict[str, Any] | None = None
_LOCAL_TEST_METRICS = Path(os.environ.get("TEST_METRICS_PATH", "artifacts/test_metrics.json"))


def _build_predictor(model_uri: str) -> Predictor:
    threshold_path = os.environ.get("THRESHOLD_PATH", "artifacts/threshold.json")
    tta_passes = int(os.environ.get("TTA_N_PASSES", "8"))
    device = os.environ.get("DEVICE", "cpu")
    return Predictor(
        model_uri=model_uri,
        threshold_path=threshold_path,
        tta_n_passes=tta_passes,
        device=device,
    )


def _load_predictor() -> None:
    """Resolve and load the model (runs in a worker thread at startup)."""
    global predictor

    try:
        model_uri = resolve_model_uri()
    except Exception as exc:
        logger.warning("Could not resolve model URI", error=str(exc))
        return

    logger.info("Starting up", model_uri=model_uri)
    try:
        predictor = _build_predictor(model_uri)
        logger.info("API ready")
    except Exception as exc:
        # Graceful degradation: API stays up; predict endpoints return 503.
        logger.warning("Model not loaded — predict endpoints will return 503", error=str(exc))


def _reload_predictor_sync() -> str:
    """Re-resolve the best MLflow model and hot-swap it in if it changed.

    Builds the new Predictor before touching the global, so an in-flight
    /predict request keeps using the old (still valid) object until the new
    one is fully loaded — the swap itself is a single atomic assignment.
    """
    global predictor

    model_uri = resolve_model_uri()
    if predictor is not None and predictor.model_uri == model_uri:
        logger.info("Reload requested — already serving the resolved model", model_uri=model_uri)
        return model_uri

    logger.info("Reloading model", model_uri=model_uri)
    new_predictor = _build_predictor(model_uri)
    predictor = new_predictor
    logger.info("Model reloaded", model_uri=model_uri)
    return model_uri


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
    root_path="/api",
)

# Same-origin in production (nginx proxies /api on the frontend's own port), so this
# only needs to cover local dev (Vite on :3000) and the standalone hosted frontend.
_default_origins = "http://localhost:3000,http://18.219.3.159:3000"
_cors_origins = os.environ.get("CORS_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
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


@app.get("/test-metrics", tags=["System"])
async def test_metrics() -> JSONResponse:
    """Return held-out test metrics for the champion model.

    Resolution order:
    1. ``artifacts/test_metrics.json`` on disk (local dev or baked into the image)
    2. ``test_metrics.json`` artifact from the highest-val-AUROC MLflow run

    The full payload includes AUROC, sensitivity/specificity with bootstrap CIs,
    ECE, ROC curve coordinates, reliability diagram data, and a threshold sweep
    for the interactive slider.  Returns 503 when unavailable.
    """
    global _test_metrics_cache

    if _test_metrics_cache is not None:
        return JSONResponse(_test_metrics_cache)

    # 1 — local file
    if _LOCAL_TEST_METRICS.exists():
        try:
            data = json.loads(_LOCAL_TEST_METRICS.read_text(encoding="utf-8"))
            _test_metrics_cache = data
            logger.info("Served test metrics from local file", path=str(_LOCAL_TEST_METRICS))
            return JSONResponse(data)
        except Exception as exc:
            logger.warning("Could not read local test_metrics.json", error=str(exc))

    # 2 — MLflow artifact download
    try:
        tracking_uri = ensure_tracking_uri()
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = [e for e in client.search_experiments() if e.lifecycle_stage == "active"]
        exp_ids = [e.experiment_id for e in experiments]

        runs = client.search_runs(
            experiment_ids=exp_ids,
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.`val/auroc` DESC"],
            max_results=50,
        )

        for run in runs:
            run_id = run.info.run_id
            try:
                artifacts = client.list_artifacts(run_id)
                has_json = any(a.path == "test_metrics.json" for a in artifacts)
                if not has_json:
                    continue

                with tempfile.TemporaryDirectory() as tmp:
                    local_path = mlflow.artifacts.download_artifacts(
                        artifact_uri=f"runs:/{run_id}/test_metrics.json",
                        dst_path=tmp,
                    )
                    data = json.loads(Path(local_path).read_text(encoding="utf-8"))
                    _test_metrics_cache = data
                    logger.info("Served test metrics from MLflow", run_id=run_id)
                    return JSONResponse(data)
            except Exception as exc:
                logger.debug("Skipping run for test metrics", run_id=run_id, error=str(exc))
                continue
    except Exception as exc:
        logger.warning("Could not fetch test metrics from MLflow", error=str(exc))

    raise HTTPException(status_code=503, detail="Test metrics not available")


@app.get("/metadata", tags=["System"])
async def metadata() -> dict:
    """Return model and threshold metadata."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "threshold": predictor.threshold,
        "tta_passes": predictor.tta_n_passes,
        "device": str(predictor.device),
        "ood_enabled": predictor.ood_enabled,
    }


@app.post("/reload-model", tags=["System"])
async def reload_model(
    x_reload_token: str | None = Header(default=None, alias="X-Reload-Token"),
) -> dict:
    """Re-resolve the best MLflow model and hot-swap it in, no restart needed.

    Call this after a training run finishes so a new best ``val/auroc`` run
    takes effect immediately. If unset, the currently-loaded model keeps
    serving until the container itself restarts. Set ``RELOAD_TOKEN`` in any
    environment reachable from the internet — this triggers a full model
    download + load, so it should not be callable by anonymous requests.
    """
    expected_token = os.environ.get("RELOAD_TOKEN")
    if expected_token and x_reload_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid or missing reload token")

    async with _reload_lock:
        try:
            model_uri = await asyncio.to_thread(_reload_predictor_sync)
        except Exception as exc:
            logger.warning("Model reload failed", error=str(exc))
            raise HTTPException(status_code=503, detail=f"Reload failed: {exc}") from exc

    return {"reloaded": True, "model_uri": model_uri, "model_loaded": predictor is not None}


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
