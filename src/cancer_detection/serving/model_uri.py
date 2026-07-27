"""Resolve which MLflow model the API should load.

Default selection is the finished run with the highest validation AUROC
(``val/auroc``) that logged a ``model`` artifact. Override with ``MODEL_URI``
(e.g. ``runs:/<run_id>/model`` or ``models:/melanoma-classifier/1``).
"""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)

# Hosted tracking server on EC2 (public IP). Override with MLFLOW_TRACKING_URI
# for a local server or if the instance address changes.
DEFAULT_TRACKING_URI = "http://18.219.3.159:5000"
VAL_AUROC_METRIC = "val/auroc"


def ensure_tracking_uri() -> str:
    """Use ``MLFLOW_TRACKING_URI`` when set; otherwise the hosted EC2 server."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


def resolve_model_uri(explicit: str | None = None) -> str:
    """Return an MLflow model URI the Predictor can load.

    Resolution order:
    1. Explicit ``MODEL_URI`` / ``explicit`` argument
    2. Finished run with the highest ``val/auroc`` that logged a ``model`` artifact
    """
    if explicit:
        return explicit

    env_uri = os.environ.get("MODEL_URI")
    if env_uri:
        return env_uri

    tracking_uri = ensure_tracking_uri()
    client = MlflowClient(tracking_uri=tracking_uri)

    uri = _best_val_auroc_run_model_uri(client)
    if uri:
        return uri

    raise RuntimeError(
        "No MLflow run found with a logged model and val/auroc metric. "
        "Train a model first (python scripts/train.py), or set MODEL_URI "
        "explicitly (e.g. MODEL_URI=runs:/<run_id>/model)."
    )


def _best_val_auroc_run_model_uri(client: MlflowClient) -> str | None:
    """Pick the finished run with highest ``val/auroc`` that has a model artifact."""
    experiments = [
        exp for exp in client.search_experiments() if exp.lifecycle_stage == "active"
    ]
    experiment_ids = [exp.experiment_id for exp in experiments]
    if not experiment_ids:
        return None

    # Prefer server-side ordering; fall back to client-side sort if unsupported.
    try:
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string="attributes.status = 'FINISHED'",
            order_by=[f"metrics.`{VAL_AUROC_METRIC}` DESC"],
            max_results=100,
        )
    except Exception:
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string="attributes.status = 'FINISHED'",
            max_results=100,
        )
        runs = sorted(
            runs,
            key=lambda r: r.data.metrics.get(VAL_AUROC_METRIC, float("-inf")),
            reverse=True,
        )

    for run in runs:
        auroc = run.data.metrics.get(VAL_AUROC_METRIC)
        if auroc is None:
            continue
        if not _has_model_artifact(client, run.info.run_id):
            continue
        uri = f"runs:/{run.info.run_id}/model"
        logger.info(
            "Resolved model via highest val/auroc",
            uri=uri,
            val_auroc=auroc,
            run_id=run.info.run_id,
        )
        return uri
    return None


def _has_model_artifact(client: MlflowClient, run_id: str) -> bool:
    """Return True if the run logged a ``model`` artifact.

    MLflow 2 lists ``model`` at the run root. MLflow 3 often omits logged
    models from the root listing even though ``model/MLmodel`` exists — so we
    also probe the ``model`` path directly.
    """
    try:
        root = client.list_artifacts(run_id)
    except Exception:
        return False
    if any(a.path == "model" or a.path.startswith("model/") for a in root):
        return True
    try:
        nested = client.list_artifacts(run_id, "model")
    except Exception:
        return False
    return bool(nested) and any(
        a.path == "model" or a.path.startswith("model/") for a in nested
    )
