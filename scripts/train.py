"""Training entrypoint — launched via Hydra.

Examples:
    # Full training run with defaults:
    python scripts/train.py

    # Quick smoke test (2 batches, CPU, local MLflow):
    python scripts/train.py training=fast_dev

    # Ablation sweep across models and learning rates:
    python scripts/train.py -m model=efficientnet_b2,efficientnet_b4 training.lr=1e-3,5e-4

    # Override single hyperparameters:
    python scripts/train.py training.batch_size=64 training.epochs=50
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

# Windows terminals default to CP1252 which cannot encode emoji characters that
# MLflow prints to stdout (e.g. the 🏃 in the "View run at: …" summary line).
# Reconfigure both streams to UTF-8 before any library code touches them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import hydra
import mlflow
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from cancer_detection.data.datamodule import ISICDataModule
from cancer_detection.training.callbacks import ThresholdCalibrationCallback, log_peak_val_metrics
from cancer_detection.training.lit_module import MelanomaLitModule
from cancer_detection.utils.logger import configure_logging, get_logger
from cancer_detection.utils.seed import set_seed

configure_logging()
logger = get_logger(__name__)

# Path to the MLflow SQLite backend, resolved relative to this file so it works
# regardless of the working directory Hydra sets at runtime.
_MLFLOW_DB = Path(__file__).resolve().parent.parent / "mlflow.db"


def _register_run_guardian(run_id: str) -> dict[str, bool]:
    """Guarantee a run is never left as RUNNING after the process exits.

    Returns a mutable ``status`` dict; set ``status["success"] = True`` before
    the process exits to record the run as FINISHED rather than FAILED.

    Strategy: on ``atexit``, try the MLflow HTTP client first (fast path that
    also updates the UI immediately); if that fails for any reason (network
    timeout, encoding error, server down), fall back to writing directly to the
    SQLite backend file — which is always local and never times out.
    """
    status: dict[str, bool] = {"success": False}

    def _guardian() -> None:
        final_status = "FINISHED" if status["success"] else "FAILED"
        # Fast path: HTTP client (already has UTF-8 stdout from module init).
        try:
            mlflow.MlflowClient().set_terminated(run_id, status=final_status)
            return
        except Exception:
            pass

        # Fallback: write directly to the SQLite backend, bypassing HTTP entirely.
        if not _MLFLOW_DB.exists():
            return
        try:
            import sqlite3
            from datetime import datetime, timezone

            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            with sqlite3.connect(str(_MLFLOW_DB), timeout=10) as conn:
                conn.execute(
                    "UPDATE runs SET status=?, end_time=? WHERE run_uuid=? AND status='RUNNING'",
                    (final_status, now_ms, run_id),
                )
            logger.warning(
                "MLflow HTTP teardown failed — run status written directly to SQLite",
                run_id=run_id,
                final_status=final_status,
            )
        except Exception as exc:
            logger.warning("Run guardian SQLite fallback also failed", error=str(exc))

    atexit.register(_guardian)
    return status


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> float:
    """Train a MelanomaClassifier and register the best checkpoint in MLflow.

    Returns:
        Best validation AUROC (useful for Hydra multirun comparisons).
    """
    set_seed(cfg.training.seed)
    logger.info("Config", cfg=OmegaConf.to_yaml(cfg))

    fast_dev_run: bool = cfg.training.get("fast_dev_run", False)
    mode = "smoke test (fast_dev)" if fast_dev_run else "full training run"
    logger.info(f"Starting {mode} — initialising MLflow …")

    # Prefer MLFLOW_TRACKING_URI so local training can stream to a remote
    # server (e.g. EC2) without editing Hydra configs.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.training.mlflow_uri
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI", uri=tracking_uri)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.training.experiment_name,
        tracking_uri=tracking_uri,
        log_model=True,
    )
    # Eagerly materialise the run so the guardian has a valid run_id before
    # training begins, and register the atexit handler immediately.
    run_id = mlflow_logger.run_id
    run_status = _register_run_guardian(run_id)

    logger.info("Loading data splits and building data module …")
    datamodule = ISICDataModule(cfg.data, cfg.training)

    logger.info("Building model — backbone weights will be downloaded if not cached …", backbone=cfg.model.backbone)
    lit_module = MelanomaLitModule(cfg.model, cfg.training)

    artifacts_dir = Path("artifacts")
    callbacks = [
        EarlyStopping(
            monitor="val/auroc",
            mode="max",
            patience=cfg.training.early_stopping_patience,
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="val/auroc",
            mode="max",
            save_top_k=3,
            filename="epoch={epoch}-auroc={val/auroc:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ThresholdCalibrationCallback(
            target_sensitivity=0.80,
            output_path=str(artifacts_dir / "threshold.json"),
        ),
    ]

    deterministic: bool = cfg.training.get("deterministic", True)
    # cuDNN benchmarking picks the fastest convolution algorithm per input shape, but
    # the choice is nondeterministic, so it is mutually exclusive with reproducible runs.
    torch.backends.cudnn.benchmark = not deterministic

    logger.info("Starting trainer …", fast_dev_run=fast_dev_run, deterministic=deterministic)
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=10,
        fast_dev_run=fast_dev_run,
        deterministic=deterministic,
    )

    peak_auroc: float | None = None
    with mlflow.start_run(run_id=run_id):
        # Log the full resolved config as an artifact for complete reproducibility
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(
            {k: v for k, v in resolved_cfg.get("training", {}).items()}  # type: ignore[union-attr]
        )
        mlflow.log_dict(resolved_cfg, "hydra_config.json")  # type: ignore[arg-type]

        trainer.fit(lit_module, datamodule)
        # Fit succeeded — mark FINISHED even if artifact upload below fails.
        # The atexit guardian otherwise records FAILED for any unclean exit.
        run_status["success"] = True
        logger.info("Trainer.fit complete — saving artifacts …")

        # MLflow overview shows the latest value per metric key; rewrite val/*
        # and epoch to the peak-AUROC epoch so the UI matches ModelCheckpoint.
        try:
            peak_auroc = log_peak_val_metrics(run_id)
        except Exception as exc:
            logger.warning("Could not rewrite MLflow metrics to peak AUROC", error=str(exc))
            peak_auroc = None

        if not fast_dev_run:
            best_ckpt = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
            logger.info("Best checkpoint", path=best_ckpt)
            try:
                # MLflow 3 defaults to serialization_format="pt2", which requires an
                # input_example for torch.export tracing. Our multimodal forward
                # (image, metadata) is awkward to export that way, and serving loads
                # via mlflow.pytorch.load_model — pickle remains the compatible path.
                model_info = mlflow.pytorch.log_model(
                    lit_module.model,
                    name="model",
                    registered_model_name="melanoma-classifier",
                    serialization_format="pickle",
                )
                # Keep a registry alias for the MLflow UI; the API selects
                # models by highest val/auroc, not by this alias.
                registered_version = getattr(model_info, "registered_model_version", None)
                if registered_version is not None:
                    try:
                        mlflow.MlflowClient().set_registered_model_alias(
                            "melanoma-classifier",
                            "champion",
                            str(registered_version),
                        )
                        logger.info(
                            "Set melanoma-classifier@champion",
                            version=str(registered_version),
                        )
                    except Exception as alias_exc:
                        logger.warning(
                            "Could not set champion alias",
                            error=str(alias_exc),
                        )
                if (artifacts_dir / "threshold.json").exists():
                    mlflow.log_artifact(str(artifacts_dir / "threshold.json"))
            except Exception as exc:
                logger.warning(
                    "Post-training artifact logging failed — checkpoints on disk are still valid",
                    error=str(exc),
                    best_checkpoint=best_ckpt,
                )

    best_auroc = peak_auroc
    if best_auroc is None:
        ckpt_best = getattr(trainer.checkpoint_callback, "best_model_score", None)
        if ckpt_best is not None:
            best_auroc = float(ckpt_best)
        else:
            best_auroc = float(trainer.callback_metrics.get("val/auroc", 0.0))
    logger.info("Training complete", best_val_auroc=float(best_auroc))
    return float(best_auroc)


if __name__ == "__main__":
    train()
