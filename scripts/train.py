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

from datetime import UTC

import hydra
import mlflow
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from cancer_detection.data.datamodule import ISICDataModule
from cancer_detection.training.callbacks import log_peak_val_metrics
from cancer_detection.training.lit_module import MelanomaLitModule
from cancer_detection.training.threshold import calibrate_threshold
from cancer_detection.utils.logger import configure_logging, get_logger
from cancer_detection.utils.seed import set_seed

_log_path = configure_logging(name="train")
logger = get_logger(__name__)
if _log_path is not None:
    logger.info("Writing logs to file", path=str(_log_path))

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
            from datetime import datetime

            now_ms = int(datetime.now(UTC).timestamp() * 1000)
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
    deterministic: bool = cfg.training.get("deterministic", True)
    set_seed(cfg.training.seed, deterministic=deterministic)
    logger.info("Config", cfg=OmegaConf.to_yaml(cfg))

    fast_dev_run: bool = cfg.training.get("fast_dev_run", False)
    mode = "smoke test (fast_dev)" if fast_dev_run else "full training run"
    logger.info(f"Starting {mode} — initialising MLflow …")

    # Default is hosted EC2 (configs/training/*.yaml). Override with
    # MLFLOW_TRACKING_URI for a local server without editing Hydra configs.
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

    logger.info(
        "Building model — backbone weights will be downloaded if not cached …",
        backbone=cfg.model.backbone,
    )
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
    ]

    # cudnn.deterministic / cudnn.benchmark are already set coherently by set_seed above.
    logger.info("Starting trainer …", fast_dev_run=fast_dev_run, deterministic=deterministic)
    trainer_kwargs: dict = {}
    if not fast_dev_run:
        # 4 validations/epoch instead of 1 — with only 6-8 epochs, a single check per
        # epoch gives too few samples of the val curve to trust the argmax (a previous
        # run's "best" checkpoint was its very first validation ever performed).
        # early_stopping_patience is expressed in units of these checks, not epochs.
        trainer_kwargs["val_check_interval"] = 0.25
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=10,
        fast_dev_run=fast_dev_run,
        deterministic=deterministic,
        **trainer_kwargs,
    )

    peak_auroc: float | None = None
    with mlflow.start_run(run_id=run_id):
        # Log the full resolved config as an artifact for complete reproducibility
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(
            {k: v for k, v in resolved_cfg.get("training", {}).items()}  # type: ignore[union-attr]
        )
        mlflow.log_param("backbone", cfg.model.backbone)
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
            # Store repo-relative in threshold.json / test_metrics.json — /test-metrics
            # serves that file verbatim, and an absolute local path there is both
            # useless to a remote caller and an unintended disclosure of local layout.
            project_root = Path(__file__).resolve().parent.parent
            try:
                checkpoint_ref = str(Path(best_ckpt).resolve().relative_to(project_root))
            except ValueError:
                checkpoint_ref = best_ckpt

            # Lightning leaves the *final* epoch's weights in memory after fit(); it
            # does not restore the best ones. Logging lit_module.model directly would
            # therefore deploy the most overfit epoch of the run while the metrics
            # reported alongside it describe the best epoch.
            best_module = MelanomaLitModule.load_from_checkpoint(
                best_ckpt, map_location="cpu", weights_only=False
            )
            deployable = best_module.model.eval()

            # Calibrate against the deployable weights, through the same 8-pass TTA
            # averaging the API applies, so the threshold matches the served pipeline.
            calibration_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                payload = calibrate_threshold(
                    deployable,
                    pd.read_csv(Path(cfg.data.processed_dir) / "val.csv"),
                    cfg.data.image_dir,
                    calibration_device,
                    target_sensitivity=cfg.training.target_sensitivity,
                    output_path=artifacts_dir / "threshold.json",
                    image_size=cfg.data.image_size,
                    batch_size=cfg.training.batch_size * 2,
                    num_workers=cfg.data.num_workers,
                    run_id=run_id,
                    checkpoint=checkpoint_ref,
                )
                mlflow.log_metric("calibrated_threshold", payload["threshold"])
                mlflow.log_metric("val/auroc_tta", payload["val_auroc"])
            except Exception as exc:
                logger.warning("Threshold calibration failed", error=str(exc))

            try:
                # MLflow 3 defaults to serialization_format="pt2", which requires an
                # input_example for torch.export tracing. Our multimodal forward
                # (image, metadata) is awkward to export that way, and serving loads
                # via mlflow.pytorch.load_model — pickle remains the compatible path.
                model_info = mlflow.pytorch.log_model(
                    deployable,
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
