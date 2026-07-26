from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from mlflow.entities import Metric

from cancer_detection.evaluation.metrics import find_optimal_threshold
from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)


def _metric_at_step(history: list, step: int) -> float | None:
    """Return the metric value logged at ``step``, or the latest value at/before it."""
    exact = next((m.value for m in history if m.step == step), None)
    if exact is not None and math.isfinite(float(exact)):
        return float(exact)
    prior = [m for m in history if m.step <= step and math.isfinite(float(m.value))]
    if not prior:
        return None
    return float(max(prior, key=lambda m: m.step).value)


def log_peak_val_metrics(run_id: str) -> float | None:
    """Re-log overview metrics so MLflow shows the peak-AUROC epoch, not the last.

    MLflow's run overview always displays the *latest* value for each metric key.
    Lightning logs ``val/*`` and ``epoch`` every validation epoch, so the overview
    ends on the final epoch. After fit, rewrite those keys with the values from
    the step where ``val/auroc`` was highest (same step index, new timestamp).
    """
    client = mlflow.MlflowClient()
    auroc_hist = client.get_metric_history(run_id, "val/auroc")
    finite = [m for m in auroc_hist if math.isfinite(float(m.value))]
    if not finite:
        logger.warning("No finite val/auroc history — skipping peak metric rewrite", run_id=run_id)
        return None

    peak = max(finite, key=lambda m: float(m.value))
    peak_step = int(peak.step)
    peak_auroc = float(peak.value)

    f1_hist = client.get_metric_history(run_id, "val/f1")
    loss_hist = client.get_metric_history(run_id, "val/loss")
    epoch_hist = client.get_metric_history(run_id, "epoch")

    peak_f1 = _metric_at_step(f1_hist, peak_step)
    peak_loss = _metric_at_step(loss_hist, peak_step)
    peak_epoch = _metric_at_step(epoch_hist, peak_step)

    # MLflow's run overview uses the value at the highest step (not the newest
    # timestamp). Log one past the final training step so peak metrics win.
    overview_step = max(int(m.step) for m in finite) + 1
    ts = int(time.time() * 1000)
    to_log: list[Metric] = [
        Metric(key="val/auroc", value=peak_auroc, timestamp=ts, step=overview_step),
    ]
    if peak_epoch is not None:
        to_log.append(Metric(key="epoch", value=peak_epoch, timestamp=ts, step=overview_step))
    if peak_f1 is not None:
        to_log.append(Metric(key="val/f1", value=peak_f1, timestamp=ts, step=overview_step))
    if peak_loss is not None:
        to_log.append(Metric(key="val/loss", value=peak_loss, timestamp=ts, step=overview_step))

    client.log_batch(run_id=run_id, metrics=to_log)
    logger.info(
        "MLflow overview metrics set to peak-AUROC epoch",
        run_id=run_id,
        peak_epoch=peak_epoch,
        peak_step=peak_step,
        overview_step=overview_step,
        val_auroc=peak_auroc,
        val_f1=peak_f1,
        val_loss=peak_loss,
    )
    return peak_auroc


class ThresholdCalibrationCallback(Callback):
    """At the end of training, find the optimal decision threshold on validation set.

    Runs one full pass over the validation dataloader, collects predicted
    probabilities and ground-truth labels, then calls find_optimal_threshold()
    to select the threshold that achieves target_sensitivity while maximising
    specificity. Saves the threshold as a JSON file and logs it to MLflow.
    """

    def __init__(
        self,
        target_sensitivity: float = 0.80,
        output_path: str = "artifacts/threshold.json",
    ) -> None:
        self.target_sensitivity = target_sensitivity
        self.output_path = Path(output_path)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.fast_dev_run:
            logger.info("Skipping threshold calibration in fast_dev_run mode")
            return
        logger.info("Running threshold calibration on validation set")
        pl_module.eval()
        all_probs: list[float] = []
        all_labels: list[float] = []

        val_loader = trainer.datamodule.val_dataloader()  # type: ignore[union-attr]
        with torch.no_grad():
            for batch in val_loader:
                images, metadata, labels = batch
                images = images.to(pl_module.device)
                metadata = metadata.to(pl_module.device)
                logits = pl_module(images, metadata)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        threshold = find_optimal_threshold(y_true, y_prob, self.target_sensitivity)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threshold": threshold,
            "target_sensitivity": self.target_sensitivity,
        }
        self.output_path.write_text(json.dumps(payload, indent=2))
        logger.info("Threshold calibrated", threshold=threshold)

        try:
            mlflow.log_artifact(str(self.output_path))
            mlflow.log_metric("calibrated_threshold", threshold)
        except Exception:
            pass  # MLflow may not be active during smoke tests
