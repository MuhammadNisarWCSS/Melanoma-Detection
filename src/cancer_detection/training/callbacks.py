from __future__ import annotations

import math

import mlflow

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
    """Log the peak-AUROC epoch's metrics under separate ``best/*`` keys.

    Lightning logs ``val/*`` every validation check, so the last point in that
    series is the final epoch, not necessarily the best one. Rather than append a
    fabricated point to the ``val/auroc`` history (which would make the MLflow
    chart show a decay followed by a jump back up at a step the model was never
    actually at), log the peak under distinct ``best/*`` keys with no step — the
    original ``val/*`` history is left untouched and honest.
    """
    client = mlflow.MlflowClient()
    auroc_hist = client.get_metric_history(run_id, "val/auroc")
    finite = [m for m in auroc_hist if math.isfinite(float(m.value))]
    if not finite:
        logger.warning("No finite val/auroc history — skipping peak metric log", run_id=run_id)
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

    client.log_metric(run_id, "best/val_auroc", peak_auroc)
    if peak_epoch is not None:
        client.log_metric(run_id, "best/epoch", peak_epoch)
    if peak_f1 is not None:
        client.log_metric(run_id, "best/val_f1", peak_f1)
    if peak_loss is not None:
        client.log_metric(run_id, "best/val_loss", peak_loss)

    logger.info(
        "Logged peak-AUROC epoch under best/* keys",
        run_id=run_id,
        peak_epoch=peak_epoch,
        peak_step=peak_step,
        val_auroc=peak_auroc,
        val_f1=peak_f1,
        val_loss=peak_loss,
    )
    return peak_auroc


# Threshold calibration used to live here as a Lightning Callback, but on_fit_end
# fires while the module still holds the *final* epoch's weights. Those are not the
# weights that get deployed, so the saved threshold described a model nobody served.
# scripts/train.py now reloads the best checkpoint first and calls
# cancer_detection.training.threshold.calibrate_threshold explicitly, which also runs
# the same TTA averaging the API uses.
