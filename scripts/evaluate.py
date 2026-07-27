"""Evaluate a trained checkpoint on the held-out test split.

Unlike scripts/train.py, this never touches train.csv or val.csv — it scores
data/processed/test.csv, which no part of training or threshold calibration has
seen. Those numbers are the honest estimate of deployed performance.

Examples:
    # Evaluate the highest-AUROC checkpoint found on disk:
    python scripts/evaluate.py

    # Evaluate a specific checkpoint:
    python scripts/evaluate.py --ckpt "1/f644.../checkpoints/epoch=2-auroc=0.9156.ckpt"

    # Override the decision threshold (default: artifacts/threshold.json):
    python scripts/evaluate.py --threshold 0.5

    # Pass Hydra overrides through to the data/model config:
    python scripts/evaluate.py --override data.num_workers=4 training.batch_size=32
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Windows terminals default to CP1252 which cannot encode emoji characters that
# MLflow prints to stdout (e.g. the 🏃 in the "View run at: …" summary line).
# Reconfigure both streams to UTF-8 before any library code touches them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import pandas as pd
from hydra import compose, initialize_config_dir
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from cancer_detection.data.datamodule import ISICDataModule
from cancer_detection.training.lit_module import MelanomaLitModule
from cancer_detection.utils.logger import configure_logging, get_logger
from cancer_detection.utils.seed import set_seed

_log_path = configure_logging(name="evaluate")
logger = get_logger(__name__)
if _log_path is not None:
    logger.info("Writing logs to file", path=str(_log_path))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "threshold.json"
RESULTS_DIR = PROJECT_ROOT / "artifacts"

# ModelCheckpoint writes "epoch=2-auroc=0.9156.ckpt"; the run id is the directory
# two levels up, e.g. "1/<run_id>/checkpoints/<file>.ckpt".
_AUROC_IN_NAME = re.compile(r"auroc=(\d+\.\d+)")


def find_best_checkpoint(search_root: Path) -> Path:
    """Return the checkpoint with the highest val AUROC encoded in its filename.

    Ignores mlartifacts/, which holds MLflow's duplicate copies of the same files.
    """
    candidates = [
        p
        for p in search_root.rglob("*.ckpt")
        if ".venv" not in p.parts and "mlartifacts" not in p.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt files found under {search_root}. Train a model first "
            f"(python scripts/train.py) or pass --ckpt explicitly."
        )

    def score(path: Path) -> float:
        match = _AUROC_IN_NAME.search(path.name)
        return float(match.group(1)) if match else -1.0

    return max(candidates, key=score)


def resolve_threshold(explicit: float | None, threshold_path: Path) -> float:
    """Decide which decision threshold to score at.

    Priority: explicit CLI value, then the calibrated threshold written by
    ThresholdCalibrationCallback, then 0.5. The calibrated value matters a lot
    here — at a 1.76% positive rate, 0.5 buys specificity at the cost of missing
    most of the cancers, which is the wrong trade for a screening model.
    """
    if explicit is not None:
        logger.info("Using threshold from --threshold", threshold=explicit)
        return explicit

    if threshold_path.exists():
        payload = json.loads(threshold_path.read_text())
        threshold = float(payload["threshold"])
        logger.info(
            "Using calibrated threshold",
            threshold=threshold,
            target_sensitivity=payload.get("target_sensitivity"),
            source=str(threshold_path),
        )
        return threshold

    logger.warning("No calibrated threshold found — falling back to 0.5", searched=str(threshold_path))
    return 0.5


def infer_run_id(ckpt_path: Path) -> str | None:
    """Recover the MLflow run id from a checkpoint path, if it looks like one.

    Lightning writes checkpoints to <experiment_id>/<run_id>/checkpoints/, so the
    grandparent directory name is the run id for MLflow-logged runs.
    """
    if ckpt_path.parent.name != "checkpoints":
        return None
    candidate = ckpt_path.parent.parent.name
    return candidate if re.fullmatch(r"[0-9a-f]{32}", candidate) else None


def load_config(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    cfg = load_config(args.override)
    set_seed(cfg.training.seed)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        print(f"\nEvaluating checkpoint: {ckpt_path}\n")
    else:
        ckpt_path = find_best_checkpoint(PROJECT_ROOT)
        auroc_match = _AUROC_IN_NAME.search(ckpt_path.name)
        auroc_str = auroc_match.group(1) if auroc_match else "unknown"
        print(f"\nBest on-disk checkpoint (val AUROC={auroc_str}):\n  {ckpt_path}\n")
    logger.info("Evaluating checkpoint", path=str(ckpt_path))


    # weights_only=False because save_hyperparameters() pickles the OmegaConf configs
    # into the checkpoint, and PyTorch >=2.6 refuses those under the default weights-only
    # unpickler. Safe here: these checkpoints are written by this repo's own train.py.
    lit_module = MelanomaLitModule.load_from_checkpoint(
        str(ckpt_path), map_location="cpu", weights_only=False
    )
    lit_module.test_threshold = resolve_threshold(args.threshold, Path(args.threshold_file))

    datamodule = ISICDataModule(cfg.data, cfg.training)

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=cfg.training.precision,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(lit_module, datamodule=datamodule)

    metrics = lit_module.test_metrics
    if not metrics:
        raise RuntimeError("Test loop produced no metrics — is data/processed/test.csv empty?")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "test_metrics.json"
    payload = {**metrics, "checkpoint": str(ckpt_path)}
    results_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote test metrics", path=str(results_path))

    if args.save_predictions:
        _save_predictions(cfg, lit_module)

    _log_to_mlflow(cfg, metrics, args.run_id or infer_run_id(ckpt_path))
    _print_report(metrics, ckpt_path)
    return metrics


def _save_predictions(cfg: DictConfig, lit_module: MelanomaLitModule) -> None:
    """Dump per-image probabilities so thresholds can be re-swept without re-inference."""
    probs = lit_module.test_probs
    test_df = pd.read_csv(Path(cfg.data.processed_dir) / "test.csv")
    if probs is None or len(probs) != len(test_df):
        # Row order only lines up because test_dataloader uses shuffle=False; bail out
        # rather than emit a misaligned file if that ever stops holding.
        logger.warning(
            "Prediction vector does not match test.csv — skipping predictions CSV",
            n_probs=0 if probs is None else len(probs),
            n_rows=len(test_df),
        )
        return
    test_df["probability"] = probs
    test_df["predicted"] = (probs >= lit_module.test_threshold).astype(int)
    out = RESULTS_DIR / "test_predictions.csv"
    test_df.to_csv(out, index=False)
    logger.info("Wrote per-image predictions", path=str(out))


def _log_to_mlflow(cfg: DictConfig, metrics: dict[str, float], run_id: str | None) -> None:
    """Attach test metrics to the originating training run when we can identify it."""
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.training.mlflow_uri
        mlflow.set_tracking_uri(tracking_uri)
        if run_id:
            logger.info("Logging test metrics to originating run", run_id=run_id)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics({f"test/{k}": float(v) for k, v in metrics.items()})
        else:
            mlflow.set_experiment(cfg.training.experiment_name)
            with mlflow.start_run(run_name="test-evaluation"):
                mlflow.log_metrics({f"test/{k}": float(v) for k, v in metrics.items()})
    except Exception as exc:
        logger.warning("Could not log to MLflow — metrics still saved locally", error=str(exc))


def _print_report(metrics: dict[str, float], ckpt_path: Path) -> None:
    tp, fp = int(metrics["tp"]), int(metrics["fp"])
    tn, fn = int(metrics["tn"]), int(metrics["fn"])
    print("\n" + "=" * 58)
    print("  HELD-OUT TEST RESULTS  (data/processed/test.csv)")
    print("=" * 58)
    print(f"  checkpoint    {ckpt_path.name}")
    print(f"  threshold     {metrics['threshold']:.6f}")
    print("-" * 58)
    print(f"  AUROC         {metrics['auroc']:.4f}")
    print(f"  pAUC (>80%)   {metrics['pauc']:.4f}")
    print(f"  F1            {metrics['f1']:.4f}")
    print(f"  Sensitivity   {metrics['sensitivity']:.4f}   ({tp}/{tp + fn} cancers caught)")
    print(f"  Specificity   {metrics['specificity']:.4f}   ({fp} false alarms of {fp + tn} benign)")
    print("-" * 58)
    print(f"  TP {tp:5d}   FP {fp:5d}")
    print(f"  FN {fn:5d}   TN {tn:5d}")
    print("=" * 58 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to evaluate (default: highest val AUROC on disk)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: artifacts/threshold.json)")
    parser.add_argument("--threshold-file", type=str, default=str(DEFAULT_THRESHOLD_PATH))
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run to attach metrics to (default: inferred from ckpt path)")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--save-predictions", action="store_true", help="Also write per-image probabilities to artifacts/test_predictions.csv")
    parser.add_argument("--override", nargs="*", default=[], help="Hydra config overrides, e.g. data.num_workers=4")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
