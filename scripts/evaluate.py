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
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from cancer_detection.data.datamodule import ISICDataModule
from cancer_detection.evaluation.calibration import (
    expected_calibration_error,
    reliability_diagram_data,
)
from cancer_detection.evaluation.metrics import (
    bootstrap_ci,
    partial_auc,
    roc_curve_points,
    threshold_sweep,
)
from cancer_detection.serving.ood import EmbeddingOODDetector
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


def _relative_to_project_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def resolve_threshold(
    explicit: float | None,
    threshold_path: Path,
    ckpt_run_id: str | None = None,
) -> float:
    """Decide which decision threshold to score at.

    Priority: explicit CLI value, then the calibrated threshold written by
    ThresholdCalibrationCallback, then 0.5. The calibrated value matters a lot
    here — at a 1.76% positive rate, 0.5 buys specificity at the cost of missing
    most of the cancers, which is the wrong trade for a screening model.

    When ckpt_run_id is provided and threshold.json records a different run_id,
    a loud warning is emitted so callers know they may be scoring at a stale
    threshold written by a different training run.
    """
    if explicit is not None:
        logger.info("Using threshold from --threshold", threshold=explicit)
        return explicit

    if threshold_path.exists():
        payload = json.loads(threshold_path.read_text())
        threshold = float(payload["threshold"])
        saved_run_id = payload.get("run_id")

        if ckpt_run_id and saved_run_id and saved_run_id != ckpt_run_id:
            logger.warning(
                "STALE THRESHOLD WARNING: artifacts/threshold.json was written by a "
                "different training run than the checkpoint being evaluated. Pass "
                "--threshold <value> to score at the correct calibrated threshold.",
                threshold_run_id=saved_run_id,
                checkpoint_run_id=ckpt_run_id,
                threshold=threshold,
            )
            print(
                f"\n  *** WARNING: threshold.json belongs to run {saved_run_id[:8]}…, "
                f"but checkpoint belongs to run {ckpt_run_id[:8]}…\n"
                f"  Use --threshold {payload['threshold']:.6f} if this is intentional, "
                f"or re-run threshold calibration for the correct checkpoint.\n"
            )

        logger.info(
            "Using calibrated threshold",
            threshold=threshold,
            target_sensitivity=payload.get("target_sensitivity"),
            source=str(threshold_path),
        )
        return threshold

    logger.warning(
        "No calibrated threshold found — falling back to 0.5", searched=str(threshold_path)
    )
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


def infer_val_auroc(ckpt_path: Path) -> float | None:
    """Extract val AUROC from checkpoint filename, e.g. epoch=2-auroc=0.9220.ckpt."""
    match = _AUROC_IN_NAME.search(ckpt_path.name)
    return float(match.group(1)) if match else None


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

    ckpt_run_id = infer_run_id(ckpt_path)

    # weights_only=False because save_hyperparameters() pickles the OmegaConf configs
    # into the checkpoint, and PyTorch >=2.6 refuses those under the default weights-only
    # unpickler. Safe here: these checkpoints are written by this repo's own train.py.
    lit_module = MelanomaLitModule.load_from_checkpoint(
        str(ckpt_path), map_location="cpu", weights_only=False
    )
    lit_module.test_threshold = resolve_threshold(
        args.threshold,
        Path(args.threshold_file),
        ckpt_run_id=ckpt_run_id,
    )

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

    y_true = lit_module.test_labels
    y_prob = lit_module.test_probs

    # ---- Calibration --------------------------------------------------------
    ece = float(expected_calibration_error(y_true, y_prob, n_bins=10))
    reliability = reliability_diagram_data(y_true, y_prob, n_bins=10)
    logger.info("Calibration computed", ece=ece)

    # ---- Bootstrap CIs ------------------------------------------------------
    print("\nComputing bootstrap confidence intervals (n=2000, stratified)…")
    auroc_ci = bootstrap_ci(y_true, y_prob, roc_auc_score)
    pauc_fn = lambda yt, yp: partial_auc(yt, yp, min_tpr=0.80)  # noqa: E731
    pauc_ci = bootstrap_ci(y_true, y_prob, pauc_fn)

    threshold = float(metrics["threshold"])

    def _sensitivity(yt: np.ndarray, yp: np.ndarray) -> float:
        pred = (yp >= threshold).astype(int)
        tp = int(((yt == 1) & (pred == 1)).sum())
        fn = int(((yt == 1) & (pred == 0)).sum())
        return tp / (tp + fn + 1e-8)

    def _specificity(yt: np.ndarray, yp: np.ndarray) -> float:
        pred = (yp >= threshold).astype(int)
        tn = int(((yt == 0) & (pred == 0)).sum())
        fp = int(((yt == 0) & (pred == 1)).sum())
        return tn / (tn + fp + 1e-8)

    sens_ci = bootstrap_ci(y_true, y_prob, _sensitivity)
    spec_ci = bootstrap_ci(y_true, y_prob, _specificity)

    ci: dict[str, dict[str, float]] = {
        "auroc": {"lo": auroc_ci[0], "hi": auroc_ci[1]},
        "pauc": {"lo": pauc_ci[0], "hi": pauc_ci[1]},
        "sensitivity": {"lo": sens_ci[0], "hi": sens_ci[1]},
        "specificity": {"lo": spec_ci[0], "hi": spec_ci[1]},
    }

    # ---- ROC curve and threshold sweep --------------------------------------
    roc = roc_curve_points(y_true, y_prob, max_points=200)
    sweep = threshold_sweep(y_true, y_prob, n_points=100)

    # ---- Backbone and val AUROC from checkpoint path ------------------------
    backbone: str | None = None
    try:
        hparams = lit_module.hparams  # type: ignore[attr-defined]
        backbone = getattr(getattr(hparams, "model_cfg", None), "backbone", None)
    except Exception as exc:
        logger.warning("Could not read backbone name from checkpoint hparams", error=str(exc))
    val_auroc = infer_val_auroc(ckpt_path)

    # ---- Assemble and write the full payload --------------------------------
    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    tn = int(metrics["tn"])
    fn = int(metrics["fn"])
    n_test = tp + fp + tn + fn
    n_positive = tp + fn

    age_bands = _age_band_metrics(cfg, y_prob, threshold)

    # The sensitivity the threshold was calibrated to hit on val — reported so the
    # printed operating point names its own policy instead of a stale literal.
    target_sensitivity: float | None = None
    threshold_file = Path(args.threshold_file)
    if args.threshold is None and threshold_file.exists():
        try:
            target_sensitivity = float(json.loads(threshold_file.read_text())["target_sensitivity"])
        except (KeyError, ValueError) as exc:
            logger.warning("Could not read target_sensitivity", error=str(exc))

    payload: dict = {
        **metrics,
        "ece": ece,
        "age_bands": age_bands,
        "ppv": metrics.get("ppv"),
        "npv": metrics.get("npv"),
        "n_test": n_test,
        "n_positive": n_positive,
        "prevalence": n_positive / (n_test + 1e-8),
        "val_auroc": val_auroc,
        "backbone": backbone,
        "ci": ci,
        "roc": roc,
        "reliability": reliability,
        "sweep": sweep,
        # Repo-relative — /test-metrics serves this payload verbatim to any caller,
        # and an absolute local path is both useless remotely and an unintended
        # disclosure of local machine layout.
        "checkpoint": _relative_to_project_root(ckpt_path),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "test_metrics.json"
    results_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote test metrics", path=str(results_path))

    if args.save_predictions:
        _save_predictions(cfg, lit_module)

    # ---- OOD detector: fit once here and log it, so a bare API container (no
    # dataset on disk) can download and load it instead of silently running with
    # OOD detection disabled.
    ood_path: Path | None = None
    try:
        ood_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ood_detector = EmbeddingOODDetector.fit(
            lit_module.model,
            Path(cfg.data.processed_dir) / "train.csv",
            cfg.data.image_dir,
            ood_device,
        )
        if ood_detector is not None:
            ood_path = RESULTS_DIR / "ood_detector.npz"
            ood_detector.save(ood_path)
            logger.info("OOD detector fitted and saved", path=str(ood_path))
    except Exception as exc:
        logger.warning("OOD detector fit failed — API will fit its own at startup", error=str(exc))

    run_id = args.run_id or ckpt_run_id
    _log_to_mlflow(cfg, metrics, run_id, ece, ci, results_path, ood_path)
    _print_report(metrics, ckpt_path, ci, ece, age_bands, target_sensitivity)
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


_AGE_BANDS: tuple[tuple[int, int], ...] = ((0, 45), (45, 65), (65, 200))


def _age_band_metrics(
    cfg: DictConfig,
    y_prob: np.ndarray,
    threshold: float,
) -> list[dict] | None:
    """Break sensitivity/specificity out by patient age band.

    A single pooled sensitivity hides the model's age prior: on run 739da85a the
    mean probability assigned to a *malignant* case was 0.21 under 45 against 0.60
    for 45-65, and five of seven missed melanomas were under 45. Training
    prevalence genuinely rises with age (0.9% under 45 vs 4.8% over 65), so the
    prior is not a bug — but pooling it away means nobody sees the cases it costs.

    Each band carries its own ``n_positive`` because the bands are small: at 30
    test positives a band sensitivity quoted without its denominator invites
    reading noise as signal.
    """
    test_df = pd.read_csv(Path(cfg.data.processed_dir) / "test.csv")
    if y_prob is None or len(y_prob) != len(test_df):
        # Same shuffle=False alignment assumption as _save_predictions.
        logger.warning(
            "Prediction vector does not match test.csv — skipping age-band metrics",
            n_probs=0 if y_prob is None else len(y_prob),
            n_rows=len(test_df),
        )
        return None
    if "age_approx" not in test_df.columns:
        return None

    labels = test_df["target"].to_numpy()
    ages = test_df["age_approx"].to_numpy(dtype=float)
    pred = (y_prob >= threshold).astype(int)

    bands: list[dict] = []
    for lo, hi in _AGE_BANDS:
        in_band = (ages > lo) & (ages <= hi) & ~np.isnan(ages)
        pos = in_band & (labels == 1)
        neg = in_band & (labels == 0)
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        bands.append(
            {
                "band": f"{lo}-{hi}" if hi < 200 else f"{lo}+",
                "n_positive": n_pos,
                "n_negative": n_neg,
                "tp": int((pos & (pred == 1)).sum()),
                "fn": int((pos & (pred == 0)).sum()),
                "sensitivity": float((pos & (pred == 1)).sum() / n_pos) if n_pos else None,
                "specificity": float((neg & (pred == 0)).sum() / n_neg) if n_neg else None,
                "mean_prob_malignant": float(y_prob[pos].mean()) if n_pos else None,
                "mean_prob_benign": float(y_prob[neg].mean()) if n_neg else None,
            }
        )
    return bands


def _log_to_mlflow(
    cfg: DictConfig,
    metrics: dict[str, float],
    run_id: str | None,
    ece: float,
    ci: dict[str, dict[str, float]],
    results_path: Path,
    ood_path: Path | None = None,
) -> None:
    """Attach test metrics to the originating training run when we can identify it."""
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.training.mlflow_uri
        mlflow.set_tracking_uri(tracking_uri)

        scalar_extras: dict[str, float] = {
            "test/ece": ece,
        }
        for metric_name, bounds in ci.items():
            scalar_extras[f"test/{metric_name}_ci_lo"] = bounds["lo"]
            scalar_extras[f"test/{metric_name}_ci_hi"] = bounds["hi"]
        # ppv and npv may already be in metrics if compute_metrics was extended
        for key in ("ppv", "npv"):
            if key in metrics:
                scalar_extras[f"test/{key}"] = float(metrics[key])

        all_scalars = {f"test/{k}": float(v) for k, v in metrics.items()}
        all_scalars.update(scalar_extras)

        def _log_artifacts() -> None:
            mlflow.log_metrics(all_scalars)
            mlflow.log_artifact(str(results_path))
            if ood_path is not None and ood_path.exists():
                mlflow.log_artifact(str(ood_path))
                json_sidecar = ood_path.with_suffix(".json")
                if json_sidecar.exists():
                    mlflow.log_artifact(str(json_sidecar))

        if run_id:
            logger.info("Logging test metrics to originating run", run_id=run_id)
            with mlflow.start_run(run_id=run_id):
                _log_artifacts()
        else:
            mlflow.set_experiment(cfg.training.experiment_name)
            with mlflow.start_run(run_name="test-evaluation"):
                _log_artifacts()
    except Exception as exc:
        logger.warning("Could not log to MLflow — metrics still saved locally", error=str(exc))


def _print_report(
    metrics: dict[str, float],
    ckpt_path: Path,
    ci: dict[str, dict[str, float]],
    ece: float,
    age_bands: list[dict] | None = None,
    target_sensitivity: float | None = None,
) -> None:
    tp, fp = int(metrics["tp"]), int(metrics["fp"])
    tn, fn = int(metrics["tn"]), int(metrics["fn"])
    n = tp + fp + tn + fn
    n_pos = tp + fn
    ppv = metrics.get("ppv", tp / (tp + fp + 1e-8))
    npv = metrics.get("npv", tn / (tn + fn + 1e-8))

    def ci_str(key: str) -> str:
        if key not in ci:
            return ""
        return f"  95% CI [{ci[key]['lo']:.4f}, {ci[key]['hi']:.4f}]"

    print("\n" + "=" * 68)
    print("  HELD-OUT TEST RESULTS  (data/processed/test.csv)")
    print("=" * 68)
    print(f"  checkpoint    {ckpt_path.name}")
    target_str = (
        f"≥{target_sensitivity * 100:.0f}% sensitivity on val"
        if target_sensitivity is not None
        else "sensitivity target unknown"
    )
    print(f"  threshold     {metrics['threshold']:.6f}  (calibrated at {target_str})")
    print(f"  n_test        {n}  ({n_pos} malignant, {n - n_pos} benign)")
    print(f"  prevalence    {n_pos / (n + 1e-8) * 100:.2f}%")
    print("-" * 68)
    print(f"  AUROC         {metrics['auroc']:.4f}{ci_str('auroc')}")
    print(f"  pAUC (>80%)   {metrics['pauc']:.4f}{ci_str('pauc')}")
    print(
        f"  Sensitivity   {metrics['sensitivity']:.4f}  ({tp}/{tp + fn} cancers){ci_str('sensitivity')}"
    )
    print(
        f"  Specificity   {metrics['specificity']:.4f}  ({fp} FP of {fp + tn} benign){ci_str('specificity')}"
    )
    print(f"  PPV           {ppv:.4f}  (precision — {tp} TP vs {fp} FP)")
    print(f"  NPV           {npv:.4f}")
    print(f"  ECE           {ece:.4f}  (calibration error; lower is better)")
    print(
        f"  F1            {metrics['f1']:.4f}  (depressed by {n_pos / (n + 1e-8) * 100:.1f}% prevalence)"
    )
    print("-" * 68)
    print(f"  TP {tp:5d}   FP {fp:5d}")
    print(f"  FN {fn:5d}   TN {tn:5d}")
    print("=" * 68 + "\n")
    print(f"  Clinical framing: catches {tp} of {tp + fn} melanomas at the cost of")
    print(f"  {fp} benign referrals ({fp / (tp + 1e-8):.1f} false alarms per cancer found).\n")

    if age_bands:
        print("  Recall by patient age (small denominators — read with the counts):")
        for b in age_bands:
            if not b["n_positive"]:
                continue
            print(
                f"    {b['band']:>6}  {b['tp']}/{b['n_positive']} melanomas"
                f"   mean p(malignant)={b['mean_prob_malignant']:.3f}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint to evaluate (default: highest val AUROC on disk)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: artifacts/threshold.json)",
    )
    parser.add_argument("--threshold-file", type=str, default=str(DEFAULT_THRESHOLD_PATH))
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run to attach metrics to (default: inferred from ckpt path)",
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also write per-image probabilities to artifacts/test_predictions.csv",
    )
    parser.add_argument(
        "--override", nargs="*", default=[], help="Hydra config overrides, e.g. data.num_workers=4"
    )
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
