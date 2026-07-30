"""One-off: republish a Lightning checkpoint as the MLflow model the API loads.

Use this when the currently served artifact is the *final* epoch of a run (the
bug fixed in scripts/train.py) and you want the site to serve the best epoch
without waiting for a full retrain.

Example:
    python scripts/republish_checkpoint.py \\
        --ckpt "1/5c1b857d8b924b10a83cfcf53121d64c/checkpoints/epoch=2-auroc=0.9220.ckpt" \\
        --download-to serving_model
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_detection.training.lit_module import MelanomaLitModule  # noqa: E402
from cancer_detection.training.threshold import calibrate_threshold  # noqa: E402
from cancer_detection.utils.logger import configure_logging, get_logger  # noqa: E402

configure_logging(name="republish")
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Lightning .ckpt to publish")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://18.219.3.159:5000"),
    )
    parser.add_argument("--experiment", default="melanoma-detection")
    parser.add_argument(
        "--val-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "val.csv"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "jpeg_448",
    )
    parser.add_argument(
        "--threshold-out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "threshold.json",
    )
    parser.add_argument(
        "--download-to",
        type=Path,
        default=None,
        help="If set, download the logged model here (e.g. serving_model/) for Docker bake-in",
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=0.80,
        help="Sensitivity the threshold is calibrated to hit on val (see training/default.yaml)",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tta-passes", type=int, default=8)
    parser.add_argument("--run-name", default="republish-best-checkpoint")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint", path=str(args.ckpt), device=str(device))
    lit = MelanomaLitModule.load_from_checkpoint(
        str(args.ckpt), map_location="cpu", weights_only=False
    )
    model = lit.model.eval()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        mlflow.log_param("source_checkpoint", str(args.ckpt))
        mlflow.log_param("republish", True)

        payload = calibrate_threshold(
            model,
            pd.read_csv(args.val_csv),
            args.image_dir,
            device,
            target_sensitivity=args.target_sensitivity,
            output_path=args.threshold_out,
            image_size=args.image_size,
            batch_size=args.batch_size,
            tta_passes=args.tta_passes,
            run_id=run_id,
            checkpoint=str(args.ckpt),
        )
        mlflow.log_metric("calibrated_threshold", payload["threshold"])
        mlflow.log_metric("val/auroc", payload["val_auroc"])
        mlflow.log_metric("val/auroc_tta", payload["val_auroc"])
        mlflow.log_artifact(str(args.threshold_out))

        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="melanoma-classifier",
            serialization_format="pickle",
        )
        logger.info(
            "Published model",
            run_id=run_id,
            model_uri=f"runs:/{run_id}/model",
            registered=getattr(model_info, "registered_model_version", None),
            threshold=payload["threshold"],
            val_auroc=payload["val_auroc"],
        )

        if args.download_to is not None:
            dest = args.download_to
            if dest.exists():
                for child in dest.iterdir():
                    if child.name == ".gitkeep":
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            dest.mkdir(parents=True, exist_ok=True)
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/model",
                dst_path=str(dest),
            )
            # download_artifacts may nest under dest/model — flatten if needed
            nested = Path(local)
            if nested.name == "model" and nested.parent == dest:
                for item in nested.iterdir():
                    target = dest / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(target))
                nested.rmdir()
            logger.info("Model downloaded for Docker bake-in", path=str(dest))

    print(f"\nDone. MODEL_URI=runs:/{run_id}/model")
    print(f"Threshold written to {args.threshold_out}")
    if args.download_to:
        print(f"Baked copy at {args.download_to} — rebuild the API image to ship it.")


if __name__ == "__main__":
    main()
