"""Quantify how far predictions drift when images leave the ISIC acquisition pipeline.

The model is trained on dermoscopy JPEGs that were all captured with contact
dermatoscopes and then downsampled once by scripts/resize_images.py. An image
pulled off the web has been through a different camera, a different resize and a
different JPEG encoder. This script measures how much of the model's output is
driven by the lesion and how much by that acquisition fingerprint.

Four groups are scored:
    train      -- images the model was fit on (memorisation ceiling)
    test       -- held-out images, unmodified
    degraded   -- the *same* test images, re-encoded to look web-sourced
    custom     -- any loose image files passed with --images

If `degraded` collapses toward zero relative to `test`, the model depends on the
acquisition pipeline rather than the lesion, and web uploads will read benign.

Examples:
    # Score the checkpoint the API should be serving
    python scripts/diagnose.py --model "1/<run_id>/checkpoints/epoch=2-auroc=0.9220.ckpt"

    # Compare against whatever MLflow actually has
    python scripts/diagnose.py --model mlartifacts/1/models/<model_id>/artifacts

    # Include your own downloaded melanoma images
    python scripts/diagnose.py --model <...> --images path/to/web_images
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_detection.data.metadata import MetadataEncoder  # noqa: E402
from cancer_detection.data.transforms import get_val_transforms  # noqa: E402
from cancer_detection.models.classifier import MelanomaClassifier  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "threshold.json"


def load_model(spec: str, device: torch.device) -> MelanomaClassifier:
    """Load either a Lightning checkpoint or a logged MLflow model."""
    if spec.endswith(".ckpt"):
        from cancer_detection.training.lit_module import MelanomaLitModule

        # weights_only=False because save_hyperparameters() pickles OmegaConf configs
        # into the checkpoint; these files are written by this repo's own train.py.
        lit = MelanomaLitModule.load_from_checkpoint(spec, map_location="cpu", weights_only=False)
        model = lit.model
    else:
        import mlflow.pytorch

        model = mlflow.pytorch.load_model(spec)

    return model.eval().to(device)


def degrade_to_web(image: np.ndarray, long_edge: int = 500, quality: int = 60) -> np.ndarray:
    """Re-encode an image so it resembles one downloaded from a web page.

    Square-crops, downsamples hard, then round-trips through a lossy JPEG encoder.
    None of these touch the lesion itself, so a model reading the lesion should be
    largely unaffected.
    """
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = image[top : top + side, left : left + side]

    pil = Image.fromarray(square).resize((long_edge, long_edge), Image.BILINEAR)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("RGB"))


def score_images(
    model: MelanomaClassifier,
    image_paths: list[Path],
    metadata_rows: list[pd.Series],
    device: torch.device,
    degrade: bool = False,
    batch_size: int = 16,
) -> np.ndarray:
    """Return sigmoid probabilities for a list of images, single pass, no TTA."""
    transform = get_val_transforms()
    encoder = MetadataEncoder()
    probabilities: list[float] = []

    for start in range(0, len(image_paths), batch_size):
        chunk_paths = image_paths[start : start + batch_size]
        chunk_meta = metadata_rows[start : start + batch_size]

        images, metas = [], []
        for path, row in zip(chunk_paths, chunk_meta):
            array = np.array(Image.open(path).convert("RGB"))
            if degrade:
                array = degrade_to_web(array)
            images.append(transform(image=array)["image"])
            metas.append(encoder.encode(row))

        image_batch = torch.stack(images).to(device)
        meta_batch = torch.stack(metas).to(device)
        with torch.no_grad():
            logits = model(image_batch, meta_batch)
        probabilities.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())

        print(f"  scored {min(start + batch_size, len(image_paths))}/{len(image_paths)}", end="\r")

    print(" " * 40, end="\r")
    return np.array(probabilities)


def sample_split(
    csv_path: Path, image_dir: Path, n_per_class: int, seed: int
) -> tuple[list[Path], list[pd.Series], np.ndarray]:
    """Draw a class-balanced sample so positives are actually represented."""
    df = pd.read_csv(csv_path)
    positives = df[df["target"] == 1]
    negatives = df[df["target"] == 0]

    positives = positives.sample(min(n_per_class, len(positives)), random_state=seed)
    negatives = negatives.sample(min(n_per_class, len(negatives)), random_state=seed)
    sample = pd.concat([positives, negatives])

    paths = [image_dir / f"{name}.jpg" for name in sample["image_name"]]
    rows = [row for _, row in sample.iterrows()]
    return paths, rows, sample["target"].to_numpy()


def describe(
    name: str, probs: np.ndarray, threshold: float, labels: np.ndarray | None = None
) -> dict:
    """Print and return the distribution summary for one group."""
    if len(probs) == 0:
        print(f"{name:<12} (no images)")
        return {}

    stats = {
        "n": int(len(probs)),
        "median": float(np.median(probs)),
        "mean": float(probs.mean()),
        "p90": float(np.quantile(probs, 0.90)),
        "max": float(probs.max()),
        "frac_above_threshold": float((probs >= threshold).mean()),
    }
    print(
        f"{name:<12} n={stats['n']:<5} median={stats['median']:.4f}  "
        f"mean={stats['mean']:.4f}  p90={stats['p90']:.4f}  max={stats['max']:.4f}  "
        f"called malignant: {stats['frac_above_threshold'] * 100:.1f}%"
    )

    if labels is not None and len(set(labels.tolist())) > 1:
        pos, neg = probs[labels == 1], probs[labels == 0]
        stats["median_malignant"] = float(np.median(pos))
        stats["median_benign"] = float(np.median(neg))
        stats["sensitivity"] = float((pos >= threshold).mean())
        stats["specificity"] = float((neg < threshold).mean())
        print(
            f"{'':<12}   malignant median={stats['median_malignant']:.4f} "
            f"(sens {stats['sensitivity'] * 100:.0f}%)   "
            f"benign median={stats['median_benign']:.4f} "
            f"(spec {stats['specificity'] * 100:.0f}%)"
        )
    return stats


def resolve_threshold(explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    if DEFAULT_THRESHOLD_PATH.exists():
        return float(json.loads(DEFAULT_THRESHOLD_PATH.read_text())["threshold"])
    return 0.5


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, help="Path to a .ckpt file or an MLflow model directory/URI"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: artifacts/threshold.json)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Directory of loose images to score (e.g. web downloads)",
    )
    parser.add_argument(
        "--image-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "jpeg_448"
    )
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument(
        "--n-per-class", type=int, default=60, help="Images sampled per class per split"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "diagnostics.json"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    threshold = resolve_threshold(args.threshold)

    print(f"\nModel:     {args.model}")
    print(f"Device:    {device}")
    print(f"Threshold: {threshold:.6f}\n")

    model = load_model(args.model, device)
    results: dict[str, dict] = {}

    print("=" * 78)
    print("  PROBABILITY DISTRIBUTION BY IMAGE SOURCE")
    print("=" * 78)

    train_paths, train_rows, train_labels = sample_split(
        args.processed_dir / "train.csv", args.image_dir, args.n_per_class, args.seed
    )
    results["train"] = describe(
        "train", score_images(model, train_paths, train_rows, device), threshold, train_labels
    )

    test_paths, test_rows, test_labels = sample_split(
        args.processed_dir / "test.csv", args.image_dir, args.n_per_class, args.seed
    )
    test_probs = score_images(model, test_paths, test_rows, device)
    results["test"] = describe("test", test_probs, threshold, test_labels)

    degraded_probs = score_images(model, test_paths, test_rows, device, degrade=True)
    results["degraded"] = describe("degraded", degraded_probs, threshold, test_labels)

    if args.images is not None and args.images.exists():
        custom_paths = sorted(
            p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if custom_paths:
            default_row = pd.Series(
                {"age_approx": 50.0, "sex": "unknown", "anatom_site_general_challenge": "unknown"}
            )
            custom_probs = score_images(
                model, custom_paths, [default_row] * len(custom_paths), device
            )
            results["custom"] = describe("custom", custom_probs, threshold)
            print()
            for path, prob in zip(custom_paths, custom_probs):
                verdict = "MALIGNANT" if prob >= threshold else "benign"
                print(f"    {path.name:<40} {prob:.4f}  -> {verdict}")

    print("=" * 78)

    if results.get("test") and results.get("degraded"):
        before, after = results["test"]["median"], results["degraded"]["median"]
        shift = after / before if before > 0 else float("nan")
        print(
            f"\nWeb-style re-encoding moves the median probability from {before:.4f} "
            f"to {after:.4f} ({shift:.2f}x)."
        )
        sens_before = results["test"].get("sensitivity")
        sens_after = results["degraded"].get("sensitivity")
        if sens_before is not None and sens_after is not None:
            print(
                f"Sensitivity on the same lesions drops from {sens_before * 100:.0f}% "
                f"to {sens_after * 100:.0f}%."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"model": args.model, "threshold": threshold, "groups": results}, indent=2)
    )
    print(f"\nWritten to {args.output}\n")


if __name__ == "__main__":
    main()
