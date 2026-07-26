"""Measure single-process data loading throughput and project epoch time.

Times the full per-sample path (JPEG decode + augmentation + tensor conversion) so the
input pipeline can be compared against GPU throughput before committing to a long run.

Usage:
    python scripts/benchmark_loader.py
    python scripts/benchmark_loader.py --samples 500 --image-dir data/raw/jpeg/train
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from cancer_detection.data.dataset import ISICDataset
from cancer_detection.data.metadata import MetadataEncoder
from cancer_detection.data.transforms import get_train_transforms


def benchmark(processed_dir: Path, image_dir: Path, image_size: int, samples: int, batch_size: int) -> None:
    df = pd.read_csv(processed_dir / "train.csv")
    dataset = ISICDataset(df, image_dir, get_train_transforms(image_size), MetadataEncoder())

    n = min(samples, len(dataset))
    print(f"Timing {n} samples from {image_dir} at {image_size}px …")

    dataset[0]  # warm the OS file cache and any lazy imports

    start = time.perf_counter()
    for i in range(n):
        dataset[i]
    elapsed = time.perf_counter() - start

    per_sample_ms = elapsed / n * 1000
    rate = n / elapsed
    steps = len(df) // batch_size
    epoch_min = len(df) / rate / 60

    print(f"\n  {per_sample_ms:6.2f} ms per sample")
    print(f"  {rate:6.1f} samples/sec single-process")
    print(f"\nProjected for {len(df):,} training images ({steps:,} steps at batch {batch_size}):")
    print(f"  {epoch_min:.1f} min per epoch if data loading is the bottleneck")


def profile_stages(processed_dir: Path, image_dir: Path, image_size: int, samples: int) -> None:
    """Time JPEG decode and each augmentation separately to locate the bottleneck."""
    import albumentations as A
    import numpy as np
    from PIL import Image

    df = pd.read_csv(processed_dir / "train.csv")
    paths = [image_dir / f"{name}.jpg" for name in df["image_name"].head(samples)]

    start = time.perf_counter()
    images = [np.array(Image.open(p).convert("RGB")) for p in paths]
    decode_ms = (time.perf_counter() - start) / len(paths) * 1000
    print(f"  {'decode':<22} {decode_ms:7.2f} ms")

    stages: list[tuple[str, A.BasicTransform]] = [
        ("RandomResizedCrop", A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0))),
        ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("VerticalFlip", A.VerticalFlip(p=1.0)),
        ("RandomRotate90", A.RandomRotate90(p=1.0)),
        ("ShiftScaleRotate", A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1.0)),
        ("HueSaturationValue", A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0)),
        ("CoarseDropout", A.CoarseDropout(max_holes=8, max_height=image_size // 16, max_width=image_size // 16, min_holes=1, p=1.0)),
        ("GaussNoise", A.GaussNoise(p=1.0)),
        ("Normalize", A.Normalize()),
    ]

    cropped = [
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0))(image=img)["image"]
        for img in images
    ]

    total = decode_ms
    for name, transform in stages:
        source = images if name == "RandomResizedCrop" else cropped
        start = time.perf_counter()
        for img in source:
            transform(image=img)
        stage_ms = (time.perf_counter() - start) / len(source) * 1000
        total += stage_ms
        print(f"  {name:<22} {stage_ms:7.2f} ms")

    print(f"  {'-' * 30}")
    print(f"  {'total (p=1.0 each)':<22} {total:7.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the ISIC input pipeline")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/processed/jpeg_256"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--profile", action="store_true", help="Break time down per stage")
    args = parser.parse_args()

    if args.profile:
        profile_stages(args.processed_dir, args.image_dir, args.image_size, args.samples)
    else:
        benchmark(args.processed_dir, args.image_dir, args.image_size, args.samples, args.batch_size)


if __name__ == "__main__":
    main()
