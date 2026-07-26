"""Build stratified train/val/test CSV splits from raw ISIC 2020 data.

Reads:   data/raw/train.csv  (from Kaggle download)
Writes:  data/processed/{train,val,test}.csv

Each output CSV has columns:
    image_name, target, age_approx, sex, anatom_site_general_challenge

Stratification is done on the binary 'target' column to preserve the
~1.76% positive rate in every split (critical for unbiased evaluation).

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --raw-dir data/raw --processed-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
KEEP_COLS = [
    "image_name",
    "target",
    "age_approx",
    "sex",
    "anatom_site_general_challenge",
]


def prepare(raw_dir: Path, processed_dir: Path, val_split: float, test_split: float, seed: int, overwrite: bool = False) -> None:
    csv_path = raw_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/download_data.py first."
        )

    existing = [processed_dir / f"{s}.csv" for s in ("train", "val", "test") if (processed_dir / f"{s}.csv").exists()]
    if existing and not overwrite:
        print(f"Splits already exist in {processed_dir} ({', '.join(p.name for p in existing)}).")
        print("Nothing to do. Pass --overwrite to regenerate them.")
        return

    print(f"Reading {csv_path} …")
    df = pd.read_csv(csv_path)
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing from train.csv: {missing}")

    df = df[KEEP_COLS].copy()

    # Verify images exist — warn if any are missing
    img_dir = raw_dir / "jpeg" / "train"
    if img_dir.exists():
        missing_imgs = [
            name for name in df["image_name"]
            if not (img_dir / f"{name}.jpg").exists()
        ]
        if missing_imgs:
            print(f"Warning: {len(missing_imgs)} images listed in CSV not found on disk")

    print(f"Total samples: {len(df):,}  |  Positives: {df['target'].sum():,}  ({df['target'].mean() * 100:.2f}%)")

    # First carve out the test split, then split remainder into train/val
    train_val, test = train_test_split(
        df,
        test_size=test_split,
        stratify=df["target"],
        random_state=seed,
    )
    adjusted_val_frac = val_split / (1.0 - test_split)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_frac,
        stratify=train_val["target"],
        random_state=seed,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        pos_rate = split["target"].mean() * 100
        print(f"  {name:5s}: {len(split):6,} samples  |  {pos_rate:.2f}% positive")

    print(f"\nSplit CSVs written to {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ISIC 2020 train/val/test splits")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split CSVs")
    args = parser.parse_args()

    prepare(args.raw_dir, args.processed_dir, args.val_split, args.test_split, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
