"""Build patient-disjoint train/val/test CSV splits from raw ISIC 2020 data.

Reads:   data/raw/train.csv  (from Kaggle download)
Writes:  data/processed/{train,val,test}.csv

Each output CSV has columns:
    image_name, patient_id, target, age_approx, sex, anatom_site_general_challenge

Why grouping matters
--------------------
ISIC 2020 contains 33,126 images from only 2,056 patients — a median of 12 images
each. A per-image random split therefore places nearly every patient on both sides
of the boundary: measured on the previous split, 1,656 of 1,657 test images shared a
patient with training. The model can then recognise the patient (skin tone, hair,
imaging setup, neighbouring nevi) rather than the lesion, and the reported test
score is inflated.

StratifiedGroupKFold keeps every patient wholly inside one split while holding the
~1.76% positive rate roughly constant across all three.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
GROUP_COL = "patient_id"
KEEP_COLS = [
    "image_name",
    "patient_id",
    "target",
    "age_approx",
    "sex",
    "anatom_site_general_challenge",
]


def _group_split(
    df: pd.DataFrame, holdout_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off roughly holdout_fraction of the rows without splitting any patient.

    StratifiedGroupKFold has no direct test_size, so the fraction is expressed as
    1/n_splits and a single fold is taken as the holdout.
    """
    n_splits = max(2, round(1.0 / holdout_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, holdout_idx = next(splitter.split(df, df["target"], groups=df[GROUP_COL]))
    return df.iloc[keep_idx].copy(), df.iloc[holdout_idx].copy()


def assert_patient_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly rather than silently regenerate a leaking split."""
    patients = {name: set(split[GROUP_COL]) for name, split in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[a] & patients[b]
        if overlap:
            raise AssertionError(
                f"{len(overlap)} patient(s) appear in both {a} and {b}: {sorted(overlap)[:5]}"
            )


def prepare(
    raw_dir: Path,
    processed_dir: Path,
    val_split: float,
    test_split: float,
    seed: int,
    overwrite: bool = False,
) -> None:
    csv_path = raw_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the ISIC 2020 dataset from Kaggle and place "
            f"train.csv + jpeg/train/ under {raw_dir}/ — see the Quickstart in README.md."
        )

    existing = [
        processed_dir / f"{s}.csv"
        for s in ("train", "val", "test")
        if (processed_dir / f"{s}.csv").exists()
    ]
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
        missing_imgs = [name for name in df["image_name"] if not (img_dir / f"{name}.jpg").exists()]
        if missing_imgs:
            print(f"Warning: {len(missing_imgs)} images listed in CSV not found on disk")

    print(
        f"Total samples: {len(df):,}  |  Positives: {df['target'].sum():,}  "
        f"({df['target'].mean() * 100:.2f}%)  |  Patients: {df[GROUP_COL].nunique():,}"
    )

    # Carve out the test split first, then split the remainder into train/val. Both
    # steps group on patient_id, so a patient never crosses a split boundary.
    train_val, test = _group_split(df, test_split, seed)
    adjusted_val_frac = val_split / (1.0 - test_split)
    train, val = _group_split(train_val, adjusted_val_frac, seed)

    splits = {"train": train, "val": val, "test": test}
    assert_patient_disjoint(splits)

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(processed_dir / f"{name}.csv", index=False)

    print()
    for name, split in splits.items():
        print(
            f"  {name:5s}: {len(split):6,} samples  |  "
            f"{split['target'].mean() * 100:.2f}% positive  |  "
            f"{split['target'].sum():3,} malignant  |  "
            f"{split[GROUP_COL].nunique():,} patients"
        )

    print("\nNo patient appears in more than one split.")
    print(f"Split CSVs written to {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare patient-disjoint ISIC 2020 train/val/test splits"
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split CSVs")
    args = parser.parse_args()

    prepare(
        args.raw_dir,
        args.processed_dir,
        args.val_split,
        args.test_split,
        args.seed,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
