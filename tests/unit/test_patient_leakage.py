"""Regression tests: no patient may appear in more than one split.

ISIC 2020 has 33,126 images from only 2,056 patients. A per-image random split
places nearly every patient on both sides of the train/test boundary, and the
model can then recognise the patient rather than the lesion. These tests lock
the StratifiedGroupKFold contract so that leak cannot silently reappear.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def _load_prepare_module():
    """Load scripts/prepare_data.py without requiring a scripts package."""
    path = PROJECT_ROOT / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_data"] = module
    spec.loader.exec_module(module)
    return module


prepare_data = _load_prepare_module()
GROUP_COL = prepare_data.GROUP_COL
KEEP_COLS = prepare_data.KEEP_COLS
assert_patient_disjoint = prepare_data.assert_patient_disjoint
prepare = prepare_data.prepare


def _load_splits() -> dict[str, pd.DataFrame]:
    paths = {name: PROCESSED / f"{name}.csv" for name in ("train", "val", "test")}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"Split CSVs not present: {missing}")
    return {name: pd.read_csv(path) for name, path in paths.items()}


def test_split_csvs_contain_patient_id() -> None:
    splits = _load_splits()
    for name, df in splits.items():
        assert GROUP_COL in df.columns, f"{name}.csv is missing {GROUP_COL}"
        assert df[GROUP_COL].notna().all()
        assert df[GROUP_COL].nunique() >= 1


def test_no_patient_overlap_between_splits() -> None:
    splits = _load_splits()
    assert_patient_disjoint(splits)


def test_assert_patient_disjoint_raises_on_overlap() -> None:
    shared = pd.DataFrame(
        {
            "image_name": ["a", "b"],
            GROUP_COL: ["P1", "P1"],
            "target": [0, 1],
        }
    )
    with pytest.raises(AssertionError, match="appear in both"):
        assert_patient_disjoint(
            {
                "train": shared.iloc[:1],
                "val": shared.iloc[1:],
                "test": shared.iloc[:0],
            }
        )


def test_prepare_regenerates_disjoint_splits(tmp_path: Path) -> None:
    """End-to-end: prepare() on a tiny synthetic CSV never leaks a patient."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    rows = []
    for patient in range(40):
        for img in range(3):
            rows.append(
                {
                    "image_name": f"img_{patient}_{img}",
                    "patient_id": f"IP_{patient:04d}",
                    "target": int(patient % 10 == 0),  # 10% positive patients
                    "age_approx": 50.0,
                    "sex": "female",
                    "anatom_site_general_challenge": "torso",
                }
            )
    pd.DataFrame(rows)[KEEP_COLS].to_csv(raw / "train.csv", index=False)

    prepare(raw, processed, val_split=0.20, test_split=0.20, seed=0, overwrite=True)

    splits = {name: pd.read_csv(processed / f"{name}.csv") for name in ("train", "val", "test")}
    assert_patient_disjoint(splits)
    for name, df in splits.items():
        assert len(df) > 0, f"{name} is empty"
        assert set(df.columns) >= set(KEEP_COLS)
