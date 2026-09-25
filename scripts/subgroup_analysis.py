"""Per-subgroup performance on the held-out test split.

Reads artifacts/test_predictions.csv (from `evaluate.py --save-predictions`) and reports
sensitivity, specificity and AUROC by sex, age band and anatomical site. The test split has only
30 malignant images, so most subgroups have a handful of positives; the point is to show where the
model is unproven, not to claim it is fair.

    python scripts/subgroup_analysis.py [--predictions PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

AGE_BINS = [0, 40, 60, 200]
AGE_LABELS = ["<40", "40-59", "60+"]


def summarise(df: pd.DataFrame) -> dict:
    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]
    out = {
        "n": int(len(df)),
        "n_malignant": int(len(pos)),
        "sensitivity": float(pos["predicted"].mean()) if len(pos) else None,
        "specificity": float(1 - neg["predicted"].mean()) if len(neg) else None,
        "auroc": None,
    }
    if len(pos) >= 5 and len(neg) >= 5:
        out["auroc"] = float(roc_auc_score(df["target"], df["probability"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default="artifacts/test_predictions.csv")
    ap.add_argument("--out", default="artifacts/subgroup_metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    df["sex"] = df["sex"].fillna("unknown")
    df["site"] = df["anatom_site_general_challenge"].fillna("unknown")
    df["age_band"] = pd.cut(df["age_approx"], AGE_BINS, labels=AGE_LABELS, right=False)
    df["age_band"] = df["age_band"].astype(object).where(df["age_approx"].notna(), "unknown")

    report = {"overall": summarise(df)}
    for name, col in (("sex", "sex"), ("age", "age_band"), ("site", "site")):
        report[name] = {str(k): summarise(g) for k, g in df.groupby(col, dropna=False)}

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(v: float | None) -> str:
        return "  n/a" if v is None else f"{v:5.2f}"

    print(f"{'group':<26}{'n':>6}{'pos':>5}{'sens':>7}{'spec':>7}{'auroc':>7}")
    for section in ("overall", "sex", "age", "site"):
        rows = {"overall": report["overall"]} if section == "overall" else report[section]
        for k, r in rows.items() if section != "overall" else [("overall", rows["overall"])]:
            label = k if section == "overall" else f"{section}={k}"
            print(
                f"{label:<26}{r['n']:>6}{r['n_malignant']:>5}"
                f"{fmt(r['sensitivity']):>7}{fmt(r['specificity']):>7}{fmt(r['auroc']):>7}"
            )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
