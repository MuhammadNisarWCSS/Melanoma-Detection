"""Package local MLflow tracking data for seeding a Docker/EC2 volume.

Creates ./mlflow-seed/ with:
  mlflow.db
  artifacts/   (copy of mlartifacts/)

Also marks abandoned RUNNING runs as FAILED so the UI is cleaner after import.

Usage:
    python scripts/prepare_mlflow_seed.py

Then on EC2 (after scp -r mlflow-seed/ …):
    docker compose -f docker-compose.yml -f docker-compose.seed.yml up -d --build mlflow
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DB = ROOT / "mlflow.db"
SRC_ARTIFACTS = ROOT / "mlartifacts"
DEST = ROOT / "mlflow-seed"


def main() -> int:
    if not SRC_DB.is_file():
        print(f"Missing {SRC_DB} — nothing to seed.", file=sys.stderr)
        return 1
    if not SRC_ARTIFACTS.is_dir():
        print(f"Missing {SRC_ARTIFACTS}/ — DB alone is not enough for artifacts.", file=sys.stderr)
        return 1

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    dest_db = DEST / "mlflow.db"
    dest_artifacts = DEST / "artifacts"

    shutil.copy2(SRC_DB, dest_db)
    shutil.copytree(SRC_ARTIFACTS, dest_artifacts)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    with sqlite3.connect(dest_db) as conn:
        cur = conn.execute(
            "UPDATE runs SET status=?, end_time=? WHERE status='RUNNING'",
            ("FAILED", now_ms),
        )
        closed = cur.rowcount
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_exps = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    size_mb = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"Wrote {DEST} ({size_mb:.1f} MiB)")
    print(f"  experiments={n_exps}  runs={n_runs}  closed_stale_RUNNING={closed}")
    print()
    print("Next:")
    print("  1. scp -r mlflow-seed/ ec2-user@<host>:~/CancerDetection/")
    print("  2. On EC2:")
    print("       docker compose -f docker-compose.yml -f docker-compose.seed.yml up -d --build mlflow")
    print("  3. Train from this laptop against EC2:")
    print('       $env:MLFLOW_TRACKING_URI = "http://<ec2-host>:5000"')
    print("       python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
