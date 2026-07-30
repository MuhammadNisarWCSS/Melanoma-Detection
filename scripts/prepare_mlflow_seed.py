"""Package local MLflow tracking data for seeding a Docker/EC2 volume.

Creates ./mlflow-seed/ with:
  mlflow.db
  artifacts/   (copy of mlartifacts/)

Also marks abandoned RUNNING runs as FAILED so the UI is cleaner after import.

Usage:
    python scripts/prepare_mlflow_seed.py
    python scripts/prepare_mlflow_seed.py --s3 s3://my-bucket/mlflow-seed

GitHub Actions deploy downloads that S3 prefix onto EC2 the first time the
volume is empty, so hosted MLflow shows your existing history. Later local
training defaults to http://18.219.3.159:5000 (or set MLFLOW_TRACKING_URI) so new runs
stream into the same volume live.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DB = ROOT / "mlflow.db"
SRC_ARTIFACTS = ROOT / "mlartifacts"
DEST = ROOT / "mlflow-seed"


def _package() -> tuple[int, int, int]:
    if not SRC_DB.is_file():
        raise FileNotFoundError(f"Missing {SRC_DB} — nothing to seed.")
    if not SRC_ARTIFACTS.is_dir():
        raise FileNotFoundError(f"Missing {SRC_ARTIFACTS}/ — DB alone is not enough for artifacts.")

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    dest_db = DEST / "mlflow.db"
    dest_artifacts = DEST / "artifacts"

    shutil.copy2(SRC_DB, dest_db)
    shutil.copytree(SRC_ARTIFACTS, dest_artifacts)

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    with sqlite3.connect(dest_db) as conn:
        cur = conn.execute(
            "UPDATE runs SET status=?, end_time=? WHERE status='RUNNING'",
            ("FAILED", now_ms),
        )
        closed = cur.rowcount
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_exps = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    return n_exps, n_runs, closed


def _upload_s3(uri: str) -> None:
    uri = uri.rstrip("/")
    # CRC32 avoids intermittent BadDigest/CRC64NVME failures on large
    # multipart uploads (common with AWS CLI v2 + big .ckpt files on Windows).
    cmd = [
        "aws",
        "s3",
        "sync",
        str(DEST),
        uri,
        "--delete",
        "--checksum-algorithm",
        "CRC32",
    ]
    print("Uploading seed:", " ".join(cmd))
    env = {
        **os.environ,
        "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required",
        "AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required",
    }
    subprocess.run(cmd, check=True, env=env)
    print(f"Uploaded to {uri}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s3",
        metavar="S3_URI",
        help="Upload mlflow-seed/ to this prefix (e.g. s3://my-bucket/mlflow-seed). "
        "Set the same URI as GitHub secret MLFLOW_SEED_S3_URI.",
    )
    args = parser.parse_args()

    try:
        n_exps, n_runs, closed = _package()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    size_mb = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"Wrote {DEST} ({size_mb:.1f} MiB)")
    print(f"  experiments={n_exps}  runs={n_runs}  closed_stale_RUNNING={closed}")

    if args.s3:
        try:
            _upload_s3(args.s3)
        except FileNotFoundError:
            print("aws CLI not found — install AWS CLI v2 and retry.", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as exc:
            print(f"S3 upload failed (exit {exc.returncode}).", file=sys.stderr)
            return 1
        print()
        print("Next:")
        print("  1. GitHub secret MLFLOW_SEED_S3_URI =" + f" {args.s3.rstrip('/')}")
        print("  2. Run Actions → Deploy stack (first deploy imports the seed once)")
        print("  3. Open http://18.219.3.159:3000 and train from this laptop:")
        print("       python scripts/train.py")
        return 0

    print()
    print("Next (pick one):")
    print("  A) Upload for GitHub->EC2 deploy:")
    print("       python scripts/prepare_mlflow_seed.py --s3 s3://YOUR_BUCKET/mlflow-seed")
    print("  B) Manual copy to EC2:")
    print("       scp -r mlflow-seed/ ec2-user@18.219.3.159:~/mlflow-seed/")
    print("  Then train live against the hosted server:")
    print("       python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
