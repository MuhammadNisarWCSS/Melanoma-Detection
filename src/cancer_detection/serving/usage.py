"""Privacy-minimal usage counter for /predict.

One row per successful prediction: timestamp, label, OOD flag, latency. No images,
filenames, patient metadata or IP addresses are stored. Nothing is exposed over HTTP;
read it with ``python -m cancer_detection.serving.usage`` (e.g. via ``docker exec``).
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)

_DB_PATH = Path(os.environ.get("USAGE_DB_PATH", "artifacts/usage.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    label INTEGER,
    out_of_distribution INTEGER,
    latency_ms REAL
)
"""


def _connect(path: Path = _DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute(_SCHEMA)
    return conn


def record_prediction(
    label: int | None, out_of_distribution: bool | None, latency_ms: float
) -> None:
    """Append one row. Never raises: usage tracking must not break predictions."""
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO predictions (label, out_of_distribution, latency_ms) VALUES (?, ?, ?)",
                (
                    None if label is None else int(label),
                    None if out_of_distribution is None else int(out_of_distribution),
                    latency_ms,
                ),
            )
    except Exception as exc:
        logger.warning("Could not record usage", error=str(exc))


def stats(path: Path = _DB_PATH) -> dict:
    """Aggregate usage numbers as a JSON-serialisable dict."""
    empty = {
        "total": 0,
        "last_24h": 0,
        "last_7d": 0,
        "last_30d": 0,
        "flagged": 0,
        "out_of_distribution": 0,
        "avg_latency_ms": None,
        "first": None,
        "last": None,
        "daily": [],
    }
    if not path.exists():
        return empty
    with _connect(path) as conn:
        total, flagged, ood, avg_ms, first, last = conn.execute(
            "SELECT COUNT(*), SUM(label), SUM(out_of_distribution), AVG(latency_ms), "
            "MIN(ts), MAX(ts) FROM predictions"
        ).fetchone()
        daily = conn.execute(
            "SELECT substr(ts, 1, 10) AS day, COUNT(*) FROM predictions "
            "GROUP BY day ORDER BY day DESC LIMIT 14"
        ).fetchall()
        recent = [
            conn.execute(
                "SELECT COUNT(*) FROM predictions "
                "WHERE ts >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)",
                (window,),
            ).fetchone()[0]
            for window in ("-24 hours", "-7 days", "-30 days")
        ]
    return {
        "total": total,
        "last_24h": recent[0],
        "last_7d": recent[1],
        "last_30d": recent[2],
        "flagged": flagged or 0,
        "out_of_distribution": ood or 0,
        "avg_latency_ms": avg_ms,
        "first": first,
        "last": last,
        "daily": [{"day": d, "count": n} for d, n in daily],
    }


def summary(path: Path = _DB_PATH) -> str:
    s = stats(path)
    total = s["total"]
    if not total:
        return "No predictions recorded yet."
    lines = [
        f"Total analyses:      {total}",
        f"Last 24h / 7d / 30d: {s['last_24h']} / {s['last_7d']} / {s['last_30d']}",
        f"Flagged malignant:   {s['flagged']} ({s['flagged'] / total:.1%})",
        f"Out-of-distribution: {s['out_of_distribution']} ({s['out_of_distribution'] / total:.1%})",
        f"Avg latency:         {s['avg_latency_ms']:.0f} ms",
        f"First / last:        {s['first']} / {s['last']}",
        "",
        "Per day (UTC, newest first):",
        *(f"  {d['day']}  {d['count']}" for d in s["daily"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(Path(sys.argv[1]) if len(sys.argv) > 1 else _DB_PATH))
