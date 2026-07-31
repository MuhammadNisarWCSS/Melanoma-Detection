from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog

# repo root: src/cancer_detection/utils/logger.py → parents[3]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = _PROJECT_ROOT / "logs"

_tee_installed = False


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def _should_log_to_file(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    if os.environ.get("LOG_TO_FILE") is not None:
        return _env_flag("LOG_TO_FILE", True)
    # Containers already capture stdout/stderr — skip local files there.
    return not Path("/.dockerenv").exists()


class _Tee:
    """Write to the console stream and a log file."""

    def __init__(self, console: object, log_file: object) -> None:
        self._console = console
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._console.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._console.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._console, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self._console, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._console, "errors", None)

    def reconfigure(self, **kwargs: object) -> None:
        for stream in (self._console, self._log_file):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(**kwargs)
                except TypeError:
                    pass


def _install_stdio_tee(log_path: Path) -> None:
    """Mirror stdout/stderr to ``log_path`` (Lightning / MLflow prints included)."""
    global _tee_installed
    if _tee_installed:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    log_fp.write(f"\n===== session start {stamp} =====\n")
    log_fp.flush()

    sys.stdout = _Tee(sys.__stdout__, log_fp)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, log_fp)  # type: ignore[assignment]
    _tee_installed = True


def configure_logging(
    level: str = "INFO",
    *,
    name: str = "app",
    log_file: bool | None = None,
) -> Path | None:
    """Configure structlog and optionally tee stdio into ``logs/{name}.log``.

    Set ``LOG_TO_FILE=0`` to disable file capture. In Docker (``/.dockerenv``)
    file logging is off by default so container logs stay on stdout/stderr.
    """
    # structlog.stdlib.LoggerFactory routes through the stdlib logging module, whose
    # loggers default to WARNING with no handler — INFO records were silently dropped
    # and even WARNING/ERROR only reached stderr via the unformatted "last resort"
    # handler, unformatted and easy to miss (e.g. the Predictor load failure that
    # left the API running with no model and no visible error).
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if not _should_log_to_file(log_file):
        return None

    log_name = os.environ.get("LOG_NAME", name)
    log_path = LOGS_DIR / f"{log_name}.log"
    _install_stdio_tee(log_path)
    return log_path


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)
