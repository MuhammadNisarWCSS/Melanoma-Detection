# FastAPI inference service (CPU PyTorch).
# Build from repo root:
#   docker compose build api
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEVICE=cpu \
    THRESHOLD_PATH=/app/artifacts/threshold.json \
    MLFLOW_TRACKING_URI=http://mlflow:5000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install CPU torch, then the package, then re-pin CPU torch in case pip
# resolved a default (CUDA) wheel from PyPI while installing dependencies.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir --force-reinstall --no-deps torch \
        --index-url https://download.pytorch.org/whl/cpu

COPY docker/api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /app/serving_model /app/artifacts

# Optional baked model / threshold — place files before build:
#   mlflow artifacts download -u runs:/<run_id>/model -d serving_model
#   copy artifacts/threshold.json into artifacts/
COPY serving_model/ /app/serving_model/
COPY artifacts/ /app/artifacts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
