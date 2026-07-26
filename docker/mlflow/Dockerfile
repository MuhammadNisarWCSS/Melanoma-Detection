# MLflow tracking server for the portfolio stack.
FROM python:3.11-slim-bookworm

WORKDIR /mlflow

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "mlflow>=2.14.0,<4"

RUN mkdir -p /mlflow/artifacts /seed

COPY docker/mlflow/entrypoint.sh /entrypoint.sh
# Strip CRLF in case the file was edited on Windows (avoids "no such file or directory").
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/health || exit 1

# Persist under /mlflow (compose volume). Optional /seed import on first boot
# (docker-compose.seed.yml). Artifacts use the proxied mlflow-artifacts://
# scheme so local training clients can upload to a remote EC2 server over HTTP.
ENTRYPOINT ["/entrypoint.sh"]
