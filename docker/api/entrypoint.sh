#!/bin/sh
set -e

# Prefer an explicitly set MODEL_URI. Otherwise, if a baked MLflow model
# directory was copied into the image, use that so /predict works even when
# the MLflow service has no runs yet.
if [ -z "${MODEL_URI:-}" ] && [ -f /app/serving_model/MLmodel ]; then
  export MODEL_URI=/app/serving_model
  echo "Using baked model at ${MODEL_URI}"
fi

exec uvicorn cancer_detection.serving.api:app --host 0.0.0.0 --port 8000
