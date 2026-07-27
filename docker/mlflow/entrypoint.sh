#!/bin/sh
set -e

mkdir -p /mlflow/artifacts

# One-time import of laptop tracking data (see scripts/prepare_mlflow_seed.py).
# Mount that folder at /seed via docker/docker-compose.seed.yml. After the first
# successful seed, /mlflow/.seed_applied prevents overwriting live EC2 runs.
# Force again with: docker compose exec mlflow rm /mlflow/.seed_applied
# (or wipe the volume), then recreate with the seed compose file.
if [ -f /seed/mlflow.db ] && [ ! -f /mlflow/.seed_applied ]; then
  echo "Seeding MLflow from /seed ..."
  cp /seed/mlflow.db /mlflow/mlflow.db
  if [ -d /seed/artifacts ]; then
    cp -a /seed/artifacts/. /mlflow/artifacts/
  fi
  touch /mlflow/.seed_applied
  echo "Seed complete."
fi

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --default-artifact-root mlflow-artifacts:/ \
  --artifacts-destination /mlflow/artifacts \
  --serve-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
