#!/bin/sh
set -e

exec uvicorn cancer_detection.serving.api:app --host 0.0.0.0 --port 8000
