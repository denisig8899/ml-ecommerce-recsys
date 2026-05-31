#!/usr/bin/env bash
# Запустить локальный MLflow tracking server

set -e

MLFLOW_DIR="${MLFLOW_DIR:-./mlruns}"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"

echo "Запуск MLflow UI → http://localhost:${PORT}"
python -m mlflow ui \
    --backend-store-uri "${MLFLOW_DIR}" \
    --host "${HOST}" \
    --port "${PORT}"
