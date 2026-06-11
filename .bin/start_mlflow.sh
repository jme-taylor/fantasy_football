#!/usr/bin/env bash
# Launch the MLflow UI, first killing any process already on its port.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=5000
DB_PATH="$REPO_ROOT/models/mlflow.db"

EXISTING="$(lsof -ti "tcp:$PORT" || true)"
if [ -n "$EXISTING" ]; then
  echo "Killing process(es) on port $PORT: $EXISTING"
  kill $EXISTING || true
  sleep 1
fi

mkdir -p "$REPO_ROOT/models"
echo "Starting MLflow UI at http://127.0.0.1:$PORT"
exec uv run mlflow ui \
  --backend-store-uri "sqlite:///$DB_PATH" \
  --port "$PORT"
