#!/usr/bin/env bash
# Launch the MLflow UI, first killing any process already on its port.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Avoid port 5000: macOS Control Center / AirPlay Receiver binds it by default.
PORT=5050
DB_PATH="$REPO_ROOT/models/mlflow.db"

# Kill any existing MLflow tree for this port. Match the command line, not just
# the port: `uv run mlflow ui` supervises gunicorn, so killing only the workers
# lsof reports lets the supervisor respawn them and the port never frees.
pkill -9 -f "mlflow ui .*--port $PORT" 2>/dev/null || true
EXISTING="$(lsof -ti "tcp:$PORT" || true)"
if [ -n "$EXISTING" ]; then
  echo "Killing leftover process(es) on port $PORT: $EXISTING"
  kill -9 $EXISTING || true
  sleep 1
fi

mkdir -p "$REPO_ROOT/models"
echo "Starting MLflow UI at http://127.0.0.1:$PORT"
exec uv run mlflow ui \
  --backend-store-uri "sqlite:///$DB_PATH" \
  --port "$PORT"
