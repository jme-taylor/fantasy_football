#!/usr/bin/env bash
# Stop any MLflow UI running on its port.
set -euo pipefail

PORT=5050
# Kill the whole MLflow tree (uv run -> mlflow ui -> gunicorn), not just the
# port listeners, so the supervisor cannot respawn the workers.
pkill -9 -f "mlflow ui .*--port $PORT" 2>/dev/null || true
EXISTING="$(lsof -ti "tcp:$PORT" || true)"
if [ -n "$EXISTING" ]; then
  echo "Stopping leftover MLflow on port $PORT: $EXISTING"
  kill -9 $EXISTING || true
else
  echo "No MLflow process found on port $PORT"
fi
