#!/usr/bin/env bash
# Stop any MLflow UI running on its port.
set -euo pipefail

PORT=5000
EXISTING="$(lsof -ti "tcp:$PORT" || true)"
if [ -n "$EXISTING" ]; then
  echo "Stopping MLflow on port $PORT: $EXISTING"
  kill $EXISTING
else
  echo "No MLflow process found on port $PORT"
fi
