#!/bin/sh
# Runs both the prediction worker and Flask server in one container.
# With Supabase as the database, no shared filesystem is needed —
# both processes connect to the same remote Postgres instance.
#
# main.py runs one full scoring pass and exits - it does not loop internally.
# This script loops it every PREDICT_INTERVAL_SECONDS (default 120s) in
# the background, while server.py runs in the foreground serving /traffic.

set -e

INTERVAL="${PREDICT_INTERVAL_SECONDS:-120}"

(
  while true; do
    echo "[worker] starting prediction cycle..."
    python main.py
    echo "[worker] cycle complete, sleeping ${INTERVAL}s..."
    sleep "$INTERVAL"
  done
) &

echo "Starting backend (server.py) in foreground..."
exec python server.py
