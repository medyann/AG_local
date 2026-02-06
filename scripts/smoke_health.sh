#!/usr/bin/env bash
set -euo pipefail

# Smoke-test that the server boots and answers /healthz using the mock backend.
#
# Usage:
#   ALPHAGENOME_BACKEND=mock ./scripts/smoke_health.sh

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

export ALPHAGENOME_BACKEND="${ALPHAGENOME_BACKEND:-mock}"

python3 -m pip install --quiet --upgrade pip >/dev/null
if [[ "${ALPHAGENOME_BACKEND}" == "mock" ]]; then
  python3 -m pip install --quiet -r requirements-mock.txt >/dev/null
else
  python3 -m pip install --quiet -r requirements.txt >/dev/null
fi

python3 -m uvicorn service.main:app --host "${HOST}" --port "${PORT}" >/tmp/alphagenome_api.log 2>&1 &
PID="$!"
trap 'kill "${PID}" >/dev/null 2>&1 || true' EXIT

python3 - <<'PY'
import json, os, time, urllib.request
host=os.getenv("HOST","127.0.0.1")
port=os.getenv("PORT","8000")
url=f"http://{host}:{port}/healthz"
for _ in range(50):
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            print(json.loads(r.read().decode("utf-8")))
            raise SystemExit(0)
    except Exception:
        time.sleep(0.2)
raise SystemExit("healthz did not respond")
PY

