#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$PYTHON" "$ROOT/scripts/paper/build_experiment_artifacts.py" formal \
  --results-root "$RESULTS_ROOT" \
  --log-root "$LOG_ROOT" \
  --out-root "$ROOT"
