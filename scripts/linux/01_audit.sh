#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

AUDIT_TRAIN="${AUDIT_TRAIN:-$LOCAL_ROOT/data/mempatch/train}"
AUDIT_TEST="${AUDIT_TEST:-$LOCAL_ROOT/data/mempatch/test}"
if [[ ! -d "$AUDIT_TRAIN" ]]; then
  AUDIT_TRAIN="$ROOT/hf_release/mempatch/train"
fi
if [[ ! -d "$AUDIT_TEST" ]]; then
  AUDIT_TEST="$ROOT/hf_release/mempatch/test"
fi

"$PYTHON" "$ROOT/scripts/workflows/audit_decision_boundary.py" \
  --data "$AUDIT_TRAIN" \
  --data "$AUDIT_TEST"

echo "Audit passed."
