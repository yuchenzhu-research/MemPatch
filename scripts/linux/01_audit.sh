#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

AUDIT_TRAIN="${AUDIT_TRAIN:-$(resolve_split_dir train)}"
AUDIT_TEST="${AUDIT_TEST:-$(resolve_split_dir test)}"

"$PYTHON" "$ROOT/scripts/data/audit_decision_boundary.py" \
  --data "$AUDIT_TRAIN" \
  --data "$AUDIT_TEST"

echo "Audit passed."
