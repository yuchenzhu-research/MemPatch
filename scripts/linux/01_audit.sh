#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

"$PYTHON" "$ROOT/scripts/workflows/audit_decision_boundary.py" \
  --data "$ROOT/hf_release/mempatch/train" \
  --data "$ROOT/hf_release/mempatch/test"

echo "Audit passed."
