#!/usr/bin/env bash
# Back-compat wrapper -> run_model.sh
#
#   SLUG=gemma3_12b bash scripts/linux/run_paper_model.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
SLUG="${SLUG:?set SLUG}"
PHASES="${PHASES:-auto}"
bash "$LINUX_DIR/01_audit.sh"
SLUG="$SLUG" PHASES="$PHASES" bash "$LINUX_DIR/run_model.sh"
