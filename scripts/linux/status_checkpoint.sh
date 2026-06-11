#!/usr/bin/env bash
# Quick checkpoint / train status for one slug on the Linux server.
#
#   SLUG=gemma3_12b bash scripts/linux/status_checkpoint.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"

echo "LOCAL_ROOT=$LOCAL_ROOT"
echo "RUN_ID=$RUN_ID"
echo "LOG_ROOT=$LOG_ROOT"
echo "ADAPTER_ROOT=$ADAPTER_ROOT"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo

echo "== trainer_metrics.json =="
find "$LOG_ROOT" "$ADAPTER_ROOT" -path "*${SLUG}*" -name trainer_metrics.json 2>/dev/null | sort || true

echo
echo "== adapter checkpoints =="
find "$ADAPTER_ROOT" -path "*${SLUG}_multitask_lora*" -type d -name 'checkpoint-*' 2>/dev/null | sort | head -20 || true

echo
echo "== selection json =="
ls -la "$RESULTS_ROOT/$SLUG/checkpoint_selection.json" 2>/dev/null || echo "(missing)"

echo
bash "$LINUX_DIR/status_models.sh" 2>/dev/null | grep -A6 "^slug=$SLUG" || true
