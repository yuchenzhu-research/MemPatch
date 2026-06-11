#!/usr/bin/env bash
# Build fixed stratified train.jsonl + valid.jsonl (80/20 within train3500).
#
#   SPLIT_INDEX=0 SLUG=gemma3_12b bash scripts/linux/02_prepare_split.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
SPLIT_INDEX="${SPLIT_INDEX:-0}"
SFT_DIR="$TRAIN_DATA_ROOT/${SLUG}_split${SPLIT_INDEX}"

mkdir -p "$SFT_DIR" "$LOG_ROOT"

resolve_train_scenarios
resolve_test_scenarios

"$PYTHON" "$ROOT/scripts/data/prepare_mempatch_v13_smoke.py" \
  --full-train \
  --train-data "$TRAIN_SCENARIOS" \
  --test-data "$TEST_SCENARIOS" \
  --out-dir "$SFT_DIR" \
  --seed "$SEED" \
  --split-parts "$SPLIT_PARTS" \
  --split-index "$SPLIT_INDEX"

echo "SFT ready: $SFT_DIR"
