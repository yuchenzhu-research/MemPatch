#!/usr/bin/env bash
# Build the fixed stratified train.jsonl + valid.jsonl split.
#
#   KFOLD_FOLD=0 SLUG=gemma3_12b bash scripts/linux/02_prepare_kfold.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
KFOLD_FOLD="${KFOLD_FOLD:?set KFOLD_FOLD}"
SFT_DIR="$TRAIN_DATA_ROOT/${SLUG}_fold${KFOLD_FOLD}"
ADAPTER_DIR="$ADAPTER_ROOT/${SLUG}_pathB_lora"
MLX_CONFIG="$LOG_ROOT/${SLUG}_fold${KFOLD_FOLD}_${RUN_ID}.yaml"

mkdir -p "$SFT_DIR" "$LOG_ROOT"

resolve_train_scenarios
resolve_test_scenarios

# MLX yaml is a side effect; Linux training reads JSONL only.
"$PYTHON" "$ROOT/scripts/data/prepare_mempatch_v13_smoke.py" \
  --profile "$PROFILE" \
  --full-train \
  --train-data "$TRAIN_SCENARIOS" \
  --test-data "$TEST_SCENARIOS" \
  --out-dir "$SFT_DIR" \
  --model-dir "$LOCAL_ROOT/models/_placeholder" \
  --adapter-dir "$ADAPTER_DIR" \
  --mlx-config "$MLX_CONFIG" \
  --seed "$SEED" \
  --k-folds "$VALIDATION_PARTS" \
  --fold "$KFOLD_FOLD" \
  --run-id "$RUN_ID"

echo "SFT ready: $SFT_DIR"
