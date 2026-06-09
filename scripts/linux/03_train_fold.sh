#!/usr/bin/env bash
# QLoRA train one k-fold.
#
#   KFOLD_FOLD=0 SLUG=gemma3_12b bash scripts/linux/03_train_fold.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
KFOLD_FOLD="${KFOLD_FOLD:?set KFOLD_FOLD}"
HF_MODEL="$(resolve_hf_model "$SLUG")"
SFT_DIR="$TRAIN_DATA_ROOT/${SLUG}_fold${KFOLD_FOLD}"
OUT_DIR="$ADAPTER_ROOT/${SLUG}_pathB_lora/fold${KFOLD_FOLD}/${RUN_ID}"
LOG_DIR="$LOG_ROOT/${SLUG}_fold${KFOLD_FOLD}"

if [[ ! -f "$SFT_DIR/train.jsonl" ]]; then
  echo "Missing $SFT_DIR/train.jsonl — run 02_prepare_kfold.sh first" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

"$PYTHON" "$LINUX_DIR/train_qlora.py" \
  --model-id "$HF_MODEL" \
  --train-data "$SFT_DIR/train.jsonl" \
  --valid-data "$SFT_DIR/valid.jsonl" \
  --output-dir "$OUT_DIR" \
  --log-dir "$LOG_DIR" \
  --max-steps "$TRAIN_ITERS" \
  --save-steps "$SAVE_EVERY" \
  --eval-steps "$SAVE_EVERY" \
  --seed "$SEED"

echo "Adapter checkpoints: $OUT_DIR"
