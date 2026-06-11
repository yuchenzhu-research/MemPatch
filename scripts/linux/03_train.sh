#!/usr/bin/env bash
# QLoRA multitask train: Path B responses + Path A typed actions.
#
#   SPLIT_INDEX=0 SLUG=gemma3_12b bash scripts/linux/03_train.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
SPLIT_INDEX="${SPLIT_INDEX:-0}"
HF_MODEL="$(resolve_hf_model "$SLUG")"
SFT_DIR="$TRAIN_DATA_ROOT/${SLUG}_split${SPLIT_INDEX}"
OUT_DIR="$ADAPTER_ROOT/${SLUG}_multitask_lora/split${SPLIT_INDEX}/${RUN_ID}"
LOG_DIR="$LOG_ROOT/${SLUG}_split${SPLIT_INDEX}/${RUN_ID}"

if [[ ! -f "$SFT_DIR/train.jsonl" ]]; then
  echo "Missing $SFT_DIR/train.jsonl — run 02_prepare_split.sh first" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

MAX_SEQ_LEN="$(train_max_seq_length_for_slug "$SLUG")"
echo "train max_seq_length=$MAX_SEQ_LEN (slug=$SLUG)"

TRAIN_ARGS=(
  --model-id "$HF_MODEL"
  --train-data "$SFT_DIR/train.jsonl"
  --valid-data "$SFT_DIR/valid.jsonl"
  --output-dir "$OUT_DIR"
  --log-dir "$LOG_DIR"
  --max-steps "$TRAIN_ITERS"
  --save-steps "$SAVE_EVERY"
  --eval-steps "$SAVE_EVERY"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-4}"
  --max-seq-length "$MAX_SEQ_LEN"
  --eval-accumulation-steps "${TRAIN_EVAL_ACCUMULATION_STEPS:-8}"
  --seed "$SEED"
)
if [[ -n "${RESUME_FROM:-}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "$RESUME_FROM")
fi

"$PYTHON" "$LINUX_DIR/train_qlora.py" "${TRAIN_ARGS[@]}"

echo "Adapter checkpoints: $OUT_DIR"
