#!/usr/bin/env bash
# Path B LoRA (or base) → test split predictions + headline metrics.
#
# Base (before LoRA):
#   VARIANT=base NO_ADAPTER=1 SLUG=qwen3_14b bash scripts/workflows/run_eval_test.sh
#
# After LoRA:
#   VARIANT=lora ADAPTER=local/adapters/qwen3_14b_pathB_lora/fold0/full256 \
#   SLUG=qwen3_14b bash scripts/workflows/run_eval_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/mempatch:$ROOT/scripts}"

SLUG="${SLUG:-qwen3_14b}"
VARIANT="${VARIANT:-lora}"
MODEL="${MODEL:-$ROOT/local/models/Qwen3-14B-MLX-4bit}"
OUT="${OUT:-$ROOT/local/results/$SLUG}"
TEST_SFT_DIR="${TEST_SFT_DIR:-$ROOT/local/train_data/paper/test500}"
SFT_DATA="${SFT_DATA:-$TEST_SFT_DIR/sft.jsonl}"
EVAL_DATA="${EVAL_DATA:-$ROOT/local/data/mempatch/test/scenarios.jsonl}"

if [[ ! -f "$SFT_DATA" ]]; then
  echo "Building test SFT bundle -> $TEST_SFT_DIR"
  "$PYTHON" "$ROOT/scripts/data/build_paper_eval_bundle.py" \
    --scenarios "$EVAL_DATA" \
    --out-dir "$TEST_SFT_DIR"
fi

mkdir -p "$OUT"
PRED="$OUT/test500_${VARIANT}_predictions.jsonl"
METRICS="$OUT/test500_${VARIANT}_metrics.json"

EVAL_ARGS=(
  --data "$SFT_DATA"
  --eval-data "$EVAL_DATA"
  --model "$MODEL"
  --out-predictions "$PRED"
  --out-metrics "$METRICS"
  --split-tag "test500_${VARIANT}"
  --model-tag "$SLUG"
  --variant-tag "$VARIANT"
)
if [[ -n "${NO_ADAPTER:-}" ]]; then
  EVAL_ARGS+=(--no-adapter)
else
  ADAPTER="${ADAPTER:?set ADAPTER for LoRA eval, or NO_ADAPTER=1 for base model}"
  EVAL_ARGS+=(--adapter-path "$ADAPTER")
fi

"$PYTHON" "$ROOT/scripts/eval/run_lora_test_eval.py" "${EVAL_ARGS[@]}"

"$PYTHON" "$ROOT/scripts/workflows/evaluate_mempatch_predictions.py" \
  --data "$EVAL_DATA" \
  --predictions "$PRED" \
  --print-table
