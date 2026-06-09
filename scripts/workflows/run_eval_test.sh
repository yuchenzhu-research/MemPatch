#!/usr/bin/env bash
# Path B LoRA → test500 predictions + headline metrics.
#
#   ADAPTER=local/adapters/qwen3_14b_pathB_lora/fold0/full256 \
#   MODEL=local/models/Qwen3-14B-MLX-4bit \
#   bash scripts/workflows/run_eval_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/src:$ROOT/scripts}"

SLUG="${SLUG:-qwen3_14b}"
ADAPTER="${ADAPTER:?set ADAPTER to fold/run adapter directory}"
MODEL="${MODEL:-$ROOT/local/models/Qwen3-14B-MLX-4bit}"
OUT="${OUT:-$ROOT/local/results/$SLUG}"
SFT_DATA="${SFT_DATA:-$ROOT/local/train_data/kfold/${SLUG}_fold0/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$ROOT/hf_release/mempatch/test/scenarios.jsonl}"

mkdir -p "$OUT"
PRED="$OUT/test500_predictions.jsonl"
METRICS="$OUT/test500_metrics.json"

"$PYTHON" "$ROOT/scripts/eval/run_lora_test_eval.py" \
  --data "$SFT_DATA" \
  --eval-data "$EVAL_DATA" \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --out-predictions "$PRED" \
  --out-metrics "$METRICS" \
  --split-tag test500

"$PYTHON" "$ROOT/scripts/workflows/evaluate_mempatch_predictions.py" \
  --data "$EVAL_DATA" \
  --predictions "$PRED" \
  --print-table
