#!/usr/bin/env bash
# Quick Qwen3-14B MLX LoRA smoke workflow for MemPatch v1.3.
#
# Runs a small prepare -> train -> eval loop. Generated data, adapters, logs, and
# predictions live under local/ and must not be committed.
#
# Usage:
#   ./scripts/run_qwen3_lora_smoke.sh          # prepare + train + eval
#   ./scripts/run_qwen3_lora_smoke.sh prepare
#   ./scripts/run_qwen3_lora_smoke.sh train
#   ./scripts/run_qwen3_lora_smoke.sh eval
#
# Optional overrides:
#   SMOKE_ITERS=8 EVAL_LIMIT=5 ./scripts/run_qwen3_lora_smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.:src

MODEL_DIR="${MODEL_DIR:-local/models/Qwen3-14B-MLX-4bit}"
RUN_NAME="${RUN_NAME:-qwen3_14b_mempatch_v13_quick_smoke}"
DATA_DIR="${DATA_DIR:-local/train_data/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-local/logs/$RUN_NAME}"
ADAPTER_DIR="${ADAPTER_DIR:-local/adapters/$RUN_NAME}"
MLX_CONFIG="${MLX_CONFIG:-$LOG_DIR/mlx_lora.yaml}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/train.log}"

SMOKE_ITERS="${SMOKE_ITERS:-16}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-8}"
SMOKE_SAVE_EVERY="${SMOKE_SAVE_EVERY:-8}"
SMOKE_STEPS_PER_EVAL="${SMOKE_STEPS_PER_EVAL:-8}"
SMOKE_VAL_BATCHES="${SMOKE_VAL_BATCHES:-8}"
EVAL_LIMIT="${EVAL_LIMIT:-10}"
MAX_TOKENS="${MAX_TOKENS:-256}"

export SMOKE_ITERS
export SMOKE_BATCH_SIZE
export SMOKE_GRAD_ACCUM
export SMOKE_SAVE_EVERY
export SMOKE_STEPS_PER_EVAL
export SMOKE_VAL_BATCHES

require_model() {
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Missing model directory: $MODEL_DIR" >&2
    echo "Download it first: .venv/bin/python scripts/download_mlx_model.py --preset qwen3-14b" >&2
    exit 1
  fi
}

patch_quick_config() {
  .venv/bin/python - "$MLX_CONFIG" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
replacements = {
    "batch_size": os.environ["SMOKE_BATCH_SIZE"],
    "iters": os.environ["SMOKE_ITERS"],
    "grad_accumulation_steps": os.environ["SMOKE_GRAD_ACCUM"],
    "save_every": os.environ["SMOKE_SAVE_EVERY"],
    "steps_per_eval": os.environ["SMOKE_STEPS_PER_EVAL"],
    "val_batches": os.environ["SMOKE_VAL_BATCHES"],
}
lines = path.read_text(encoding="utf-8").splitlines()
out = []
for line in lines:
    key = line.split(":", 1)[0].strip()
    if key in replacements and not line.startswith(" "):
        out.append(f"{key}: {replacements[key]}")
    else:
        out.append(line)
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"Wrote quick config overrides -> {path}")
PY
}

step_prepare() {
  require_model
  mkdir -p "$LOG_DIR" "$ADAPTER_DIR"
  .venv/bin/python scripts/prepare_mempatch_v13_smoke.py \
    --profile smoke \
    --seed 20270607 \
    --out-dir "$DATA_DIR" \
    --model-dir "$MODEL_DIR" \
    --mlx-config "$MLX_CONFIG" \
    --adapter-dir "$ADAPTER_DIR"
  patch_quick_config
  echo "Prepared SFT data: $DATA_DIR"
  echo "Prepared MLX config: $MLX_CONFIG"
}

step_train() {
  require_model
  if [[ ! -f "$MLX_CONFIG" ]]; then
    echo "Missing $MLX_CONFIG; running prepare first." >&2
    step_prepare
  fi
  mkdir -p "$LOG_DIR"
  echo "Training Qwen3 smoke: iters=$SMOKE_ITERS adapter=$ADAPTER_DIR"
  .venv/bin/python -m mlx_lm lora --config "$MLX_CONFIG" 2>&1 | tee "$TRAIN_LOG"
}

step_eval() {
  require_model
  if [[ ! -f "$ADAPTER_DIR/adapters.safetensors" ]]; then
    echo "Missing adapter weights: $ADAPTER_DIR/adapters.safetensors; run train first." >&2
    exit 1
  fi
  mkdir -p local/results

  echo "Evaluating base model on $EVAL_LIMIT validation rows"
  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$DATA_DIR/valid.jsonl" \
    --eval-data hf_release/mempatch/validation/scenarios.jsonl \
    --model "$MODEL_DIR" \
    --no-adapter \
    --limit "$EVAL_LIMIT" \
    --max-tokens "$MAX_TOKENS" \
    --out-predictions "local/results/${RUN_NAME}_base_valid${EVAL_LIMIT}_predictions.jsonl" \
    --out-metrics "local/results/${RUN_NAME}_base_valid${EVAL_LIMIT}_metrics.json"

  echo "Evaluating LoRA adapter on $EVAL_LIMIT validation rows"
  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$DATA_DIR/valid.jsonl" \
    --eval-data hf_release/mempatch/validation/scenarios.jsonl \
    --model "$MODEL_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --limit "$EVAL_LIMIT" \
    --max-tokens "$MAX_TOKENS" \
    --out-predictions "local/results/${RUN_NAME}_lora_valid${EVAL_LIMIT}_predictions.jsonl" \
    --out-metrics "local/results/${RUN_NAME}_lora_valid${EVAL_LIMIT}_metrics.json"

  .venv/bin/python scripts/analyze_mlx_lora_errors.py \
    --data hf_release/mempatch/validation/scenarios.jsonl \
    --base-predictions "local/results/${RUN_NAME}_base_valid${EVAL_LIMIT}_predictions.jsonl" \
    --lora-predictions "local/results/${RUN_NAME}_lora_valid${EVAL_LIMIT}_predictions.jsonl" \
    --out-json "local/results/${RUN_NAME}_valid${EVAL_LIMIT}_error_analysis.json" \
    --show-cases 5
}

print_help() {
  cat <<EOF
Qwen3 MemPatch quick smoke LoRA

Defaults:
  model:      $MODEL_DIR
  data:       $DATA_DIR
  adapter:    $ADAPTER_DIR
  config:     $MLX_CONFIG
  train log:  $TRAIN_LOG
  iters:      $SMOKE_ITERS
  eval limit: $EVAL_LIMIT

Commands:
  prepare   Generate smoke SFT data and quick MLX config
  train     Run quick MLX LoRA training
  eval      Evaluate base and LoRA on a small validation slice
  all       prepare + train + eval

Example:
  SMOKE_ITERS=8 EVAL_LIMIT=5 ./scripts/run_qwen3_lora_smoke.sh all
EOF
}

case "${1:-all}" in
  prepare) step_prepare ;;
  train) step_train ;;
  eval) step_eval ;;
  all) step_prepare; step_train; step_eval ;;
  help|-h|--help) print_help ;;
  *)
    echo "Unknown command: $1" >&2
    print_help
    exit 1
    ;;
esac
