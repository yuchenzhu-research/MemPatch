#!/usr/bin/env bash
# MemPatch heavy MLX LoRA workflow (manual steps — do not git add local/).
#
# Hardware note: tuned for Apple Silicon with ~48 GB unified memory (Qwen3-14B-MLX-4bit).
# Heavy profile: batch_size=2, iters=1024, grad_accum=4, rank=16, 7 LoRA layers.
# Expect several hours of training; monitor peak mem in train.log.
#
# Usage:
#   chmod +x scripts/run_mempatch_heavy_lora.sh
#   ./scripts/run_mempatch_heavy_lora.sh          # print instructions only
#   ./scripts/run_mempatch_heavy_lora.sh prepare  # step 1 only
#   ./scripts/run_mempatch_heavy_lora.sh train    # step 2 only (long-running)
#   ./scripts/run_mempatch_heavy_lora.sh eval     # step 3 only (after train)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=.:src

DATA_DIR="local/train_data/mempatch_v13_heavy"
MLX_CONFIG="local/logs/qwen3_14b_mempatch_v13_heavy/mlx_lora.yaml"
ADAPTER_DIR="local/adapters/qwen3_14b_mempatch_v13_heavy"
TRAIN_LOG="local/logs/qwen3_14b_mempatch_v13_heavy/train.log"

step_prepare() {
  echo "== Step 1: Prepare heavy SFT bundle (2700 train, 200 valid, 50 hard) =="
  .venv/bin/python scripts/prepare_mempatch_v13_smoke.py \
    --profile heavy \
    --full-train \
    --seed 20270607 \
    --out-dir "$DATA_DIR" \
    --mlx-config "$MLX_CONFIG" \
    --adapter-dir "$ADAPTER_DIR"
  echo "Wrote: $DATA_DIR/{train,valid,hard_balanced50,hard_balanced50_sft}.jsonl"
  echo "Wrote: $MLX_CONFIG"
}

step_train() {
  echo "== Step 2: Heavy LoRA training (1024 iters — long-running) =="
  echo "Log: $TRAIN_LOG"
  mkdir -p "$(dirname "$TRAIN_LOG")"
  .venv/bin/python -m mlx_lm lora --config "$MLX_CONFIG" 2>&1 | tee "$TRAIN_LOG"
}

step_eval() {
  echo "== Step 3: Eval heavy adapter on valid200 + hard50 =="
  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$DATA_DIR/valid.jsonl" \
    --eval-data hf_release/mempatch/validation/scenarios.jsonl \
    --no-adapter \
    --out-predictions local/results/qwen3_14b_base_heavy_valid_predictions.jsonl \
    --out-metrics local/results/qwen3_14b_base_heavy_valid_metrics.json

  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$DATA_DIR/valid.jsonl" \
    --eval-data hf_release/mempatch/validation/scenarios.jsonl \
    --adapter-path "$ADAPTER_DIR" \
    --out-predictions local/results/qwen3_14b_heavy_valid_predictions.jsonl \
    --out-metrics local/results/qwen3_14b_heavy_valid_metrics.json

  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$DATA_DIR/hard_balanced50_sft.jsonl" \
    --eval-data hf_release/mempatch/test/scenarios.jsonl \
    --no-adapter \
    --out-predictions local/results/qwen3_14b_base_heavy_hard50_predictions.jsonl \
    --out-metrics local/results/qwen3_14b_base_heavy_hard50_metrics.json

  .venv/bin/python scripts/run_mlx_lora_smoke_eval.py \
    --data "$DATA_DIR/hard_balanced50_sft.jsonl" \
    --eval-data hf_release/mempatch/test/scenarios.jsonl \
    --adapter-path "$ADAPTER_DIR" \
    --out-predictions local/results/qwen3_14b_heavy_hard50_predictions.jsonl \
    --out-metrics local/results/qwen3_14b_heavy_hard50_metrics.json

  .venv/bin/python scripts/analyze_mlx_lora_errors.py \
    --base-predictions local/results/qwen3_14b_base_heavy_valid_predictions.jsonl \
    --lora-predictions local/results/qwen3_14b_heavy_valid_predictions.jsonl \
    --out-json local/results/qwen3_14b_heavy_valid_error_analysis.json \
    --show-cases 8

  .venv/bin/python scripts/analyze_mlx_lora_errors.py \
    --data hf_release/mempatch/test/scenarios.jsonl \
    --base-predictions local/results/qwen3_14b_base_heavy_hard50_predictions.jsonl \
    --lora-predictions local/results/qwen3_14b_heavy_hard50_predictions.jsonl \
    --out-json local/results/qwen3_14b_heavy_hard50_error_analysis.json \
    --show-cases 8
}

print_instructions() {
  cat <<'EOF'
MemPatch heavy LoRA — run these steps manually from repo root:

  export PYTHONPATH=.:src

  # 1) Prepare data + MLX config (2700 train, 200 valid, rank-16 LoRA yaml)
  ./scripts/run_mempatch_heavy_lora.sh prepare

  # 2) Train (1024 iters; tee log for peak-mem / loss monitoring)
  ./scripts/run_mempatch_heavy_lora.sh train

  # 3) Eval + error analysis (after training finishes)
  ./scripts/run_mempatch_heavy_lora.sh eval

Or invoke steps individually via the functions above.
Do not git add local/ (models, adapters, logs, results stay gitignored).
EOF
}

case "${1:-}" in
  prepare) step_prepare ;;
  train)   step_train ;;
  eval)    step_eval ;;
  ""|help|-h|--help) print_instructions ;;
  *)
    echo "Unknown step: $1" >&2
    print_instructions
    exit 1
    ;;
esac
