#!/usr/bin/env bash
# Heavy MemPatch LoRA training for additional open-weight models (train only).
#
# Same heavy profile as Qwen3-14B:
#   2700 train / 200 valid SFT bundle (shared)
#   batch=2, grad_accum=4, rank=16, 1024 iters, 7 LoRA layers
#
# Usage (repo root):
#   chmod +x scripts/run_mempatch_heavy_lora_models.sh
#   ./scripts/run_mempatch_heavy_lora_models.sh                    # help
#   ./scripts/run_mempatch_heavy_lora_models.sh download gemma3-12b
#   ./scripts/run_mempatch_heavy_lora_models.sh download deepseek-r1-14b
#   ./scripts/run_mempatch_heavy_lora_models.sh prepare gemma3-12b
#   ./scripts/run_mempatch_heavy_lora_models.sh train gemma3-12b
#   ./scripts/run_mempatch_heavy_lora_models.sh train deepseek-r1-14b

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.:src

DATA_DIR="local/train_data/mempatch_v13_heavy"

model_slug() {
  case "$1" in
    gemma3-12b) echo "gemma3_12b_mempatch_v13_heavy" ;;
    deepseek-r1-14b) echo "deepseek_r1_14b_mempatch_v13_heavy" ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

model_dir() {
  case "$1" in
    gemma3-12b) echo "local/models/gemma-3-12b-it-4bit" ;;
    deepseek-r1-14b) echo "local/models/DeepSeek-R1-Distill-Qwen-14B-4bit" ;;
    *) return 1 ;;
  esac
}

log_dir_for() {
  echo "local/logs/$(model_slug "$1")"
}

adapter_dir_for() {
  echo "local/adapters/$(model_slug "$1")"
}

mlx_config_for() {
  echo "$(log_dir_for "$1")/mlx_lora.yaml"
}

step_download() {
  local model="$1"
  .venv/bin/python scripts/download_mlx_model.py --preset "$model"
}

step_prepare() {
  local model="$1"
  local slug
  slug="$(model_slug "$model")"
  mkdir -p "$(log_dir_for "$model")" "$(adapter_dir_for "$model")"
  if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
    echo "Shared heavy SFT missing; generating once from MemPatch train split..."
    .venv/bin/python scripts/prepare_mempatch_v13_smoke.py \
      --profile heavy \
      --full-train \
      --seed 20270607 \
      --out-dir "$DATA_DIR" \
      --model-dir "$(model_dir "$model")" \
      --mlx-config "$(mlx_config_for "$model")" \
      --adapter-dir "$(adapter_dir_for "$model")"
  else
    echo "Reusing shared SFT at $DATA_DIR; writing MLX config for $slug"
    .venv/bin/python scripts/prepare_mempatch_v13_smoke.py \
      --profile heavy \
      --config-only \
      --out-dir "$DATA_DIR" \
      --model-dir "$(model_dir "$model")" \
      --mlx-config "$(mlx_config_for "$model")" \
      --adapter-dir "$(adapter_dir_for "$model")"
  fi
}

step_train() {
  local model="$1"
  local cfg log
  cfg="$(mlx_config_for "$model")"
  log="$(log_dir_for "$model")/train.log"
  if [[ ! -f "$cfg" ]]; then
    echo "Missing $cfg — run: $0 prepare $model" >&2
    exit 1
  fi
  if [[ ! -d "$(model_dir "$model")" ]]; then
    echo "Missing base model $(model_dir "$model") — run: $0 download $model" >&2
    exit 1
  fi
  echo "Training $model (1024 iters) -> log: $log"
  mkdir -p "$(dirname "$log")"
  .venv/bin/python -m mlx_lm lora --config "$cfg" 2>&1 | tee "$log"
}

print_help() {
  cat <<EOF
Heavy MemPatch LoRA — additional models (train only, no eval)

Shared SFT: $DATA_DIR
Heavy hyperparams: batch=2, grad_accum=4, rank=16, iters=1024 (same as Qwen3-14B)

Models:
  gemma3-12b       mlx-community/gemma-3-12b-it-4bit
  deepseek-r1-14b   mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit (~8.3 GB)

Typical two-terminal workflow:

  # Terminal A
  export HTTP_PROXY=http://127.0.0.1:1082 HTTPS_PROXY=http://127.0.0.1:1082
  ./scripts/run_mempatch_heavy_lora_models.sh download gemma3-12b
  ./scripts/run_mempatch_heavy_lora_models.sh prepare gemma3-12b
  ./scripts/run_mempatch_heavy_lora_models.sh train gemma3-12b

  # Terminal B
  export HTTP_PROXY=http://127.0.0.1:1082 HTTPS_PROXY=http://127.0.0.1:1082
  ./scripts/run_mempatch_heavy_lora_models.sh download deepseek-r1-14b
  ./scripts/run_mempatch_heavy_lora_models.sh prepare deepseek-r1-14b
  ./scripts/run_mempatch_heavy_lora_models.sh train deepseek-r1-14b

Memory note (~48 GB unified): avoid running both trains while Qwen test500 eval is active.
Do not git add local/.
EOF
}

case "${1:-}" in
  download) step_download "${2:?model}" ;;
  prepare)  step_prepare "${2:?model}" ;;
  train)    step_train "${2:?model}" ;;
  ""|help|-h|--help) print_help ;;
  *)
    echo "Unknown command: $1" >&2
    print_help
    exit 1
    ;;
esac
