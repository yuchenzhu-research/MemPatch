#!/usr/bin/env bash
# MemPatch paper experiments: next-gen open models (Mac-feasible tier).
#
# Models (local MLX):
#   qwen35_27b   — mlx-community/Qwen3.5-27B-4bit  (Qwen3.5 flagship practical tier)
#   gemma4_12b   — mlx-community/gemma-4-12B-it-4bit
#   deepseek_r1  — mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit (reasoning proxy; V4-Pro/Flash too large locally)
#
# NOT local: Qwen3.5-397B-A17B MoE, DeepSeek-V4-Pro/Flash — use API row separately if needed.
#
# MemPatch paper experiments (called by run_paper_pipeline.sh for train/eval/export).
#
# Prefer the full pipeline:
#   bash scripts/run_paper_pipeline.sh
#
# This script alone (skip download if models ready):
#   SKIP_DOWNLOAD=1 bash scripts/run_paper_experiments.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/src:$ROOT/scripts}"
RESULTS="${RESULTS:-$ROOT/local/results/paper}"
PAPER_DATA="${PAPER_DATA:-$ROOT/local/train_data/paper}"
SHARED_SFT="${SHARED_SFT:-$PAPER_DATA/shared_sft}"
TEST_BUNDLE="${TEST_BUNDLE:-$PAPER_DATA/test500}"
LOGS="${LOGS:-$ROOT/local/logs/paper}"
SEED="${SEED:-42}"

# paper profile: rank-16 LoRA, 256 iters (loss usually flat by ~200)
PAPER_TRAIN_PROFILE="${PAPER_TRAIN_PROFILE:-paper}"
QWEN35_TRAIN_PROFILE="${QWEN35_TRAIN_PROFILE:-$PAPER_TRAIN_PROFILE}"
OTHER_TRAIN_PROFILE="${OTHER_TRAIN_PROFILE:-$PAPER_TRAIN_PROFILE}"
# Set USE_MIRROR=0 to download from huggingface.co directly
USE_MIRROR="${USE_MIRROR:-1}"

MODELS="${MODELS:-qwen35_27b,gemma4_12b,deepseek_r1}"

declare -A PRESET_SLUG=(
  [qwen35_27b]="qwen35-27b"
  [gemma4_12b]="gemma4-12b"
  [deepseek_r1]="deepseek-r1-14b"
)
declare -A MODEL_SLUG=(
  [qwen35_27b]="qwen35_27b"
  [gemma4_12b]="gemma4_12b"
  [deepseek_r1]="deepseek_r1_14b"
)
declare -A LOCAL_NAME=(
  [qwen35_27b]="Qwen3.5-27B-4bit"
  [gemma4_12b]="gemma-4-12B-it-4bit"
  [deepseek_r1]="DeepSeek-R1-Distill-Qwen-14B-4bit"
)

log() { printf '[paper] %s\n' "$*"; }

require_python() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "missing venv python: $PYTHON" >&2
    exit 1
  fi
}

download_model() {
  local preset="$1"
  local local_name="$2"
  local model_dir="$ROOT/local/models/$local_name"
  local mirror_args=()
  if [[ "$USE_MIRROR" == "1" ]]; then
    mirror_args=(--mirror --disable-xet)
  fi
  if [[ -d "$model_dir" ]] && "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
      --preset "$preset" --verify-local 2>/dev/null; then
    log "model ok: $local_name"
    return 0
  fi
  log "downloading preset=$preset -> $local_name (mirror=$USE_MIRROR)"
  "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
    --preset "$preset" \
    --max-workers 1 \
    --retries 10 \
    --timeout 300 \
    "${mirror_args[@]}"
  "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
    --preset "$preset" \
    --verify-local
}

require_model() {
  local model_dir="$1"
  if [[ ! -d "$model_dir" ]]; then
    echo "missing model dir: $model_dir" >&2
    exit 1
  fi
}

run_path_b() {
  local slug="$1"
  local variant="$2"
  local model_dir="$3"
  local adapter_dir="$4"
  local sft_jsonl="$5"
  local eval_scenarios="$6"
  local split_tag="$7"

  local out_dir="$RESULTS/$slug"
  mkdir -p "$out_dir"
  local pred="$out_dir/pathB_${variant}_${split_tag}_predictions.jsonl"
  local metrics="$out_dir/pathB_${variant}_${split_tag}_metrics.json"
  local adapter_args=()
  if [[ "$variant" == "lora" ]]; then
    adapter_args=(--adapter-path "$adapter_dir")
  else
    adapter_args=(--no-adapter)
  fi

  log "Path B | model=$slug variant=$variant split=$split_tag"
  "$PYTHON" "$ROOT/scripts/run_mlx_lora_smoke_eval.py" \
    --data "$sft_jsonl" \
    --model "$model_dir" \
    "${adapter_args[@]}" \
    --out-predictions "$pred" \
    --eval-data "$eval_scenarios" \
    --out-metrics "$metrics" \
    --model-tag "$slug" \
    --variant-tag "$variant" \
    --split-tag "$split_tag" \
    --max-tokens 256 \
    --temp 0.0
}

run_path_a() {
  local slug="$1"
  local model_dir="$2"
  local eval_scenarios="$3"
  local split_tag="$4"

  local out_dir="$RESULTS/$slug"
  mkdir -p "$out_dir"
  local pred="$out_dir/pathA_base_${split_tag}_predictions.jsonl"
  local metrics="$out_dir/pathA_base_${split_tag}_metrics.json"

  log "Path A | model=$slug variant=base split=$split_tag"
  "$PYTHON" "$ROOT/scripts/run_mlx_revision_module_eval.py" \
    --data "$eval_scenarios" \
    --model "$model_dir" \
    --no-adapter \
    --out-predictions "$pred" \
    --eval-data "$eval_scenarios" \
    --out-metrics "$metrics" \
    --model-tag "$slug" \
    --variant-tag base \
    --split-tag "$split_tag" \
    --max-tokens 512 \
    --temp 0.0
}

train_path_b_lora() {
  local slug="$1"
  local model_dir="$2"
  local adapter_dir="$3"
  local mlx_config="$4"
  local profile="$5"

  log "Train Path B LoRA | model=$slug profile=$profile"
  "$PYTHON" "$ROOT/scripts/prepare_mempatch_v13_smoke.py" \
    --profile "$profile" \
    --out-dir "$SHARED_SFT" \
    --model-dir "$model_dir" \
    --adapter-dir "$adapter_dir" \
    --mlx-config "$mlx_config" \
    --config-only
  "$PYTHON" -m mlx_lm lora --config "$mlx_config"
}

main() {
  require_python
  mkdir -p "$RESULTS" "$LOGS" "$PAPER_DATA"

  IFS=',' read -r -a MODEL_KEYS <<< "$MODELS"

  if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    log "Phase 0/5: download + verify MLX models (profile=$PAPER_TRAIN_PROFILE, mirror=$USE_MIRROR)"
    if [[ "${DOWNLOAD_BACKGROUND:-0}" == "1" ]]; then
      USE_MIRROR="$USE_MIRROR" bash "$ROOT/scripts/download_paper_models.sh" --background
    else
      for key in "${MODEL_KEYS[@]}"; do
        download_model "${PRESET_SLUG[$key]}" "${LOCAL_NAME[$key]}"
      done
    fi
  else
    log "Phase 0/5: SKIP_DOWNLOAD=1"
  fi

  log "Phase 1/5: build test500 eval bundle"
  "$PYTHON" "$ROOT/scripts/build_paper_eval_bundle.py" \
    --out-dir "$TEST_BUNDLE"

  if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    log "Phase 2/5: shared SFT quotas (profile=$PAPER_TRAIN_PROFILE)"
    "$PYTHON" "$ROOT/scripts/prepare_mempatch_v13_smoke.py" \
      --profile "$PAPER_TRAIN_PROFILE" \
      --out-dir "$SHARED_SFT" \
      --model-dir "$ROOT/local/models/${LOCAL_NAME[${MODEL_KEYS[0]}]}" \
      --adapter-dir "$ROOT/local/adapters/paper_placeholder" \
      --mlx-config "$LOGS/placeholder.yaml" \
      --seed "$SEED"
  else
    log "Phase 2/5: SKIP_TRAIN=1"
  fi

  for key in "${MODEL_KEYS[@]}"; do
    local slug="${MODEL_SLUG[$key]}"
    local model_dir="$ROOT/local/models/${LOCAL_NAME[$key]}"
    require_model "$model_dir"
    local adapter_dir="$ROOT/local/adapters/${slug}_pathB_lora"
    local mlx_config="$LOGS/$slug/mlx_lora.yaml"
    mkdir -p "$LOGS/$slug"

    local profile="$OTHER_TRAIN_PROFILE"
    if [[ "$key" == "qwen35_27b" ]]; then
      profile="$QWEN35_TRAIN_PROFILE"
    fi

    if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
      train_path_b_lora "$slug" "$model_dir" "$adapter_dir" "$mlx_config" "$profile"
    fi

    log "Phase 3/5: eval model=$slug on test500"
    run_path_b "$slug" lora "$model_dir" "$adapter_dir" \
      "$TEST_BUNDLE/sft.jsonl" \
      "$TEST_BUNDLE/scenarios.jsonl" test500
    run_path_a "$slug" "$model_dir" \
      "$TEST_BUNDLE/scenarios.jsonl" test500
  done

  log "Phase 4/6: export benchmark-paper assets (figures JSON + tables CSV)"
  "$PYTHON" "$ROOT/scripts/export_benchmark_paper_assets.py" \
    --results-dir "$RESULTS" \
    --out-dir "$RESULTS/export/benchmark_paper" \
    --primary-split test500

  log "Phase 5/6: render figure PNGs (requires matplotlib)"
  if "$PYTHON" -c "import matplotlib" 2>/dev/null; then
    "$PYTHON" "$ROOT/scripts/plot_benchmark_paper_figures.py" \
      --export-dir "$RESULTS/export/benchmark_paper" \
      --out-dir "$RESULTS/export/benchmark_paper/figures"
  else
    log "matplotlib not installed; skip PNG render (pip install matplotlib)"
  fi

  log "Phase 6/6: done -> $RESULTS/export/benchmark_paper/"
}

main "$@"
