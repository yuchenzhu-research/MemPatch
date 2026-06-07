#!/usr/bin/env bash
# MemPatch ablation bench: train 3 MLX models (Path B LoRA) and compare Path A vs Path B.
#
# Path B (DirectJudge): SFT on response JSON -> MLX LoRA -> benchmark score
# Path A (Full module): MLX typed actions -> DPA -> projection -> benchmark score
#
# Usage:
#   ./scripts/run_ablation_bench.sh
#   L3_CASES=25 L4_CASES=25 TRAIN_PROFILE=bench ./scripts/run_ablation_bench.sh
#   SKIP_TRAIN=1 ./scripts/run_ablation_bench.sh
#
# Budget (~90 min with defaults on M-series 14B-class models):
#   - 3 x LoRA train (bench/32 iters): ~35-45 min
#   - 3 models x (Path B LoRA + Path A base) x 2 splits x 50 cases: ~40-50 min
#   Optional Path B base adds ~15-20 min (set SKIP_PATH_B_BASE=1 to skip).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT:$ROOT/src"

L3_CASES="${L3_CASES:-50}"
L4_CASES="${L4_CASES:-50}"
TRAIN_PROFILE="${TRAIN_PROFILE:-bench}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PATH_B_BASE="${SKIP_PATH_B_BASE:-1}"
SEED="${SEED:-20270607}"

ABLATION_DATA="$ROOT/local/train_data/ablation"
SHARED_SFT="$ABLATION_DATA/shared"
RESULTS="$ROOT/local/results/ablation"
LOGS="$ROOT/local/logs/ablation"

declare -a MODEL_SLUGS=(qwen3 gemma3 deepseek_r1)
declare -a MODEL_DIRS=(
  "$ROOT/local/models/Qwen3-14B-MLX-4bit"
  "$ROOT/local/models/gemma-3-12b-it-4bit"
  "$ROOT/local/models/DeepSeek-R1-Distill-Qwen-14B-4bit"
)

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require_model() {
  local model_dir="$1"
  if [[ ! -f "$model_dir/config.json" ]]; then
    echo "error: MLX model not found: $model_dir" >&2
    exit 1
  fi
}

run_path_b() {
  local slug="$1"
  local variant="$2"
  local split="$3"
  local model_dir="$4"
  local adapter_dir="$5"
  local sft_data="$6"
  local eval_scenarios="$7"
  local split_tag="$8"

  local out_dir="$RESULTS/$slug"
  mkdir -p "$out_dir"
  local pred="$out_dir/pathB_${variant}_${split_tag}_predictions.jsonl"
  local metrics="$out_dir/pathB_${variant}_${split_tag}_metrics.json"

  local -a adapter_args=(--no-adapter)
  if [[ "$variant" == "lora" ]]; then
    adapter_args=(--adapter-path "$adapter_dir")
  fi

  log "Path B | model=$slug variant=$variant split=$split_tag"
  "$PYTHON" "$ROOT/scripts/run_mlx_lora_smoke_eval.py" \
    --data "$sft_data" \
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

  log "Train Path B LoRA | model=$slug profile=$TRAIN_PROFILE"
  "$PYTHON" "$ROOT/scripts/prepare_mempatch_v13_smoke.py" \
    --profile "$TRAIN_PROFILE" \
    --out-dir "$SHARED_SFT" \
    --model-dir "$model_dir" \
    --adapter-dir "$adapter_dir" \
    --mlx-config "$mlx_config" \
    --config-only
  "$PYTHON" -m mlx_lm lora --config "$mlx_config"
}

main() {
  log "MemPatch ablation bench | L3=$L3_CASES L4=$L4_CASES profile=$TRAIN_PROFILE"
  for model_dir in "${MODEL_DIRS[@]}"; do
    require_model "$model_dir"
  done

  log "Phase 1/4: build eval slices + shared Path B SFT"
  "$PYTHON" "$ROOT/scripts/build_ablation_eval_slices.py" \
    --out-dir "$ABLATION_DATA" \
    --l3-count "$L3_CASES" \
    --l4-count "$L4_CASES" \
    --seed "$SEED"

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    log "Phase 2/4: prepare shared train SFT (500 smoke quotas)"
    "$PYTHON" "$ROOT/scripts/prepare_mempatch_v13_smoke.py" \
      --profile "$TRAIN_PROFILE" \
      --out-dir "$SHARED_SFT" \
      --model-dir "${MODEL_DIRS[0]}" \
      --adapter-dir "$ROOT/local/adapters/ablation_placeholder" \
      --mlx-config "$LOGS/placeholder.yaml" \
      --seed "$SEED"
  else
    log "Phase 2/4: SKIP_TRAIN=1"
  fi

  local idx
  for idx in "${!MODEL_SLUGS[@]}"; do
    local slug="${MODEL_SLUGS[$idx]}"
    local model_dir="${MODEL_DIRS[$idx]}"
    local adapter_dir="$ROOT/local/adapters/${slug}_pathB_lora"
    local mlx_config="$LOGS/$slug/mlx_lora.yaml"
    mkdir -p "$LOGS/$slug"

    if [[ "$SKIP_TRAIN" != "1" ]]; then
      train_path_b_lora "$slug" "$model_dir" "$adapter_dir" "$mlx_config"
    fi

    log "Phase 3/4: eval model=$slug"
    if [[ "$SKIP_PATH_B_BASE" != "1" ]]; then
      run_path_b "$slug" base l3 "$model_dir" "$adapter_dir" \
        "$ABLATION_DATA/eval_l3_valid50/sft.jsonl" \
        "$ABLATION_DATA/eval_l3_valid50/scenarios.jsonl" l3
      run_path_b "$slug" base l4 "$model_dir" "$adapter_dir" \
        "$ABLATION_DATA/eval_l4_hard50/sft.jsonl" \
        "$ABLATION_DATA/eval_l4_hard50/scenarios.jsonl" l4
    fi

    run_path_b "$slug" lora l3 "$model_dir" "$adapter_dir" \
      "$ABLATION_DATA/eval_l3_valid50/sft.jsonl" \
      "$ABLATION_DATA/eval_l3_valid50/scenarios.jsonl" l3
    run_path_b "$slug" lora l4 "$model_dir" "$adapter_dir" \
      "$ABLATION_DATA/eval_l4_hard50/sft.jsonl" \
      "$ABLATION_DATA/eval_l4_hard50/scenarios.jsonl" l4

    run_path_a "$slug" "$model_dir" \
      "$ABLATION_DATA/eval_l3_valid50/scenarios.jsonl" l3
    run_path_a "$slug" "$model_dir" \
      "$ABLATION_DATA/eval_l4_hard50/scenarios.jsonl" l4
  done

  log "Phase 4/4: summarize"
  "$PYTHON" "$ROOT/scripts/summarize_ablation_matrix.py" \
    --results-dir "$RESULTS" \
    --out-json "$RESULTS/ablation_matrix.json"

  log "Done. Results -> $RESULTS"
}

main "$@"
