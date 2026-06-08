#!/usr/bin/env bash
# MemPatch paper pipeline (single entry): download → smoke → LoRA → test500 → export/plot
#
# Default models (48GB Mac, same paper LoRA on all rows):
#   qwen3_14b, gemma3_12b, mistral_nemo_12b, llama3_1_8b
#
# Usage:
#   bash scripts/run_paper_pipeline.sh                    # full run
#   bash scripts/run_paper_pipeline.sh --check-only       # dataset + HF mirror check
#   bash scripts/run_paper_pipeline.sh --download-only    # download + verify only
#   bash scripts/run_paper_pipeline.sh --download-bg      # download in background, exit
#   SKIP_DOWNLOAD=1 bash scripts/run_paper_pipeline.sh    # skip download phase
#   SKIP_TRAIN=1 bash scripts/run_paper_pipeline.sh       # eval + export only
#   RESUME_LORA=1 FORCE_TRAIN=1 ...                     # continue LoRA from latest checkpoint
#   USE_MIRROR=0 bash scripts/run_paper_pipeline.sh       # huggingface.co direct
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/src:$ROOT/scripts}"

USE_MIRROR="${USE_MIRROR:-1}"
PAPER_TRAIN_PROFILE="${PAPER_TRAIN_PROFILE:-paper}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
MODELS="${MODELS:-qwen3_14b,gemma3_12b,mistral_nemo_12b,llama3_1_8b}"
SEED="${SEED:-42}"

RESULTS="${RESULTS:-$ROOT/local/results/paper}"
PAPER_DATA="${PAPER_DATA:-$ROOT/local/train_data/paper}"
SHARED_SFT="${SHARED_SFT:-$PAPER_DATA/shared_sft}"
TEST_BUNDLE="${TEST_BUNDLE:-$PAPER_DATA/test500}"
LOGS="${LOGS:-$ROOT/local/logs/paper}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$ROOT/local/adapters}"

DOWNLOAD_PRESETS="qwen3-14b gemma3-12b mistral-nemo-12b llama-3.1-8b-instruct"

model_key_preset() {
  case "$1" in
    qwen3_14b) echo "qwen3-14b" ;;
    gemma3_12b) echo "gemma3-12b" ;;
    mistral_nemo_12b) echo "mistral-nemo-12b" ;;
    llama3_1_8b) echo "llama-3.1-8b-instruct" ;;
    gemma4_12b) echo "gemma4-12b" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_key_slug() {
  case "$1" in
    qwen3_14b) echo "qwen3_14b" ;;
    gemma3_12b) echo "gemma3_12b" ;;
    mistral_nemo_12b) echo "mistral_nemo_12b" ;;
    llama3_1_8b) echo "llama3_1_8b" ;;
    gemma4_12b) echo "gemma4_12b" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_key_local_name() {
  case "$1" in
    qwen3_14b) echo "Qwen3-14B-MLX-4bit" ;;
    gemma3_12b) echo "gemma-3-12b-it-4bit" ;;
    mistral_nemo_12b) echo "Mistral-Nemo-Instruct-2407-4bit" ;;
    llama3_1_8b) echo "Meta-Llama-3.1-8B-Instruct-4bit" ;;
    gemma4_12b) echo "gemma-4-12B-it-4bit" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

adapter_train_complete() {
  [[ -f "$1/adapters.safetensors" ]]
}

log() { printf '[pipeline] %s\n' "$*"; }

require_python() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "error: missing venv python: $PYTHON" >&2
    exit 1
  fi
}

mirror_args() {
  MIRROR_ARGS=()
  if [[ "$USE_MIRROR" == "1" ]]; then
    MIRROR_ARGS=(--mirror --disable-xet)
  fi
}

download_cli_args() {
  mirror_args
  DOWNLOAD_CLI=(
    --max-workers 1
    --retries 10
    --timeout 300
    "${MIRROR_ARGS[@]}"
  )
}

preset_local_name() {
  case "$1" in
    qwen3-14b) echo "Qwen3-14B-MLX-4bit" ;;
    gemma3-12b) echo "gemma-3-12b-it-4bit" ;;
    mistral-nemo-12b) echo "Mistral-Nemo-Instruct-2407-4bit" ;;
    llama-3.1-8b-instruct) echo "Meta-Llama-3.1-8B-Instruct-4bit" ;;
    gemma4-12b) echo "gemma-4-12B-it-4bit" ;;
    *) echo "$1" ;;
  esac
}

preset_status() {
  local preset="$1"
  local local_name
  local_name="$(preset_local_name "$preset")"
  if [[ -d "$ROOT/local/models/$local_name" ]] && "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
      --preset "$preset" --verify-local >/dev/null 2>&1; then
    echo "complete"
  elif [[ -d "$ROOT/local/models/$local_name" ]]; then
    echo "partial"
  else
    echo "missing"
  fi
}

print_model_status() {
  printf '\n%-22s %-12s %s\n' "PRESET" "STATUS" "LOCAL_DIR"
  printf '%.0s-' {1..70}; echo
  local preset local_name
  for preset in $DOWNLOAD_PRESETS; do
    local_name="$(preset_local_name "$preset")"
    printf '%-22s %-12s %s\n' "$preset" "$(preset_status "$preset")" "$ROOT/local/models/$local_name"
  done
  echo
}

phase_audit_dataset() {
  log "Dataset audit (4000 scenarios, decision-boundary gate)"
  PYTHONPATH=.:src "$PYTHON" "$ROOT/scripts/audit_decision_boundary.py" \
    --data "$ROOT/hf_release/mempatch/train" \
    --data "$ROOT/hf_release/mempatch/validation" \
    --data "$ROOT/hf_release/mempatch/test"
}

phase_check_downloadable() {
  log "HF mirror connectivity (--check, no weights)"
  mirror_args
  for preset in $DOWNLOAD_PRESETS; do
    log "  check preset=$preset"
    "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
      --preset "$preset" --check "${MIRROR_ARGS[@]}"
  done
}

download_preset() {
  local preset="$1"
  local local_name
  local_name="$(preset_local_name "$preset")"
  download_cli_args

  if [[ "$(preset_status "$preset")" == "complete" ]]; then
    log "already complete: $local_name"
    return 0
  fi

  log "downloading preset=$preset -> $local_name (mirror=$USE_MIRROR)"
  "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
    --preset "$preset" "${DOWNLOAD_CLI[@]}"
  "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
    --preset "$preset" --verify-local
}

download_preset_background() {
  local preset="$1"
  local log_file="$LOGS/download_${preset//-/_}.log"
  download_cli_args

  if [[ "$(preset_status "$preset")" == "complete" ]]; then
    log "skip background (complete): $(preset_local_name "$preset")"
    return 0
  fi
  if pgrep -f "download_mlx_model.py --preset ${preset}" >/dev/null 2>&1; then
    log "skip background (running): $preset"
    return 0
  fi

  mkdir -p "$LOGS"
  log "background download preset=$preset -> $log_file"
  nohup "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
    --preset "$preset" "${DOWNLOAD_CLI[@]}" >>"$log_file" 2>&1 &
  log "  pid=$! log=$log_file"
}

phase_download() {
  for preset in $DOWNLOAD_PRESETS; do
    download_preset "$preset"
  done
  print_model_status
}

phase_download_background() {
  for preset in $DOWNLOAD_PRESETS; do
    download_preset_background "$preset"
  done
  print_model_status
}

run_path_b() {
  local slug="$1" variant="$2" model_dir="$3" adapter_dir="$4"
  local sft_jsonl="$5" eval_scenarios="$6" split_tag="$7"
  local out_dir="$RESULTS/$slug"
  mkdir -p "$out_dir"
  local adapter_args=(--no-adapter)
  [[ "$variant" == "lora" ]] && adapter_args=(--adapter-path "$adapter_dir")

  log "Path B | model=$slug variant=$variant split=$split_tag"
  "$PYTHON" "$ROOT/scripts/run_mlx_lora_smoke_eval.py" \
    --data "$sft_jsonl" --model "$model_dir" "${adapter_args[@]}" \
    --out-predictions "$out_dir/pathB_${variant}_${split_tag}_predictions.jsonl" \
    --eval-data "$eval_scenarios" \
    --out-metrics "$out_dir/pathB_${variant}_${split_tag}_metrics.json" \
    --model-tag "$slug" --variant-tag "$variant" --split-tag "$split_tag" \
    --max-tokens 256 --temp 0.0
}

run_path_a() {
  local slug="$1" model_dir="$2" eval_scenarios="$3" split_tag="$4"
  local out_dir="$RESULTS/$slug"
  mkdir -p "$out_dir"
  log "Path A | model=$slug split=$split_tag"
  "$PYTHON" "$ROOT/scripts/run_mlx_revision_module_eval.py" \
    --data "$eval_scenarios" --model "$model_dir" --no-adapter \
    --out-predictions "$out_dir/pathA_base_${split_tag}_predictions.jsonl" \
    --eval-data "$eval_scenarios" \
    --out-metrics "$out_dir/pathA_base_${split_tag}_metrics.json" \
    --model-tag "$slug" --variant-tag base --split-tag "$split_tag" \
    --max-tokens 512 --temp 0.0
}

write_training_manifest() {
  local slug="$1" adapter_dir="$2" mlx_config="$3" profile="$4"
  "$PYTHON" - "$adapter_dir" "$mlx_config" "$profile" "$slug" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_mempatch_v13_smoke import MLX_PROFILES

adapter_dir = Path(sys.argv[1])
config_path = Path(sys.argv[2])
profile = sys.argv[3]
slug = sys.argv[4]
target_iters = int(MLX_PROFILES[profile]["iters"])
checkpoints = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
completed_iters = int(checkpoints[-1].name.split("_", 1)[0]) if checkpoints else 0
yaml_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
mask_match = re.search(r"^mask_prompt:\s*(\S+)\s*$", yaml_text, re.MULTILINE)
manifest = {
    "model": slug,
    "profile": profile,
    "target_iters": target_iters,
    "completed_iters": completed_iters,
    "checkpoints": [path.name for path in checkpoints],
    "adapter_config": str(adapter_dir / "adapter_config.json"),
    "mlx_config": str(config_path),
    "mask_prompt": mask_match.group(1) if mask_match else None,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
manifest_path = adapter_dir / "training_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[pipeline] wrote {manifest_path} ({completed_iters}/{target_iters} iters)")
PY
}

train_path_b_lora() {
  local slug="$1" model_dir="$2" adapter_dir="$3" mlx_config="$4" profile="$5"
  log "Train Path B LoRA | model=$slug profile=$profile"
  "$PYTHON" "$ROOT/scripts/prepare_mempatch_v13_smoke.py" \
    --profile "$profile" --out-dir "$SHARED_SFT" \
    --model-dir "$model_dir" --adapter-dir "$adapter_dir" \
    --mlx-config "$mlx_config" --config-only
  mkdir -p "$adapter_dir"
  if [[ "${RESUME_LORA:-0}" == "1" ]]; then
    "$PYTHON" - "$mlx_config" "$adapter_dir" "$profile" <<'PY'
import re
import sys
from pathlib import Path

from prepare_mempatch_v13_smoke import MLX_PROFILES

config_path = Path(sys.argv[1])
adapter_dir = Path(sys.argv[2])
profile = sys.argv[3]
target_iters = int(MLX_PROFILES[profile]["iters"])
text = config_path.read_text(encoding="utf-8")
checkpoints = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
if not checkpoints:
    raise SystemExit(f"error: RESUME_LORA=1 but no checkpoint in {adapter_dir}")
latest = checkpoints[-1]
done = int(latest.name.split("_", 1)[0])
remaining = max(target_iters - done, 1)
text = re.sub(r"^iters:\s*\d+\s*$", f"iters: {remaining}", text, count=1, flags=re.MULTILINE)
resume_line = f'resume_adapter_file: "{latest.resolve()}"'
if re.search(r"^resume_adapter_file:", text, re.MULTILINE):
    text = re.sub(r"^resume_adapter_file:.*$", resume_line, text, count=1, flags=re.MULTILINE)
else:
    text = text.rstrip() + "\n" + resume_line + "\n"
config_path.write_text(text, encoding="utf-8")
print(f"[pipeline] resume LoRA from iter {done}, train {remaining} more (target {target_iters})")
PY
  fi
  "$PYTHON" -m mlx_lm lora --config "$mlx_config"
  write_training_manifest "$slug" "$adapter_dir" "$mlx_config" "$profile"
}

phase_train_eval_export() {
  mkdir -p "$RESULTS" "$LOGS" "$PAPER_DATA"
  IFS=',' read -r -a MODEL_KEYS <<< "$MODELS"

  log "build test500 eval bundle"
  "$PYTHON" "$ROOT/scripts/build_paper_eval_bundle.py" --out-dir "$TEST_BUNDLE"

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    log "shared SFT quotas (profile=$PAPER_TRAIN_PROFILE)"
    "$PYTHON" "$ROOT/scripts/prepare_mempatch_v13_smoke.py" \
      --profile "$PAPER_TRAIN_PROFILE" --out-dir "$SHARED_SFT" \
      --model-dir "$ROOT/local/models/$(model_key_local_name "${MODEL_KEYS[0]}")" \
      --adapter-dir "$ROOT/local/adapters/paper_placeholder" \
      --mlx-config "$LOGS/placeholder.yaml" --seed "$SEED"
  fi

  for key in "${MODEL_KEYS[@]}"; do
    local slug model_dir
    slug="$(model_key_slug "$key")"
    model_dir="$ROOT/local/models/$(model_key_local_name "$key")"
    local adapter_dir="$ADAPTER_ROOT/${slug}_pathB_lora"
    local mlx_config="$LOGS/$slug/mlx_lora.yaml"
    mkdir -p "$LOGS/$slug"

    if [[ ! -d "$model_dir" ]]; then
      echo "error: missing model $model_dir (run download first)" >&2
      exit 1
    fi
    if [[ "$SKIP_TRAIN" != "1" ]]; then
      "$PYTHON" "$ROOT/scripts/check_mlx_lora_model.py" "$model_dir"
    fi

    if [[ "$SKIP_TRAIN" != "1" ]]; then
      if adapter_train_complete "$adapter_dir" && [[ "${FORCE_TRAIN:-0}" != "1" ]]; then
        log "skip LoRA train (adapter exists): $slug"
      else
        train_path_b_lora "$slug" "$model_dir" "$adapter_dir" "$mlx_config" "$PAPER_TRAIN_PROFILE"
      fi
    fi

    run_path_b "$slug" lora "$model_dir" "$adapter_dir" \
      "$TEST_BUNDLE/sft.jsonl" "$TEST_BUNDLE/scenarios.jsonl" test500
    run_path_a "$slug" "$model_dir" "$TEST_BUNDLE/scenarios.jsonl" test500
  done

  log "export benchmark-paper assets"
  "$PYTHON" "$ROOT/scripts/export_benchmark_paper_assets.py" \
    --results-dir "$RESULTS" \
    --out-dir "$RESULTS/export/benchmark_paper" \
    --primary-split test500

  if "$PYTHON" -c "import matplotlib" 2>/dev/null; then
    "$PYTHON" "$ROOT/scripts/plot_benchmark_paper_figures.py" \
      --export-dir "$RESULTS/export/benchmark_paper" \
      --out-dir "$RESULTS/export/benchmark_paper/figures"
  else
    log "matplotlib not installed; skip PNG (pip install matplotlib)"
  fi
}

main() {
  require_python
  mkdir -p "$LOGS"

  case "${1:-full}" in
    --check-only)
      phase_audit_dataset
      phase_check_downloadable
      print_model_status
      ;;
    --download-only)
      phase_check_downloadable
      phase_download
      ;;
    --download-bg)
      phase_download_background
      ;;
    full|"")
      phase_audit_dataset
      if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
        phase_check_downloadable
        phase_download
      else
        log "SKIP_DOWNLOAD=1"
        print_model_status
      fi
      phase_train_eval_export
      log "Done -> $RESULTS/export/benchmark_paper/"
      ;;
    *)
      echo "usage: $0 [--check-only|--download-only|--download-bg|full]" >&2
      exit 1
      ;;
  esac
}

main "$@"
