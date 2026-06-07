#!/usr/bin/env bash
# MemPatch paper pipeline — download → verify → DeepSeek smoke → LoRA → test500 → export/plot
#
# Models (MLX, Mac-feasible):
#   qwen35-27b   Qwen3.5-27B-4bit        (~16 GiB, hf-mirror + curl)
#   gemma4-12b   Gemma-4-12B-it-4bit     (~8 GiB)
#   deepseek_r1  DeepSeek-R1-Distill-14B (~7.8 GiB, needs thinking-close + JSON prefill)
#
# Usage:
#   bash scripts/run_paper_pipeline.sh                 # full pipeline
#   bash scripts/run_paper_pipeline.sh --check-only    # can we download? (no weights)
#   bash scripts/run_paper_pipeline.sh --download-only # download + verify only
#   SKIP_DOWNLOAD=1 bash scripts/run_paper_pipeline.sh # train+eval (models already local)
#   USE_MIRROR=0 bash scripts/run_paper_pipeline.sh    # huggingface.co direct
#
# Background download (if you only want weights first):
#   bash scripts/download_paper_models.sh --background
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$ROOT:$ROOT/src:$ROOT/scripts}"

USE_MIRROR="${USE_MIRROR:-1}"
PAPER_TRAIN_PROFILE="${PAPER_TRAIN_PROFILE:-paper}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_DEEPSEEK_SMOKE="${SKIP_DEEPSEEK_SMOKE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

DEEPSEEK_MODEL="$ROOT/local/models/DeepSeek-R1-Distill-Qwen-14B-4bit"
DEEPSEEK_ADAPTER="$ROOT/local/adapters/deepseek_r1_14b_pathB_lora"
SMOKE_DIR="$ROOT/local/train_data/paper/smoke5"
SMOKE_SFT="$SMOKE_DIR/sft.jsonl"
SMOKE_SCENARIOS="$SMOKE_DIR/scenarios.jsonl"
LOGS="$ROOT/local/logs/paper"

log() { printf '[pipeline] %s\n' "$*"; }

require_python() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "error: missing venv python: $PYTHON" >&2
    exit 1
  fi
}

phase_check_downloadable() {
  log "Phase 0/7: check mirror/HF connectivity (metadata + config.json, no weights)"
  USE_MIRROR="$USE_MIRROR" bash "$ROOT/scripts/download_paper_models.sh" --check
}

phase_download() {
  log "Phase 1/7: download + verify all paper models"
  USE_MIRROR="$USE_MIRROR" bash "$ROOT/scripts/download_paper_models.sh"
}

phase_deepseek_smoke() {
  if [[ "$SKIP_DEEPSEEK_SMOKE" == "1" ]]; then
    log "Phase 2/7: SKIP_DEEPSEEK_SMOKE=1"
    return 0
  fi
  if [[ ! -d "$DEEPSEEK_MODEL" ]]; then
    log "Phase 2/7: skip DeepSeek smoke (model not local)"
    return 0
  fi
  log "Phase 2/7: DeepSeek R1 JSON smoke (5 cases, thinking-off + brace prefill)"
  mkdir -p "$LOGS"
  local pred="$LOGS/deepseek_smoke5_predictions.jsonl"
  local metrics="$LOGS/deepseek_smoke5_metrics.json"
  local adapter_args=()
  if [[ -d "$DEEPSEEK_ADAPTER" ]]; then
    adapter_args=(--adapter-path "$DEEPSEEK_ADAPTER")
  else
    log "  (no adapter yet — testing base + R1 prompt fix only)"
    adapter_args=(--no-adapter)
  fi
  if [[ ! -f "$SMOKE_SFT" ]]; then
    log "  building 5-case validation smoke bundle"
    "$PYTHON" "$ROOT/scripts/build_paper_eval_bundle.py" \
      --scenarios "$ROOT/hf_release/mempatch/validation/scenarios.jsonl" \
      --out-dir "$SMOKE_DIR" \
      --limit 5
  fi
  "$PYTHON" "$ROOT/scripts/run_mlx_lora_smoke_eval.py" \
    --model "$DEEPSEEK_MODEL" \
    "${adapter_args[@]}" \
    --data "$SMOKE_SFT" \
    --eval-data "$SMOKE_SCENARIOS" \
    --limit 5 \
    --out-predictions "$pred" \
    --out-metrics "$metrics" \
    --max-tokens 256 \
    --temp 0.0

  local ok_count
  ok_count=$("$PYTHON" -c "
import json
from pathlib import Path
rows = [json.loads(l) for l in Path('$pred').read_text().splitlines() if l.strip()]
ok = sum(1 for r in rows if r.get('response'))
print(ok, len(rows))
")
  local ok="${ok_count%% *}"
  local total="${ok_count##* }"
  log "  DeepSeek smoke: ${ok}/${total} JSON parse ok"
  if [[ "$ok" -lt 4 ]]; then
    echo "error: DeepSeek R1 smoke failed (${ok}/${total}). Check scripts/mlx_chat_utils.py" >&2
    exit 1
  fi
}

phase_experiments() {
  log "Phase 3–7/7: LoRA train → test500 eval → export → plot (run_paper_experiments.sh)"
  SKIP_DOWNLOAD=1 \
  SKIP_TRAIN="$SKIP_TRAIN" \
  USE_MIRROR="$USE_MIRROR" \
  PAPER_TRAIN_PROFILE="$PAPER_TRAIN_PROFILE" \
  bash "$ROOT/scripts/run_paper_experiments.sh"
}

main() {
  require_python
  mkdir -p "$LOGS"

  local mode="${1:-full}"
  case "$mode" in
    --check-only)
      phase_check_downloadable
      USE_MIRROR="$USE_MIRROR" bash "$ROOT/scripts/download_paper_models.sh" --status
      log "All three presets reachable via mirror when USE_MIRROR=1."
      ;;
    --download-only)
      phase_check_downloadable
      phase_download
      ;;
    full|"")
      if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
        phase_check_downloadable
        phase_download
      else
        log "Phase 0–1/7: SKIP_DOWNLOAD=1"
        USE_MIRROR="$USE_MIRROR" bash "$ROOT/scripts/download_paper_models.sh" --status
      fi
      phase_deepseek_smoke
      phase_experiments
      log "Pipeline complete. Results: $ROOT/local/results/paper/export/benchmark_paper/"
      ;;
    *)
      echo "usage: $0 [--check-only|--download-only|full]" >&2
      exit 1
      ;;
  esac
}

main "$@"
