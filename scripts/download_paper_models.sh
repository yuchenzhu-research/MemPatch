#!/usr/bin/env bash
# Download all three MemPatch paper MLX models (mirror-friendly).
#
# Models:
#   qwen35-27b   — mlx-community/Qwen3.5-27B-4bit       (~16 GiB)
#   gemma4-12b   — mlx-community/gemma-4-12B-it-4bit    (~8 GiB)
#   deepseek-r1  — mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit (~7.8 GiB)
#
# Usage:
#   bash scripts/download_paper_models.sh              # download missing / incomplete
#   bash scripts/download_paper_models.sh --check      # connectivity only
#   bash scripts/download_paper_models.sh --background # start incomplete in background
#   USE_MIRROR=0 bash scripts/download_paper_models.sh # huggingface.co direct
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LOGS="${LOGS:-$ROOT/local/logs/paper}"
USE_MIRROR="${USE_MIRROR:-1}"

PRESETS="qwen35-27b gemma4-12b deepseek-r1-14b"

log() { printf '[download-paper] %s\n' "$*"; }

preset_local_name() {
  case "$1" in
    qwen35-27b) echo "Qwen3.5-27B-4bit" ;;
    gemma4-12b) echo "gemma-4-12B-it-4bit" ;;
    deepseek-r1-14b) echo "DeepSeek-R1-Distill-Qwen-14B-4bit" ;;
    *) echo "$1" ;;
  esac
}

require_python() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "missing venv python: $PYTHON" >&2
    exit 1
  fi
}

mirror_args() {
  MIRROR_ARGS=()
  if [[ "$USE_MIRROR" == "1" ]]; then
    MIRROR_ARGS=(--mirror --disable-xet)
  fi
}

download_args() {
  mirror_args
  DOWNLOAD_ARGS=(
    --max-workers 1
    --retries 10
    --timeout 300
    "${MIRROR_ARGS[@]}"
  )
}

preset_status() {
  local preset="$1"
  local local_name
  local_name="$(preset_local_name "$preset")"
  local model_dir="$ROOT/local/models/$local_name"
  if [[ -d "$model_dir" ]] && "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
      --preset "$preset" --verify-local >/dev/null 2>&1; then
    echo "complete"
  elif [[ -d "$model_dir" ]]; then
    echo "partial"
  else
    echo "missing"
  fi
}

check_preset() {
  local preset="$1"
  mirror_args
  log "check preset=$preset"
  "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
    --preset "$preset" \
    --check \
    "${MIRROR_ARGS[@]}"
}

download_preset() {
  local preset="$1"
  local local_name
  local_name="$(preset_local_name "$preset")"
  download_args

  if [[ "$(preset_status "$preset")" == "complete" ]]; then
    log "already complete: $local_name"
    return 0
  fi

  log "downloading preset=$preset -> $local_name (mirror=$USE_MIRROR)"
  "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
    --preset "$preset" \
    "${DOWNLOAD_ARGS[@]}"
  "$PYTHON" "$ROOT/scripts/download_mlx_model.py" \
    --preset "$preset" \
    --verify-local
}

download_preset_background() {
  local preset="$1"
  local local_name
  local_name="$(preset_local_name "$preset")"
  local log_file="$LOGS/download_${preset//-/_}.log"
  download_args

  if [[ "$(preset_status "$preset")" == "complete" ]]; then
    log "skip background (complete): $local_name"
    return 0
  fi

  if pgrep -f "download_mlx_model.py --preset ${preset}" >/dev/null 2>&1; then
    log "skip background (already running): $preset"
    return 0
  fi

  mkdir -p "$LOGS"
  log "background download preset=$preset -> $log_file"
  (
    cd "$ROOT"
    nohup "$PYTHON" scripts/download_mlx_model.py \
      --preset "$preset" \
      "${DOWNLOAD_ARGS[@]}" \
      >>"$log_file" 2>&1 &
    echo $!
  ) | while read -r pid; do
    log "  pid=$pid log=$log_file"
  done
}

print_status_table() {
  printf '\n%-22s %-12s %s\n' "PRESET" "STATUS" "LOCAL_DIR"
  printf '%.0s-' {1..70}; echo
  local preset local_name status
  for preset in $PRESETS; do
    local_name="$(preset_local_name "$preset")"
    status="$(preset_status "$preset")"
    printf '%-22s %-12s %s\n' "$preset" "$status" "$ROOT/local/models/$local_name"
  done
  echo
}

main() {
  require_python
  mkdir -p "$LOGS"

  local mode="${1:-download}"
  case "$mode" in
    --check)
      for preset in $PRESETS; do
        check_preset "$preset"
      done
      print_status_table
      ;;
    --background)
      for preset in $PRESETS; do
        download_preset_background "$preset"
      done
      print_status_table
      ;;
    --status)
      print_status_table
      ;;
    download|"")
      for preset in $PRESETS; do
        download_preset "$preset"
      done
      print_status_table
      ;;
    *)
      echo "usage: $0 [--check|--background|--status]" >&2
      exit 1
      ;;
  esac
}

main "$@"
