#!/usr/bin/env bash
# Remove Llama artifacts from LOCAL_ROOT (Mac local/ or AutoDL mempatch_local).
set -euo pipefail
source "$(dirname "$0")/env.sh"

targets=(
  "$LOCAL_ROOT/models"
  "$LOCAL_ROOT/hf_cache"
  "$ADAPTER_ROOT"
  "$LOG_ROOT"
  "$RESULTS_ROOT"
)

for base in "${targets[@]}"; do
  [[ -d "$base" ]] || continue
  find "$base" -maxdepth 3 \( -iname '*llama*' -o -iname '*meta-llama*' \) -print -exec rm -rf {} + 2>/dev/null || true
done

echo "Llama paths under $LOCAL_ROOT cleaned."
