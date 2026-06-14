#!/usr/bin/env bash
# Five frozen formal baselines on test500, in fixed model order.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_gpu.sh"
DRY_RUN="${DRY_RUN:-0}"

run() { printf '+ '; printf '%q ' "$@"; printf '\n'; [[ "$DRY_RUN" == "1" ]] || "$@"; }

for slug in "${FORMAL_SLUGS[@]}"; do
  run env BASELINE_SET=main INCLUDE_LORA=0 RESUME=0 SLUG="$slug" \
    bash "$LINUX_DIR/run_baseline_matrix.sh"
  release_gpu
done
