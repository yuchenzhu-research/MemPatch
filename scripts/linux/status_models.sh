#!/usr/bin/env bash
# Print pipeline status for paper models.
#
#   bash scripts/linux/status_models.sh
#   SLUG=mistral_nemo_12b bash scripts/linux/status_models.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"

if [[ -n "${SLUG:-}" ]]; then
  print_model_status "$SLUG"
  exit 0
fi

for slug in "${PAPER_SLUGS[@]}"; do
  print_model_status "$slug"
  echo ""
done
