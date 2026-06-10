#!/usr/bin/env bash
# Three paper models (no Llama): mistral -> gemma -> qwen.
#
#   bash scripts/linux/run_paper_three.sh
#   SLUGS=(mistral_nemo_12b) bash scripts/linux/run_paper_three.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ "${SLUGS+x}" != x ]] || [[ ${#SLUGS[@]} -eq 0 ]]; then
  SLUGS=("${PAPER_SLUGS[@]}")
fi

bash "$LINUX_DIR/01_audit.sh"

for slug in "${SLUGS[@]}"; do
  SLUG="$slug" PHASES="${PHASES:-auto}" bash "$LINUX_DIR/run_model.sh"
done

echo "All models done. Results: $RESULTS_ROOT/"
