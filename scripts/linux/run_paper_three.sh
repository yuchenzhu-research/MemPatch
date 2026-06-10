#!/usr/bin/env bash
# Three paper models (no Llama): mistral -> gemma -> qwen.
#
#   bash scripts/linux/run_paper_three.sh
#   SLUGS=(mistral_nemo_12b) bash scripts/linux/run_paper_three.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")"
echo "[$(date '+%F %T')] run_paper_three start LOCAL_ROOT=$LOCAL_ROOT" | tee -a "$PIPELINE_LOG"

if [[ "${SLUGS+x}" != x ]] || [[ ${#SLUGS[@]} -eq 0 ]]; then
  SLUGS=("${PAPER_SLUGS[@]}")
fi

bash "$LINUX_DIR/01_audit.sh" 2>&1 | tee -a "$PIPELINE_LOG"

for slug in "${SLUGS[@]}"; do
  SLUG="$slug" PHASES="${PHASES:-auto}" bash "$LINUX_DIR/run_model.sh"
done

echo "All models done. Results: $RESULTS_ROOT/"
