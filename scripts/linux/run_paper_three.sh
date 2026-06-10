#!/usr/bin/env bash
# Three paper models (no Llama): prefetch all, then run mistral -> gemma -> qwen.
#
#   bash scripts/linux/run_paper_three.sh
#   SLUGS=(mistral_nemo_12b) bash scripts/linux/run_paper_three.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$LINUX_DIR/lib_phases.sh"

PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")"
echo "[$(date '+%F %T')] run_paper_three start LOCAL_ROOT=$LOCAL_ROOT" | tee -a "$PIPELINE_LOG"

if [[ "${SLUGS+x}" != x ]] || [[ ${#SLUGS[@]} -eq 0 ]]; then
  SLUGS=("${PAPER_SLUGS[@]}")
fi

PREFETCH_ALL_FIRST="${PREFETCH_ALL_FIRST:-1}"

if [[ "$PREFETCH_ALL_FIRST" == "1" ]]; then
  echo "[$(date '+%F %T')] prefetch missing models first: ${SLUGS[*]}" | tee -a "$PIPELINE_LOG"
  for slug in "${SLUGS[@]}"; do
    if phase_prefetch_done "$slug"; then
      echo "[$(date '+%F %T')] [$slug] prefetch already complete, skip" | tee -a "$PIPELINE_LOG"
      continue
    fi
    echo "[$(date '+%F %T')] [$slug] prefetch start" | tee -a "$PIPELINE_LOG"
    SLUG="$slug" bash "$LINUX_DIR/prefetch_model.sh" 2>&1 | tee -a "$PIPELINE_LOG"
    echo "[$(date '+%F %T')] [$slug] prefetch done" | tee -a "$PIPELINE_LOG"
  done
fi

bash "$LINUX_DIR/01_audit.sh" 2>&1 | tee -a "$PIPELINE_LOG"

for slug in "${SLUGS[@]}"; do
  SLUG="$slug" PHASES="${PHASES:-auto}" bash "$LINUX_DIR/run_model.sh"
done

echo "All models done. Results: $RESULTS_ROOT/"
