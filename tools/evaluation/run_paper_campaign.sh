#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  tools/evaluation/run_paper_campaign.sh smoke [model]
  tools/evaluation/run_paper_campaign.sh model <qwen3_14b|phi4_14b|mistral_nemo_12b>
  tools/evaluation/run_paper_campaign.sh guard
  tools/evaluation/run_paper_campaign.sh analyze

Environment:
  DATA=scratch/data/mempatch/v1.4/raw_internal/main_test_synthetic.jsonl
  OUTPUT_ROOT=runs/eval_main
  LIMIT=3                         # useful for smoke
  QWEN3_MODEL_ID=/path/or/hub/id   # optional override
  PHI4_MODEL_ID=/path/or/hub/id
  MISTRAL_MODEL_ID=/path/or/hub/id
USAGE
}

cmd="${1:-}"
case "$cmd" in
  smoke)
    model="${2:-qwen3_14b}"
    LIMIT="${LIMIT:-3}" OUTPUT_ROOT="${OUTPUT_ROOT:-runs/eval_smoke}" \
      bash tools/evaluation/server/run_all.sh "$model"
    python tools/evaluation/server/validate_run.py \
      --data "${DATA:-scratch/data/mempatch/v1.4/raw_internal/main_test_synthetic.jsonl}" \
      --run-dir "${OUTPUT_ROOT:-runs/eval_smoke}/$model" \
      --expected-cases "$LIMIT"
    ;;
  model)
    model="${2:-}"
    case "$model" in
      qwen3_14b|phi4_14b|mistral_nemo_12b)
        bash tools/evaluation/server/run_all.sh "$model"
        ;;
      *)
        usage
        exit 2
        ;;
    esac
    ;;
  guard|analyze)
    bash tools/evaluation/server/run_all.sh "$cmd"
    ;;
  *)
    usage
    exit 2
    ;;
esac
