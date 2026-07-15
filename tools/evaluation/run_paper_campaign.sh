#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  tools/evaluation/run_paper_campaign.sh smoke [model]
  tools/evaluation/run_paper_campaign.sh model <qwen3_14b|mistral_nemo_12b|phi4_14b|deepseek_r1_qwen_14b|glm4_9b|all>

Environment:
  DATA=scratch/data/mempatch/synthetic/raw_internal/main_test_synthetic.jsonl
  OUTPUT_ROOT=runs/eval_main
  LIMIT=3                         # useful for smoke
  QWEN3_MODEL_ID=/path/or/hub/id   # optional override
  MISTRAL_MODEL_ID=/path/or/hub/id
  PHI4_MODEL_ID=/path/or/hub/id
  DEEPSEEK_MODEL_ID=/path/or/hub/id
  GLM_MODEL_ID=/path/or/hub/id
USAGE
}

cmd="${1:-}"
case "$cmd" in
  smoke)
    model="${2:-qwen3_14b}"
    LIMIT="${LIMIT:-3}" OUTPUT_ROOT="${OUTPUT_ROOT:-runs/eval_smoke}" \
      bash tools/evaluation/server/run_all.sh "$model"
    python tools/evaluation/server/validate_run.py \
      --data "${DATA:-scratch/data/mempatch/synthetic/raw_internal/main_test_synthetic.jsonl}" \
      --run-dir "${OUTPUT_ROOT:-runs/eval_smoke}/$model" \
      --expected-cases "$LIMIT"
    ;;
  model)
    model="${2:-}"
    case "$model" in
      qwen3_14b|mistral_nemo_12b|phi4_14b|deepseek_r1_qwen_14b|glm4_9b|all)
        bash tools/evaluation/server/run_all.sh "$model"
        ;;
      *)
        usage
        exit 2
        ;;
    esac
    ;;
  *)
    usage
    exit 2
    ;;
esac
