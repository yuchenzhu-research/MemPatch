#!/usr/bin/env bash
set -euo pipefail

DATA="${DATA:-scratch/data/mempatch/synthetic/raw_internal/main_test_synthetic.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/eval_main}"
DTYPE="${DTYPE:-bfloat16}"
LIMIT="${LIMIT:-}"

run_model() {
  local key="$1"
  local model_id="$2"
  local args=(
    python tools/evaluation/server/run_core.py
    --data "$DATA"
    --model-key "$key"
    --model-id "$model_id"
    --output-root "$OUTPUT_ROOT"
    --dtype "$DTYPE"
    --seed 42
    --retrieval-k 3
    --response-tokens 512
    --action-tokens 384
    --resume
  )
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  "${args[@]}"
}

case "${1:-}" in
  qwen3_14b)
    run_model qwen3_14b "${QWEN3_MODEL_ID:-OpenPipe/Qwen3-14B-Instruct}"
    ;;
  phi4_14b)
    run_model phi4_14b "${PHI4_MODEL_ID:-microsoft/phi-4}"
    ;;
  mistral_nemo_12b)
    run_model mistral_nemo_12b "${MISTRAL_MODEL_ID:-mistralai/Mistral-Nemo-Instruct-2407}"
    ;;
  deepseek_r1_qwen_14b)
    run_model deepseek_r1_qwen_14b "${DEEPSEEK_MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
    ;;
  glm4_9b)
    run_model glm4_9b "${GLM_MODEL_ID:-THUDM/glm-4-9b-chat}"
    ;;
  all)
    for key in qwen3_14b mistral_nemo_12b phi4_14b deepseek_r1_qwen_14b glm4_9b; do
      bash "$0" "$key"
    done
    ;;
  *)
    echo "Usage: $0 {qwen3_14b|mistral_nemo_12b|phi4_14b|deepseek_r1_qwen_14b|glm4_9b|all}" >&2
    exit 2
    ;;
esac
