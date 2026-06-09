#!/usr/bin/env bash
# Hugging Face hub IDs for the four paper models (CUDA / QLoRA).
# Override per-slug with HF_MODEL_<SLUG_UPPER>=... if needed.

resolve_hf_model() {
  local slug="${1:?slug}"
  local upper
  upper="$(echo "$slug" | tr '[:lower:]' '[:upper:]')"
  local var="HF_MODEL_${upper}"
  if [[ -n "${!var:-}" ]]; then
    echo "${!var}"
    return 0
  fi
  case "$slug" in
    qwen3_14b) echo "OpenPipe/Qwen3-14B-Instruct" ;;
    gemma3_12b) echo "google/gemma-3-12b-it" ;;
    mistral_nemo_12b) echo "mistralai/Mistral-Nemo-Instruct-2407" ;;
    llama3_1_8b) echo "meta-llama/Meta-Llama-3.1-8B-Instruct" ;;
    *)
      echo "unknown slug: $slug (set HF_MODEL_${upper})" >&2
      return 1
      ;;
  esac
}

PAPER_SLUGS=(qwen3_14b gemma3_12b mistral_nemo_12b llama3_1_8b)
