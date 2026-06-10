#!/usr/bin/env bash
# Hugging Face hub IDs for Linux CUDA paper models (3 backbones; Llama omitted).

resolve_hf_model_hub() {
  local slug="${1:?slug}"
  case "$slug" in
    qwen3_14b) echo "OpenPipe/Qwen3-14B-Instruct" ;;
    gemma3_12b) echo "google/gemma-3-12b-it" ;;
    mistral_nemo_12b) echo "mistralai/Mistral-Nemo-Instruct-2407" ;;
    llama3_1_8b) echo "meta-llama/Meta-Llama-3.1-8B-Instruct" ;;
    *)
      echo "unknown slug: $slug" >&2
      return 1
      ;;
  esac
}

local_model_dir_for_hub() {
  local hub_id="${1:?hub}"
  local safe="${hub_id//\//--}"
  echo "${LOCAL_MODEL_ROOT:-${LOCAL_ROOT:?}/models}/${safe}"
}

resolve_hf_model() {
  local slug="${1:?slug}"
  local upper
  upper="$(echo "$slug" | tr '[:lower:]' '[:upper:]')"
  local var="HF_MODEL_${upper}"
  if [[ -n "${!var:-}" ]]; then
    echo "${!var}"
    return 0
  fi
  local hub_id
  hub_id="$(resolve_hf_model_hub "$slug")"
  local local_dir
  local_dir="$(local_model_dir_for_hub "$hub_id")"
  if [[ -f "$local_dir/config.json" ]]; then
    echo "$local_dir"
    return 0
  fi
  echo "$hub_id"
}

PAPER_SLUGS=(mistral_nemo_12b gemma3_12b qwen3_14b)
