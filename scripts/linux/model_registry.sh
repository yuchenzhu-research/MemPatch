#!/usr/bin/env bash
# Hugging Face hub IDs for Linux CUDA paper models (3 backbones; Llama omitted).

PAPER_SLUGS=(mistral_nemo_12b gemma3_12b qwen3_14b phi4_14b)

require_paper_slug() {
  local slug="${1:?slug}"
  local allowed
  for allowed in "${PAPER_SLUGS[@]}"; do
    [[ "$slug" == "$allowed" ]] && return 0
  done
  echo "unsupported Linux paper slug: $slug" >&2
  echo "allowed slugs: ${PAPER_SLUGS[*]}" >&2
  return 1
}

resolve_hf_model_hub() {
  local slug="${1:?slug}"
  require_paper_slug "$slug" || return 1
  case "$slug" in
    qwen3_14b) echo "OpenPipe/Qwen3-14B-Instruct" ;;
    gemma3_12b) echo "google/gemma-3-12b-it" ;;
    mistral_nemo_12b) echo "mistralai/Mistral-Nemo-Instruct-2407" ;;
    phi4_14b) echo "microsoft/phi-4" ;;
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

model_dir_complete() {
  local model_dir="${1:?model dir}"
  [[ -f "$model_dir/config.json" ]] || return 1

  "$PYTHON" - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"
single_path = model_dir / "model.safetensors"

if single_path.is_file():
    raise SystemExit(0)
if not index_path.is_file():
    raise SystemExit(1)

try:
    payload = json.loads(index_path.read_text())
    shards = set(payload["weight_map"].values())
except (OSError, KeyError, TypeError, ValueError):
    raise SystemExit(1)

raise SystemExit(0 if shards and all((model_dir / shard).is_file() for shard in shards) else 1)
PY
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
  hub_id="$(resolve_hf_model_hub "$slug")" || return 1
  local local_dir
  local_dir="$(local_model_dir_for_hub "$hub_id")"
  if model_dir_complete "$local_dir"; then
    echo "$local_dir"
    return 0
  fi
  echo "$hub_id"
}

# Training max_seq_length defaults to the paper protocol value for every backbone.
# Override globally with TRAIN_MAX_SEQ_LENGTH or per slug with TRAIN_MAX_SEQ_LENGTH_<SLUG>.
train_max_seq_length_for_slug() {
  local slug="${1:?slug}"
  if [[ -n "${TRAIN_MAX_SEQ_LENGTH:-}" ]]; then
    echo "$TRAIN_MAX_SEQ_LENGTH"
    return 0
  fi
  local upper var
  upper="$(echo "$slug" | tr '[:lower:]' '[:upper:]')"
  var="TRAIN_MAX_SEQ_LENGTH_${upper}"
  if [[ -n "${!var:-}" ]]; then
    echo "${!var}"
    return 0
  fi
  echo 2048
}

# In-train eval row cap (0 = full L3 val partition). Gemma 12B OOMs on 1400-row eval.
train_eval_max_samples_for_slug() {
  local slug="${1:?slug}"
  if [[ -n "${TRAIN_EVAL_MAX_SAMPLES:-}" ]]; then
    echo "$TRAIN_EVAL_MAX_SAMPLES"
    return 0
  fi
  local upper var
  upper="$(echo "$slug" | tr '[:lower:]' '[:upper:]')"
  var="TRAIN_EVAL_MAX_SAMPLES_${upper}"
  if [[ -n "${!var:-}" ]]; then
    echo "${!var}"
    return 0
  fi
  case "$slug" in
    gemma3_12b) echo "${TRAIN_EVAL_MAX_SAMPLES_GEMMA:-512}" ;;
    qwen3_14b) echo "${TRAIN_EVAL_MAX_SAMPLES_QWEN:-512}" ;;
    *) echo 0 ;;
  esac
}

train_eval_accumulation_steps_for_slug() {
  local slug="${1:?slug}"
  if [[ -n "${TRAIN_EVAL_ACCUMULATION_STEPS:-}" ]]; then
    echo "$TRAIN_EVAL_ACCUMULATION_STEPS"
    return 0
  fi
  case "$slug" in
    gemma3_12b) echo "${TRAIN_EVAL_ACCUMULATION_STEPS_GEMMA:-32}" ;;
    qwen3_14b) echo "${TRAIN_EVAL_ACCUMULATION_STEPS_QWEN:-16}" ;;
    *) echo 8 ;;
  esac
}
