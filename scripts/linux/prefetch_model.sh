#!/usr/bin/env bash
# Download full weights to LOCAL_MODEL_ROOT (plain dir, no hub cache hangs).
#
#   SLUG=gemma3_12b bash scripts/linux/prefetch_model.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
HF_MODEL="$(resolve_hf_model_hub "$SLUG")"
LOCAL_DIR="$(local_model_dir_for_hub "$HF_MODEL")"
VAR="HF_MODEL_$(echo "$SLUG" | tr '[:lower:]' '[:upper:]')"

if [[ -f "$LOCAL_DIR/config.json" ]] && compgen -G "$LOCAL_DIR/model*.safetensors" >/dev/null \
  || compgen -G "$LOCAL_DIR/model-*-of-*.safetensors" >/dev/null; then
  echo "Already present: $LOCAL_DIR"
  echo "export ${VAR}=${LOCAL_DIR}"
  exit 0
fi

mkdir -p "$LOCAL_DIR"
echo "Downloading $HF_MODEL -> $LOCAL_DIR"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"$PYTHON" - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_MODEL}",
    local_dir="${LOCAL_DIR}",
    local_dir_use_symlinks=False,
    endpoint=os.environ.get("HF_ENDPOINT"),
    token=os.environ.get("HF_TOKEN"),
)
print("OK:", "${LOCAL_DIR}")
PY

echo "export ${VAR}=${LOCAL_DIR}"
