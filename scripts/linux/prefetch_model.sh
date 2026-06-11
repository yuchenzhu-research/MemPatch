#!/usr/bin/env bash
# Download full weights to LOCAL_MODEL_ROOT (plain dir, no hub cache hangs).
#
#   SLUG=gemma3_12b bash scripts/linux/prefetch_model.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
require_paper_slug "$SLUG"
HF_MODEL="$(resolve_hf_model_hub "$SLUG")"
LOCAL_DIR="$(local_model_dir_for_hub "$HF_MODEL")"
VAR="HF_MODEL_$(echo "$SLUG" | tr '[:lower:]' '[:upper:]')"

if model_dir_complete "$LOCAL_DIR"; then
  echo "Already present: $LOCAL_DIR"
  echo "export ${VAR}=${LOCAL_DIR}"
  exit 0
fi

mkdir -p "$LOCAL_DIR"
echo "Downloading $HF_MODEL -> $LOCAL_DIR"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"$PYTHON" - <<PY
import os
import sys
from huggingface_hub import HfApi, snapshot_download

repo_id = "${HF_MODEL}"
local_dir = "${LOCAL_DIR}"
endpoint = (os.environ.get("HF_ENDPOINT") or "").strip() or "https://hf-mirror.com"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
workers = int(os.environ.get("HF_DOWNLOAD_WORKERS", "4"))

print(
    "HF prefetch config:",
    f"repo={repo_id}",
    f"endpoint={endpoint}",
    f"token={'set' if token else 'missing'}",
    f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '')}",
    f"workers={workers}",
    flush=True,
)

try:
    info = HfApi(endpoint=endpoint).model_info(repo_id, token=token, files_metadata=False)
    gated = getattr(info, "gated", None)
    print(f"HF model probe OK: private={info.private} gated={gated}", flush=True)
except Exception as exc:
    print(f"error: cannot access {repo_id}: {exc}", file=sys.stderr)
    if repo_id.startswith("google/gemma"):
        print(
            "hint: place local weights under",
            local_dir,
            "or set HF_MODEL_GEMMA3_12B to your model directory.",
            file=sys.stderr,
        )
    print(
        "hint: prefetch skips download when the local model dir is complete; "
        "set HF_MODEL_<SLUG> to point at an existing checkout.",
        file=sys.stderr,
    )
    sys.exit(1)

if os.environ.get("HF_PREFETCH_PROBE_ONLY") == "1":
    print("HF_PREFETCH_PROBE_ONLY=1; skipping snapshot_download.", flush=True)
    sys.exit(0)

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        endpoint=endpoint,
        token=token,
        ignore_patterns=["consolidated.safetensors"],
        max_workers=workers,
        etag_timeout=30,
    )
except Exception as exc:
    print(f"error: download failed for {repo_id}: {exc}", file=sys.stderr)
    print(
        "hint: if this stays at 'Fetching files: 0%', keep HF_HUB_DISABLE_XET=1, "
        "try HF_DOWNLOAD_WORKERS=1, or switch HF_ENDPOINT between hf-mirror.com and the official endpoint.",
        file=sys.stderr,
    )
    sys.exit(1)

print("OK:", local_dir)
PY

echo "export ${VAR}=${LOCAL_DIR}"
