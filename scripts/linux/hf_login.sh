#!/usr/bin/env bash
# Optional Hugging Face connectivity check (local weights do not need login).
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "HF_ENDPOINT=$HF_ENDPOINT"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$HF_TOKEN" "$PYTHON" -c 'import os; from huggingface_hub import login; login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)'
  echo "HF_TOKEN applied."
else
  echo "No HF_TOKEN set; skipping login (local/offline weights are fine)."
fi

"$PYTHON" - <<'PY'
import os
import sys

endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
print("probing:", endpoint)
try:
    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint)
    if token:
        who = api.whoami(token=token)
        print("whoami:", who.get("name", who))
    else:
        print("whoami: skipped (no token)")
except Exception as exc:
    print("warning: HF probe failed (ok when using local model dirs):", exc, file=sys.stderr)
    sys.exit(0)
print("HF OK")
PY
