#!/usr/bin/env bash
# Hugging Face login + connectivity check (no `hf` CLI required).
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "HF_ENDPOINT=$HF_ENDPOINT"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$HF_TOKEN" "$PYTHON" -c 'import os; from huggingface_hub import login; login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)'
  echo "HF_TOKEN applied."
else
  echo "error: set HF_TOKEN first" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import os
import sys

endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
print("probing:", endpoint)
try:
    from huggingface_hub import HfApi
    who = HfApi(endpoint=endpoint).whoami()
    print("whoami:", who.get("name", who))
except Exception as exc:
    print("error: HF probe failed:", exc, file=sys.stderr)
    sys.exit(1)
print("HF OK")
PY
