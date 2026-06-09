#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv"
fi

"$PYTHON" -m pip install -U pip wheel
"$PYTHON" -m pip install -e "$ROOT[dev]"

# CUDA stack — install after PyTorch image is active on the node.
"$PYTHON" -m pip install \
  "torch>=2.3" \
  "transformers>=4.51" \
  "accelerate>=1.0" \
  "peft>=0.14" \
  "trl>=0.15" \
  "bitsandbytes>=0.45" \
  "datasets>=3.0"

echo "Setup OK. Next:"
echo "  export HF_TOKEN=hf_... && hf auth login --token \"\$HF_TOKEN\""
echo "  bash scripts/linux/01_audit.sh"
