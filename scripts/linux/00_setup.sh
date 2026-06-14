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
"$PYTHON" - <<'PY'
import torch
import transformers
import accelerate
import peft
import trl
import bitsandbytes
import datasets

if not torch.cuda.is_available():
    raise SystemExit("error: PyTorch installed but CUDA is unavailable")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__} transformers={transformers.__version__} trl={trl.__version__}")
PY

echo "  export LOCAL_ROOT=/root/autodl-tmp/mempatch_local"
echo "  bash scripts/linux/01_audit.sh"
echo "  bash scripts/linux/run_paper_campaign.sh"
