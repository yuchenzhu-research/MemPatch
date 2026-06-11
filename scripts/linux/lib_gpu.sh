#!/usr/bin/env bash
# Release GPU memory between pipeline phases (train → eval → baselines).

release_gpu() {
  "$PYTHON" - <<'PY'
import gc

gc.collect()
try:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except Exception:
    pass
PY
}
