#!/usr/bin/env python3
"""Verify mlx-lm can load a local model directory for LoRA training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _resolved_model_type(model_type: str) -> str:
    from mlx_lm.utils import MODEL_REMAPPING

    return MODEL_REMAPPING.get(model_type, model_type)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_mlx_lora_model.py <model_dir>", file=sys.stderr)
        return 2

    model_dir = Path(sys.argv[1])
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        print(f"error: missing {config_path}", file=sys.stderr)
        return 1

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if not model_type:
        print(f"error: config.json has no model_type: {config_path}", file=sys.stderr)
        return 1

    try:
        import importlib

        import mlx_lm
        from mlx_lm.utils import load
    except ImportError as exc:
        print(f"error: mlx-lm not installed: {exc}", file=sys.stderr)
        return 1

    version = getattr(mlx_lm, "__version__", "unknown")
    resolved = _resolved_model_type(model_type)
    module_name = f"mlx_lm.models.{resolved}"
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(
            f"error: mlx-lm {version} cannot train model_type={model_type!r}\n"
            f"  resolved={resolved!r} module={module_name}\n"
            f"  model_dir={model_dir}\n"
            "  Fix: python -m pip install --upgrade "
            "'git+https://github.com/ml-explore/mlx-lm.git@main'\n"
            "  Or use MODELS=gemma3_12b,... instead of gemma4_12b.",
            file=sys.stderr,
        )
        return 1

    try:
        load(str(model_dir))
    except Exception as exc:
        print(
            f"error: mlx-lm {version} failed to load {model_dir}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"ok: mlx-lm {version} supports {model_type} -> {resolved} ({model_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
