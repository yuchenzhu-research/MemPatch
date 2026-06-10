"""Hugging Face hub helpers (mirror + token + local model dirs) for Linux CUDA scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def hf_endpoint() -> str | None:
    return os.environ.get("HF_ENDPOINT") or None


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def is_local_model_path(model_id: str) -> bool:
    path = Path(model_id)
    return path.is_dir() and (path / "config.json").is_file()


def hub_kwargs(model_id: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    token = hf_token()
    if token:
        kwargs["token"] = token
    if model_id and is_local_model_path(model_id):
        kwargs["local_files_only"] = True
    elif os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        kwargs["local_files_only"] = True
    return kwargs


def log_hub_config(model_id: str | None = None) -> None:
    endpoint = hf_endpoint() or "https://huggingface.co (default)"
    local = model_id and is_local_model_path(model_id)
    print(
        f"HF hub: model={model_id} local_dir={bool(local)} endpoint={endpoint} "
        f"token={'set' if hf_token() else 'missing'}",
        file=sys.stderr,
    )
