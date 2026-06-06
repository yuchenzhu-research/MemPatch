"""Load Hugging Face token from environment or repo-root .env."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"


def load_hf_token(*, required: bool = False) -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()

    if _ENV_PATH.is_file():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
                value = value.strip().strip("'\"")
                if value:
                    return value

    if required:
        raise RuntimeError(
            "HF token not found. Set HF_TOKEN in .env or export HF_TOKEN / "
            "HUGGINGFACE_HUB_TOKEN."
        )
    return None
