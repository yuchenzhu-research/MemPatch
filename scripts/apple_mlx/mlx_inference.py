"""Shared deterministic MLX generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.mlx_support.mlx_chat_utils import apply_chat_template_no_think


def load_mlx_model(model_path: Path | str) -> tuple[Any, Any]:
    from mlx_lm import load

    tokenizer_config: dict[str, Any] = {"trust_remote_code": True}
    text = str(model_path).lower()
    if "mistral" in text or "nemo" in text:
        tokenizer_config["fix_mistral_regex"] = True
    return load(str(model_path), tokenizer_config=tokenizer_config)


def generate_from_messages(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temp: float = 0.0,
    json_prefill: str = "{",
) -> tuple[str, dict[str, Any]]:
    from mlx_lm import generate
    from mlx_lm.generate import make_sampler

    prompt_messages = [m for m in messages if m.get("role") in {"system", "user"}]
    tokens, gen_meta = apply_chat_template_no_think(
        tokenizer,
        prompt_messages,
        json_prefill=json_prefill,
    )
    text = generate(
        model,
        tokenizer,
        tokens,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temp),
        verbose=False,
    )
    return text, gen_meta
