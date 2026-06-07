"""Shared MLX chat-template helpers (thinking off, JSON-friendly generation)."""

from __future__ import annotations

import json
import re
from typing import Any

THINKING_CLOSE_SUFFIX = "\n</think>\n"
JSON_BRACE_PREFILL = "{"


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("Okay,") or text.startswith("Let me "):
        brace = text.rfind("{")
        if brace != -1:
            text = text[brace:]
    return text


def prompt_needs_thinking_close(prompt: str) -> bool:
    if "<think>" not in prompt:
        return False
    tail = prompt.rsplit("<think>", 1)[-1]
    return "</think>" not in tail


# DeepSeek-R1 opens <think> in the chat template; close it and
# prefill "{" so generation starts in JSON mode instead of free-form reasoning.
def apply_chat_template_no_think(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> tuple[list[int], dict[str, Any]]:
    """Return prompt token ids and generation metadata."""
    meta: dict[str, Any] = {"json_brace_prefill": False}
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if prompt_needs_thinking_close(prompt):
        prompt = prompt + THINKING_CLOSE_SUFFIX + JSON_BRACE_PREFILL
        meta["json_brace_prefill"] = True
    else:
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return tokens, meta
        except TypeError:
            pass

    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(prompt)
        tokens = encoded if isinstance(encoded, list) else list(encoded)
        return tokens, meta

    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=False,
    )
    return tokens, meta


def normalize_generation_text(text: str, *, json_brace_prefill: bool = False) -> str:
    text = strip_thinking(text)
    if json_brace_prefill and not text.lstrip().startswith("{"):
        text = JSON_BRACE_PREFILL + text
    return text


def extract_json_object(text: str, *, json_brace_prefill: bool = False) -> dict[str, Any]:
    text = normalize_generation_text(text, json_brace_prefill=json_brace_prefill)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"no JSON object found in model output: {text[:300]!r}")
