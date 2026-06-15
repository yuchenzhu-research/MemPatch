"""Shared Hugging Face chat-template and JSON utility functions."""

from __future__ import annotations

import json
import re
from typing import Any

THINKING_CLOSE_SUFFIX = "\n</think>\n"
JSON_BRACE_PREFILL = "{"


def _coerce_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer / apply_chat_template outputs to flat token ids."""
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if hasattr(value, "ids") and not isinstance(value, (list, str, bytes)):
        value = value.ids
    return [int(token) for token in value]


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


def apply_chat_template_no_think(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    json_prefill: str = JSON_BRACE_PREFILL,
) -> tuple[list[int], dict[str, Any]]:
    """Return prompt token ids and generation metadata."""
    meta: dict[str, Any] = {"json_brace_prefill": False, "json_prefill": ""}
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if prompt_needs_thinking_close(prompt):
        prompt = prompt + THINKING_CLOSE_SUFFIX + json_prefill
        meta["json_prefill"] = json_prefill
        meta["json_brace_prefill"] = json_prefill == JSON_BRACE_PREFILL
    else:
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return _coerce_token_ids(tokens), meta
        except TypeError:
            pass

    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(prompt)
        return _coerce_token_ids(encoded), meta

    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=False,
    )
    return _coerce_token_ids(tokens), meta


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
