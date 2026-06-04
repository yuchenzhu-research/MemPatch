"""Optional model runner for producing ReTrace-Bench prediction JSONL files.

This module is intentionally separate from the official evaluator API. It calls
external LLM providers to create prediction files, then the model-agnostic
scorer in :mod:`benchmark.retrace_bench.api` can evaluate those files.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from benchmark.retrace_bench.api import load_scenarios
from benchmark.retrace_bench.general_taxonomy import (
    DECISIONS,
    FAILURE_MODE_DEFINITIONS,
    FAILURE_MODES,
    MEMORY_STATUSES,
)
from benchmark.retrace_bench.public_view import public_scenario_view

PROVIDERS = (
    "openai",
    "google",
    "gemini",
    "anthropic",
    "openai_compatible",
    "siliconflow",
    "deepseek",
    "minimax",
)

DEFAULT_BASE_URLS = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "minimax": "https://api.minimax.io/v1",
}

DEFAULT_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}

REQUIRED_RESPONSE_FIELDS = (
    "answer",
    "decision",
    "memory_state",
    "evidence_event_ids",
    "failure_diagnosis",
)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _api_key(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"Missing API key: set {env_name} or pass --api-key-env.")
    return value


def build_prompt(public_view: dict[str, Any]) -> str:
    """Build the model prompt from a sanitized, model-visible scenario view."""
    payload = json.dumps(public_view, ensure_ascii=False, indent=2, sort_keys=True)
    failure_defs = json.dumps(FAILURE_MODE_DEFINITIONS, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        "You are running ReTrace-Bench. Read the scenario and return ONLY valid JSON.\n"
        "Do not include markdown, explanations, or extra keys.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "answer": "string answer to the black-box task",\n'
        f'  "decision": "one of {list(DECISIONS)}",\n'
        f'  "memory_state": {{"memory_id": "one of {list(MEMORY_STATUSES)}"}},\n'
        '  "evidence_event_ids": ["event_id", "..."],\n'
        f'  "failure_diagnosis": "one of {list(FAILURE_MODES)}"\n'
        "}\n\n"
        "Failure diagnosis definitions:\n"
        f"{failure_defs}\n\n"
        "Scenario:\n"
        f"{payload}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse model output as JSON, accepting fenced ```json blocks."""
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output JSON must be an object")
    return value


def canonical_response(parsed: dict[str, Any]) -> dict[str, Any]:
    """Keep only evaluator-facing response fields and fill missing containers."""
    response = {field: parsed.get(field) for field in REQUIRED_RESPONSE_FIELDS}
    if response["memory_state"] is None:
        response["memory_state"] = {}
    if response["evidence_event_ids"] is None:
        response["evidence_event_ids"] = []
    if not isinstance(response["memory_state"], dict):
        response["memory_state"] = {}
    if not isinstance(response["evidence_event_ids"], list):
        response["evidence_event_ids"] = []
    return response


def _chat_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Return only JSON that matches the requested schema.",
        },
        {"role": "user", "content": prompt},
    ]


def call_openai_chat(
    *,
    model: str,
    prompt: str,
    api_key_env: str,
    base_url: str | None = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for OpenAI-compatible providers. "
            'Install with: pip install -e ".[llm]"'
        ) from exc

    client_kwargs: dict[str, Any] = {"api_key": _api_key(api_key_env), "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    result = client.chat.completions.create(
        model=model,
        messages=_chat_messages(prompt),
        temperature=temperature,
    )
    return result.choices[0].message.content or ""


def call_anthropic(
    *,
    model: str,
    prompt: str,
    api_key_env: str,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError('Missing dependency for Anthropic. Install with: pip install -e ".[llm]"') from exc

    client = Anthropic(api_key=_api_key(api_key_env), timeout=timeout)
    result = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system="Return only JSON that matches the requested schema.",
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    return "\n".join(parts)


def call_google(
    *,
    model: str,
    prompt: str,
    api_key_env: str,
    temperature: float = 0.0,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError('Missing dependency for Gemini. Install with: pip install -e ".[llm]"') from exc

    client = genai.Client(api_key=_api_key(api_key_env))
    result = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return result.text or ""


def call_model(
    *,
    provider: str,
    model: str,
    prompt: str,
    api_key_env: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider {provider!r}; expected one of {list(PROVIDERS)}")

    resolved_env = api_key_env or DEFAULT_API_KEY_ENVS.get(provider)
    if provider == "openai_compatible" and not resolved_env:
        raise ValueError("--api-key-env is required for provider openai_compatible")
    if not resolved_env:
        raise ValueError(f"No default API key env configured for provider {provider!r}")

    if provider == "openai":
        return call_openai_chat(
            model=model,
            prompt=prompt,
            api_key_env=resolved_env,
            temperature=temperature,
            timeout=timeout,
        )
    if provider in {"openai_compatible", "siliconflow", "deepseek", "minimax"}:
        resolved_base_url = base_url or DEFAULT_BASE_URLS.get(provider)
        if provider == "openai_compatible" and not resolved_base_url:
            raise ValueError("--base-url is required for provider openai_compatible")
        return call_openai_chat(
            model=model,
            prompt=prompt,
            api_key_env=resolved_env,
            base_url=resolved_base_url,
            temperature=temperature,
            timeout=timeout,
        )
    if provider in {"google", "gemini"}:
        return call_google(
            model=model,
            prompt=prompt,
            api_key_env=resolved_env,
            temperature=temperature,
        )
    if provider == "anthropic":
        return call_anthropic(
            model=model,
            prompt=prompt,
            api_key_env=resolved_env,
            temperature=temperature,
            timeout=timeout,
        )
    raise AssertionError(f"unhandled provider: {provider}")


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario_id = row.get("scenario_id")
            if scenario_id is not None:
                completed.add(str(scenario_id))
    return completed


def run_model_predictions(
    *,
    data: str | Path,
    provider: str,
    model: str,
    out_predictions: str | Path,
    max_cases: int | None = None,
    resume: bool = False,
    api_key_env: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    sleep_seconds: float = 0.0,
) -> int:
    """Run a provider over scenarios and write canonical prediction JSONL rows."""
    _load_dotenv_if_available()
    scenarios = load_scenarios(data)
    out_path = Path(out_predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = _completed_ids(out_path) if resume else set()
    mode = "a" if resume else "w"
    written = 0
    attempted = 0

    with out_path.open(mode, encoding="utf-8") as f:
        for scenario in scenarios:
            scenario_id = str(scenario["scenario_id"])
            if scenario_id in completed:
                continue
            if max_cases is not None and attempted >= max_cases:
                break
            attempted += 1

            view = public_scenario_view(scenario)
            prompt = build_prompt(view)
            text = call_model(
                provider=provider,
                model=model,
                prompt=prompt,
                api_key_env=api_key_env,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout,
            )
            response = canonical_response(extract_json_object(text))
            row = {"scenario_id": scenario_id, "response": response}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    return written
