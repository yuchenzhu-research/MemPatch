"""Optional model runner for producing MemPatch-Bench prediction JSONL files.

This module is intentionally separate from the official evaluator API. It calls
external LLM providers to create prediction files, then the model-agnostic
scorer in :mod:`benchmark.api` can evaluate those files.
"""

from __future__ import annotations

import json
import importlib
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from benchmark.api import load_scenarios
from benchmark.general_taxonomy import (
    DECISIONS,
    FAILURE_MODES,
    MEMORY_STATUSES,
)
from benchmark.public_view import public_scenario_view

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

DEFAULT_MAX_TOKENS = 1024

REQUIRED_RESPONSE_FIELDS = (
    "answer",
    "decision",
    "memory_state",
    "evidence_event_ids",
    "failure_diagnosis",
)

BAR_LENGTH = 32


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


def format_duration(seconds: float) -> str:
    """Format elapsed seconds for live runner progress logs."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = round(seconds)
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _collect_memory_ids(public_view: dict[str, Any]) -> list[str]:
    public_input = public_view.get("public_input", {})
    memory_ids = [
        m["memory_id"]
        for m in public_input.get("initial_memory", [])
        if isinstance(m, dict) and m.get("memory_id")
    ]
    for event in public_input.get("event_trace", []):
        if not isinstance(event, dict):
            continue
        for memory_id in event.get("related_memory_ids", []) or []:
            if memory_id not in memory_ids:
                memory_ids.append(memory_id)
    return memory_ids


def _arrow(run_index: int) -> str:
    return "=" * min(max(run_index, 1), BAR_LENGTH) + ">"


def _should_show_progress(run_index: int, planned_total: int, progress_interval: int) -> bool:
    interval = max(progress_interval, 1)
    return run_index == 1 or run_index == planned_total or run_index % interval == 0


def _progress_wait(
    *,
    run_index: int,
    planned_total: int,
    dataset_index: int,
    dataset_total: int,
    done: threading.Event,
    case_start: float,
    run_start: float,
) -> None:
    pos = 0
    direction = 1
    while not done.wait(0.12):
        cells = ["."] * BAR_LENGTH
        cells[pos] = "="
        cells[min(pos + 1, BAR_LENGTH - 1)] = ">"
        print(
            f"\r[{''.join(cells)}] {run_index}/{planned_total} "
            f"dataset={dataset_index}/{dataset_total} "
            f"case={time.monotonic() - case_start:.1f}s "
            f"total={format_duration(time.monotonic() - run_start)}",
            end="",
            flush=True,
        )
        pos += direction
        if pos >= BAR_LENGTH - 2 or pos <= 0:
            direction *= -1


def _progress_done(
    *,
    run_index: int,
    planned_total: int,
    dataset_index: int,
    dataset_total: int,
    written: int,
    errors: int,
    case_seconds: float,
    total_seconds: float,
) -> None:
    filled = int(BAR_LENGTH * run_index / max(planned_total, 1))
    rest = max(BAR_LENGTH - filled - 1, 0)
    bar = "=" * filled + ">" + "." * rest
    print(
        f"\r[{bar}] {run_index}/{planned_total} dataset={dataset_index}/{dataset_total} "
        f"written={written} errors={errors} case={case_seconds:.1f}s "
        f"total={format_duration(total_seconds)}",
        flush=True,
    )


def build_prompt(public_view: dict[str, Any]) -> str:
    """Build the model prompt from a sanitized, model-visible scenario view."""
    memory_ids = _collect_memory_ids(public_view)
    payload = {
        "instruction": (
            "Answer as strict JSON only. Do not use Markdown fences. "
            "Use only the visible scenario content. Do not use external knowledge. "
            "Use exact enum strings. Do not invent memory IDs or event IDs. "
            "Cite only minimal supporting event IDs."
        ),
        "required_output_schema": {
            "answer": "short final answer/action text",
            "decision": list(DECISIONS),
            "memory_state": {mid: list(MEMORY_STATUSES) for mid in memory_ids},
            "evidence_event_ids": "minimal list of event_id strings from public_input.event_trace",
            "failure_diagnosis": list(FAILURE_MODES),
        },
        **public_view,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_mode: bool = False,
    disable_thinking: bool = False,
) -> str:
    try:
        openai_module: Any = importlib.import_module("openai")
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for OpenAI-compatible providers. "
            'Install with: pip install -e ".[llm]"'
        ) from exc

    OpenAI = openai_module.OpenAI
    client_kwargs: dict[str, Any] = {"api_key": _api_key(api_key_env), "timeout": timeout}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        request_kwargs["response_format"] = {"type": "json_object"}
    if disable_thinking:
        request_kwargs["extra_body"] = {"enable_thinking": False}
    result: Any = client.chat.completions.create(**request_kwargs)
    return result.choices[0].message.content or ""


def call_anthropic(
    *,
    model: str,
    prompt: str,
    api_key_env: str,
    temperature: float = 0.0,
    timeout: float = 120.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_mode: bool = False,
    disable_thinking: bool = False,
) -> str:
    try:
        anthropic_module: Any = importlib.import_module("anthropic")
    except ImportError as exc:
        raise RuntimeError('Missing dependency for Anthropic. Install with: pip install -e ".[llm]"') from exc

    Anthropic = anthropic_module.Anthropic
    client = Anthropic(api_key=_api_key(api_key_env), timeout=timeout)
    result = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system="Return only JSON that matches the requested schema.",
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [
        text
        for block in result.content
        if getattr(block, "type", None) == "text"
        for text in [getattr(block, "text", None)]
        if isinstance(text, str)
    ]
    return "\n".join(parts)


def call_google(
    *,
    model: str,
    prompt: str,
    api_key_env: str,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_mode: bool = False,
    disable_thinking: bool = False,
) -> str:
    try:
        genai: Any = importlib.import_module("google.genai")
        genai_types: Any = importlib.import_module("google.genai.types")
    except ImportError as exc:
        raise RuntimeError('Missing dependency for Gemini. Install with: pip install -e ".[llm]"') from exc

    client = genai.Client(api_key=_api_key(api_key_env))
    result = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_mode: bool = False,
    disable_thinking: bool = False,
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
            max_tokens=max_tokens,
            json_mode=json_mode,
            disable_thinking=False,
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
            max_tokens=max_tokens,
            json_mode=json_mode,
            disable_thinking=disable_thinking,
        )
    if provider in {"google", "gemini"}:
        return call_google(
            model=model,
            prompt=prompt,
            api_key_env=resolved_env,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            disable_thinking=disable_thinking,
        )
    if provider == "anthropic":
        return call_anthropic(
            model=model,
            prompt=prompt,
            api_key_env=resolved_env,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            json_mode=json_mode,
            disable_thinking=disable_thinking,
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
    sleep_seconds: float = 0.0,
    continue_on_error: bool = False,
    progress_interval: int = 1,
    prompt_mode: str = "compact",
    json_mode: bool = False,
    disable_thinking: bool = True,
    print_prompt_stats: bool = False,
) -> int:
    """Run a provider over scenarios and write canonical prediction JSONL rows."""
    if prompt_mode != "compact":
        raise ValueError("Only prompt_mode='compact' is supported.")
    _load_dotenv_if_available()
    scenarios = load_scenarios(data)
    out_path = Path(out_predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = _completed_ids(out_path) if resume else set()
    dataset_total = len(scenarios)
    completed_existing = len(completed)
    remaining = [
        (index, scenario)
        for index, scenario in enumerate(scenarios, start=1)
        if str(scenario["scenario_id"]) not in completed
    ]
    planned_new_cases = min(max_cases, len(remaining)) if max_cases is not None else len(remaining)
    planned = remaining[:planned_new_cases]
    if print_prompt_stats:
        if planned:
            _idx, first = planned[0]
            prompt = build_prompt(public_scenario_view(first))
            print(f"prompt_chars={len(prompt)}", flush=True)
            print(f"scenario_id={first['scenario_id']}", flush=True)
        else:
            print("prompt_chars=0", flush=True)
            print("scenario_id=", flush=True)
        return 0
    mode = "a" if resume else "w"
    written = 0
    errors = 0
    skipped_existing = dataset_total - len(remaining)
    start = time.monotonic()

    print(
        f"MemPatch-Bench model runner | total={dataset_total} | resume={resume} | "
        f"completed={completed_existing} | skipped={skipped_existing} | "
        f"planned={planned_new_cases} | max_new={max_cases if max_cases is not None else 'all'}",
        flush=True,
    )

    with out_path.open(mode, encoding="utf-8") as f:
        for run_index, (dataset_index, scenario) in enumerate(planned, start=1):
            scenario_id = str(scenario["scenario_id"])
            case_start = time.monotonic()
            show_progress = _should_show_progress(run_index, planned_new_cases, progress_interval)
            spinner_done: threading.Event | None = None
            spinner_thread: threading.Thread | None = None
            if show_progress:
                spinner_done = threading.Event()
                spinner_thread = threading.Thread(
                    target=_progress_wait,
                    kwargs={
                        "run_index": run_index,
                        "planned_total": planned_new_cases,
                        "dataset_index": dataset_index,
                        "dataset_total": dataset_total,
                        "done": spinner_done,
                        "case_start": case_start,
                        "run_start": start,
                    },
                    daemon=True,
                )
                spinner_thread.start()
            try:
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
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    disable_thinking=disable_thinking,
                )
                response = canonical_response(extract_json_object(text))
                row = {"scenario_id": scenario_id, "response": response}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                total_elapsed = time.monotonic() - start
                case_elapsed = time.monotonic() - case_start
                if show_progress:
                    if spinner_done is not None:
                        spinner_done.set()
                    if spinner_thread is not None:
                        spinner_thread.join()
                    spinner_done = None
                    spinner_thread = None
                    _progress_done(
                        run_index=run_index,
                        planned_total=planned_new_cases,
                        dataset_index=dataset_index,
                        dataset_total=dataset_total,
                        written=written,
                        errors=errors,
                        case_seconds=case_elapsed,
                        total_seconds=total_elapsed,
                    )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            except Exception as exc:
                errors += 1
                total_elapsed = time.monotonic() - start
                case_elapsed = time.monotonic() - case_start
                if show_progress:
                    if spinner_done is not None:
                        spinner_done.set()
                    if spinner_thread is not None:
                        spinner_thread.join()
                    spinner_done = None
                    spinner_thread = None
                    print(
                        f"\r{_arrow(run_index)} {run_index}/{planned_new_cases} "
                        f"dataset={dataset_index}/{dataset_total} error case={case_elapsed:.1f}s "
                        f"total={format_duration(total_elapsed)} {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                if not continue_on_error:
                    raise
            finally:
                if spinner_done is not None:
                    spinner_done.set()
                if spinner_thread is not None:
                    spinner_thread.join()

    print(
        f"finished | planned={planned_new_cases} | written={written} | errors={errors} | "
        f"skipped={skipped_existing} | total={format_duration(time.monotonic() - start)} | output={out_path}",
        flush=True,
    )
    return written
