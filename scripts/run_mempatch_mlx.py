#!/usr/bin/env python3
"""Batch MemPatch predictions via local MLX models (direct baseline or LoRA)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from benchmark.mempatch_bench.model_runner import build_prompt
except ImportError:  # pragma: no cover - legacy alias
    from benchmark.retrace_bench.model_runner import build_prompt  # type: ignore[no-redef]

from benchmark.mempatch_bench.general_taxonomy import DECISIONS, FAILURE_MODES

SYSTEM_PROMPT = "You are MemPatch Revision Policy. Return only strict JSON. Do not explain."

DEFAULT_DECISION = "use_current_memory"
DEFAULT_FAILURE_DIAGNOSIS = "under_update"
ALLOWED_FAILURE_DIAGNOSIS = frozenset(FAILURE_MODES)
ALLOWED_DECISIONS = frozenset(DECISIONS)

THINKING_BLOCK_RE = re.compile(
    r"<think>.*?</think>",
    flags=re.IGNORECASE | re.DOTALL,
)

EVENT_ID_RE = re.compile(r"e-case-[a-zA-Z0-9_-]+")

RETRY_USER_PROMPT = """/no_think
Return exactly one valid JSON object with keys:
decision, memory_state, evidence_event_ids, failure_diagnosis, answer.
evidence_event_ids must be a JSON array of event_id strings.
answer must be one sentence.
Do not output any text outside JSON."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: invalid JSON: {exc}") from exc
    return rows


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        scenario_id = row.get("scenario_id")
        if scenario_id is not None:
            done.add(str(scenario_id))
    return done


def probe_public_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": row["scenario_id"],
        "public_input": row["public_input"],
    }


def build_retry_chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(probe_public_view(row), ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{payload}\n\n{RETRY_USER_PROMPT}"},
    ]


def build_chat_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    user_content = build_prompt(probe_public_view(row)) + "\n\n/no_think"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


IM_END = "<|" + "im_end|>"
IM_START = "<|" + "im_start|>"


def format_qwen_chat_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(f"{IM_START}{role}\n{content}\n{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def format_chat_prompt(
    messages: list[dict[str, str]],
    tokenizer: Any | None = None,
) -> str:
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return format_qwen_chat_prompt(messages)


def strip_thinking(text: str) -> str:
    return THINKING_BLOCK_RE.sub("", text).strip()


def _find_balanced_json_object(raw: str) -> str:
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object start")
    depth = 0
    in_string = False
    escape = False
    quote = ""
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    raise ValueError("unbalanced JSON object")


def extract_first_json_object(text: str) -> dict[str, Any]:
    raw = strip_thinking(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = json.loads(_find_balanced_json_object(raw))
    if not isinstance(value, dict):
        raise ValueError("model output JSON must be an object")
    return value


def _first_string(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, list):
        if not value:
            return default
        return str(value[0])
    return str(value)


def _as_event_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
        elif isinstance(item, list) and item:
            out.append(str(item[0]))
        elif item is not None and not isinstance(item, (dict, list)):
            out.append(str(item))
    return out


def _as_memory_state(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, status in value.items():
        if key is None:
            continue
        if isinstance(status, list):
            status = status[0] if status else ""
        out[str(key)] = str(status) if status is not None else ""
    return out


def valid_event_ids_from_row(row: dict[str, Any]) -> set[str]:
    events = row.get("public_input", {}).get("event_trace", [])
    return {
        str(event["event_id"])
        for event in events
        if isinstance(event, dict) and event.get("event_id") is not None
    }


def _canonicalize_decision(value: Any) -> str:
    raw = _first_string(value, default="")
    if raw in ALLOWED_DECISIONS:
        return raw
    return DEFAULT_DECISION


def _canonicalize_failure_diagnosis(value: Any) -> tuple[str, bool]:
    if isinstance(value, list):
        if not value:
            return DEFAULT_FAILURE_DIAGNOSIS, True
        raw = str(value[0])
    elif value is None:
        return DEFAULT_FAILURE_DIAGNOSIS, True
    else:
        raw = str(value)

    if not raw or raw.lower() == "none":
        return DEFAULT_FAILURE_DIAGNOSIS, True
    if raw in ALLOWED_FAILURE_DIAGNOSIS:
        return raw, False
    return DEFAULT_FAILURE_DIAGNOSIS, True


def _extract_event_id_candidates(value: Any) -> list[str]:
    if isinstance(value, list):
        return _as_event_id_list(value)
    if not isinstance(value, str):
        return []
    seen: set[str] = set()
    candidates: list[str] = []

    def add(event_id: str) -> None:
        if event_id and event_id not in seen:
            seen.add(event_id)
            candidates.append(event_id)

    for match in EVENT_ID_RE.findall(value):
        add(match)
    for token in re.split(r"[\s,\[\]\"']+", value.strip()):
        if token:
            add(token)
    return candidates


def _filter_evidence_event_ids(value: Any, valid_event_ids: set[str]) -> list[str]:
    return [
        event_id
        for event_id in _extract_event_id_candidates(value)
        if event_id in valid_event_ids
    ]


def canonicalize_parsed(parsed: dict[str, Any], *, valid_event_ids: set[str]) -> dict[str, Any]:
    failure_diagnosis, invalid_failure_diagnosis = _canonicalize_failure_diagnosis(
        parsed.get("failure_diagnosis")
    )
    row: dict[str, Any] = {
        "answer": _first_string(parsed.get("answer"), default=""),
        "decision": _canonicalize_decision(parsed.get("decision")),
        "memory_state": _as_memory_state(parsed.get("memory_state")),
        "evidence_event_ids": _filter_evidence_event_ids(
            parsed.get("evidence_event_ids"),
            valid_event_ids,
        ),
        "failure_diagnosis": failure_diagnosis,
    }
    if invalid_failure_diagnosis:
        row["_invalid_failure_diagnosis"] = True
    return row


def fallback_prediction(scenario_id: str, raw_output: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "answer": "",
        "decision": DEFAULT_DECISION,
        "memory_state": {},
        "evidence_event_ids": [],
        "failure_diagnosis": DEFAULT_FAILURE_DIAGNOSIS,
        "_invalid_failure_diagnosis": True,
        "_parse_error": True,
        "raw_output": raw_output,
    }


def prediction_row(
    scenario_id: str,
    parsed: dict[str, Any],
    *,
    valid_event_ids: set[str],
) -> dict[str, Any]:
    response = canonicalize_parsed(parsed, valid_event_ids=valid_event_ids)
    return {"scenario_id": scenario_id, **response}


def parse_model_output(scenario_id: str, text: str, *, valid_event_ids: set[str]) -> dict[str, Any] | None:
    try:
        return prediction_row(
            scenario_id,
            extract_first_json_object(text),
            valid_event_ids=valid_event_ids,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _load_generate_api() -> tuple[Any, Callable[..., str]]:
    from mlx_lm import generate, load

    return load, generate


def generate_with_subprocess(
    *,
    python: str,
    model: str,
    adapter_path: str | None,
    prompt: str,
    max_tokens: int,
) -> str:
    cmd = [
        python,
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        model,
        "--max-tokens",
        str(max_tokens),
        "--prompt",
        prompt,
    ]
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(f"mlx_lm generate failed ({proc.returncode}): {stderr or proc.stdout}")
    return proc.stdout


class MlxGenerator:
    def __init__(
        self,
        *,
        model: str,
        adapter_path: str | None,
        max_tokens: int,
        python: str,
        use_subprocess: bool,
    ) -> None:
        self.model = model
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.python = python
        self.use_subprocess = use_subprocess
        self._mlx_model = None
        self._tokenizer = None
        self._generate_fn = None

    def _ensure_api_model(self) -> None:
        if self._mlx_model is not None:
            return
        load, generate_fn = _load_generate_api()
        load_kwargs: dict[str, Any] = {}
        if self.adapter_path:
            load_kwargs["adapter_path"] = self.adapter_path
        self._mlx_model, self._tokenizer = load(self.model, **load_kwargs)
        self._generate_fn = generate_fn

    def generate(self, messages: list[dict[str, str]]) -> str:
        self._ensure_api_model()
        assert self._tokenizer is not None
        prompt = format_chat_prompt(messages, self._tokenizer)
        if self.use_subprocess:
            return generate_with_subprocess(
                python=self.python,
                model=self.model,
                adapter_path=self.adapter_path,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
        assert self._generate_fn is not None
        assert self._mlx_model is not None
        return self._generate_fn(
            self._mlx_model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local MLX model over MemPatch probe rows and write predictions JSONL.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="JSONL with scenario_id and public_input (e.g. hard_probe.jsonl)",
    )
    parser.add_argument("--model", required=True, help="Local MLX model path or Hugging Face repo")
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter directory for QLoRA inference",
    )
    parser.add_argument(
        "--out-predictions",
        required=True,
        help="Path to write flat predictions JSONL",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of rows to run",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append and skip scenario_ids already present in --out-predictions",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for subprocess fallback (default: current interpreter)",
    )
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="Force mlx_lm generate subprocess instead of the Python API",
    )
    parser.add_argument(
        "--print-prompt-stats",
        action="store_true",
        help="Print first prompt length and exit without loading the model",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data)
    out_path = Path(args.out_predictions)
    if not data_path.is_file():
        print(f"error: data file not found: {data_path}", file=sys.stderr)
        return 1

    rows = read_jsonl(data_path)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]

    if args.print_prompt_stats:
        if rows:
            messages = build_chat_messages(rows[0])
            prompt = format_qwen_chat_prompt(messages)
            print(f"prompt_chars={len(prompt)}", flush=True)
            print(f"scenario_id={rows[0]['scenario_id']}", flush=True)
            print(f"prompt_prefix={prompt[:240]!r}", flush=True)
        else:
            print("prompt_chars=0", flush=True)
            print("scenario_id=", flush=True)
        return 0

    use_subprocess = args.use_subprocess
    if not use_subprocess:
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            use_subprocess = True

    completed = completed_ids(out_path) if args.resume else set()
    planned = [row for row in rows if str(row["scenario_id"]) not in completed]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    label = "LoRA" if args.adapter_path else "direct"
    print(
        f"MemPatch MLX runner ({label}) | total={len(rows)} | resume={args.resume} | "
        f"skipped={len(rows) - len(planned)} | planned={len(planned)} | model={args.model}",
        flush=True,
    )

    generator = MlxGenerator(
        model=args.model,
        adapter_path=args.adapter_path,
        max_tokens=args.max_tokens,
        python=args.python,
        use_subprocess=use_subprocess,
    )

    written = 0
    parse_errors = 0
    start = time.monotonic()
    with out_path.open(mode, encoding="utf-8") as f:
        for index, row in enumerate(planned, start=1):
            scenario_id = str(row["scenario_id"])
            case_start = time.monotonic()
            messages = build_chat_messages(row)
            valid_event_ids = valid_event_ids_from_row(row)
            raw_text = generator.generate(messages)
            prediction = parse_model_output(
                scenario_id,
                raw_text,
                valid_event_ids=valid_event_ids,
            )
            if prediction is None:
                retry_messages = build_retry_chat_messages(row)
                retry_raw = generator.generate(retry_messages)
                prediction = parse_model_output(
                    scenario_id,
                    retry_raw,
                    valid_event_ids=valid_event_ids,
                )
                if prediction is not None:
                    prediction["_retried"] = True
                else:
                    prediction = fallback_prediction(scenario_id, retry_raw)
            if prediction.get("_parse_error"):
                parse_errors += 1
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            elapsed = time.monotonic() - case_start
            print(
                f"[{index}/{len(planned)}] {scenario_id} "
                f"parse_error={bool(prediction.get('_parse_error'))} "
                f"decision={prediction.get('decision')!r} "
                f"case={elapsed:.1f}s",
                flush=True,
            )

    total = time.monotonic() - start
    print(
        f"finished | written={written} | parse_errors={parse_errors} | "
        f"total={total:.1f}s | output={out_path}",
        flush=True,
    )
    return 1 if parse_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
