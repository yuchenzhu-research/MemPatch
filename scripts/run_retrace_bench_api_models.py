#!/usr/bin/env python3
"""Run OpenAI-compatible API models on ReTrace-Bench scenarios."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from benchmark.retrace_bench.general_taxonomy import (
    DECISIONS,
    FAILURE_MODE_DEFINITIONS,
    FAILURE_MODES,
    MEMORY_STATUSES,
)
from benchmark.retrace_bench.public_view import public_scenario_view
from benchmark.retrace_bench.scorers_general import aggregate_metrics, score_prediction
from scripts.run_retrace_bench_baseline import (
    _parse_llm_json_response,
    append_jsonl,
    read_jsonl,
    read_resume_jsonl,
    write_jsonl,
)


MAX_ATTEMPTS = 4
REPORT_METRICS = (
    "decision_macro_f1",
    "black_box_decision_accuracy",
    "non_answer_decision_accuracy",
    "memory_state_accuracy",
    "evidence_f1",
    "minimal_evidence_exact_match",
    "evidence_precision",
    "overcitation_rate",
    "counterevidence_recall",
    "failure_diagnosis_accuracy",
    "stale_reuse_rate",
    "latest_event_shortcut_failure_rate",
    "answer_state_consistency",
    "joint_revision_success",
    "format_failure_rate",
)


def resolve_api_config(api_key_env: str, base_url_env: str) -> tuple[str, str]:
    api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set {api_key_env}, OPENAI_API_KEY, or SILICONFLOW_API_KEY."
        )
    base_url = os.getenv(base_url_env) or os.getenv("OPENAI_BASE_URL")
    if not base_url and os.getenv("SILICONFLOW_API_KEY") and api_key == os.getenv("SILICONFLOW_API_KEY"):
        base_url = "https://api.siliconflow.cn/v1"
    if not base_url:
        raise SystemExit(
            f"Missing API base URL. Set {base_url_env} or OPENAI_BASE_URL "
            "(e.g. https://api.siliconflow.cn/v1)."
        )
    return api_key, base_url.rstrip("/")


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model.strip()).strip("_")


def collect_memory_ids(scenario: dict[str, Any]) -> list[str]:
    memory_ids = [m["memory_id"] for m in scenario["public_input"]["initial_memory"]]
    for event in scenario["public_input"]["event_trace"]:
        for mid in event.get("related_memory_ids", []):
            if mid not in memory_ids:
                memory_ids.append(mid)
    return memory_ids


def build_prompt(scenario: dict[str, Any]) -> str:
    memory_ids = collect_memory_ids(scenario)
    visible = public_scenario_view(scenario)
    payload = {
        "instruction": (
            "Answer as strict JSON only. Do not use Markdown fences. "
            "Use exact enum strings for decision, each memory_state value, and failure_diagnosis. "
            "failure_diagnosis must be exactly one enum string, not a list. "
            "Do not invent memory IDs or event IDs. "
            "Do not cite every event. Cite only the minimal event IDs needed to justify the answer."
        ),
        "required_output_schema": {
            "decision": list(DECISIONS),
            "answer": "short final answer/action text",
            "memory_state": {mid: list(MEMORY_STATUSES) for mid in memory_ids},
            "evidence_event_ids": "minimal list of event_id strings from public_input.event_trace",
            "failure_diagnosis": list(FAILURE_MODES),
        },
        "failure_mode_definitions": dict(FAILURE_MODE_DEFINITIONS),
        **visible,
    }
    if "tasks" not in payload:
        tasks: dict[str, Any] = {}
        for key in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task"):
            if key in scenario:
                tasks[key] = scenario[key]
        if not tasks and scenario.get("tasks"):
            payload["tasks"] = scenario["tasks"]
        elif tasks:
            payload.update(tasks)
    return json.dumps(payload, ensure_ascii=False)


def normalize_response(parsed: dict[str, Any]) -> dict[str, Any]:
    diag = parsed.get("failure_diagnosis")
    if isinstance(diag, list):
        parsed["failure_diagnosis"] = diag[0] if diag else None
    evidence = parsed.get("evidence_event_ids")
    if isinstance(evidence, str):
        parsed["evidence_event_ids"] = [evidence]
    elif evidence is None:
        parsed["evidence_event_ids"] = []
    memory_state = parsed.get("memory_state")
    if not isinstance(memory_state, dict):
        parsed["memory_state"] = {}
    return parsed


def call_model(
    client: Any,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
) -> str:
    last_err: Exception | None = None
    extra_kwargs: dict[str, Any] = {}
    if disable_thinking:
        extra_kwargs["extra_body"] = {"enable_thinking": False}
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=180.0,
                **extra_kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(min(2 ** attempt, 8))
    return json.dumps({"error": f"request failed after {MAX_ATTEMPTS} attempts: {type(last_err).__name__}"})


def run_model(
    scenarios: list[dict[str, Any]],
    *,
    model: str,
    client: Any,
    out_path: Path,
    raw_dir: Path,
    temperature: float,
    max_tokens: int,
    resume: bool,
    disable_thinking: bool,
) -> list[dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    completed: set[str] = set()
    if resume and out_path.exists():
        existing = {row["scenario_id"]: row for row in read_resume_jsonl(out_path)}
        predictions = [existing[s["scenario_id"]] for s in scenarios if s["scenario_id"] in existing]
        completed = set(existing)
        write_jsonl(out_path, predictions)
        print(f"[resume] Loaded {len(predictions)} predictions for {model}", flush=True)

    for idx, scenario in enumerate(scenarios, start=1):
        sid = scenario["scenario_id"]
        if sid in completed:
            continue
        print(f"[{idx}/{len(scenarios)}] {model} :: {sid}", flush=True)
        prompt = build_prompt(scenario)
        raw = call_model(
            client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
        (raw_dir / f"{sid}.json").write_text(
            json.dumps({"scenario_id": sid, "model": model, "raw_response": raw}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            parsed = normalize_response(_parse_llm_json_response(raw))
        except json.JSONDecodeError:
            parsed = {
                "answer": raw,
                "decision": None,
                "memory_state": {},
                "evidence_event_ids": [],
                "failure_diagnosis": None,
            }
        pred = {
            "scenario_id": sid,
            "model": model,
            "response": parsed,
            "domain": scenario.get("domain"),
            "primary_failure_mode": scenario.get("primary_failure_mode"),
        }
        pred["metrics"] = score_prediction(scenario, pred)
        predictions.append(pred)
        append_jsonl(out_path, pred)

    return predictions


def extract_report_metrics(aggregate: dict[str, Any]) -> dict[str, float]:
    metrics = aggregate.get("all_metrics") or aggregate.get("metrics") or {}
    return {key: float(metrics[key]) for key in REPORT_METRICS if key in metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible models on ReTrace-Bench.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--api-key-env", default="BENCH_API_KEY")
    parser.add_argument("--base-url-env", default="BENCH_API_BASE_URL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request non-thinking mode for SiliconFlow reasoning models (default: true)",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)

    api_key, base_url = resolve_api_config(args.api_key_env, args.base_url_env)
    try:
        import openai
    except ImportError as exc:
        raise SystemExit("openai package is required for API model runs") from exc

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    scenarios = read_jsonl(Path(args.data))
    if args.max_cases is not None:
        scenarios = scenarios[: args.max_cases]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    for model in models:
        slug = model_slug(model)
        out_path = out_dir / f"{slug}.predictions.jsonl"
        raw_dir = out_dir / f"{slug}.raw"
        preds = run_model(
            scenarios,
            model=model,
            client=client,
            out_path=out_path,
            raw_dir=raw_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resume=args.resume,
            disable_thinking=args.disable_thinking,
        )
        by_id = {s["scenario_id"]: s for s in scenarios}
        aggregate = aggregate_metrics(
            [
                {
                    "scenario_id": p["scenario_id"],
                    "expected_decision": by_id[p["scenario_id"]]["hidden_gold"]["expected_decision"],
                    "response": p["response"],
                    "metrics": p["metrics"],
                }
                for p in preds
            ]
        )
        aggregate["model"] = model
        aggregate["report_metrics"] = extract_report_metrics(aggregate)
        metrics_path = out_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {out_path} and {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
