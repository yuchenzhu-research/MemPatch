#!/usr/bin/env python3
"""Run general ReTrace-Bench baselines.

For backward compatibility this script remains named as before, but when the
input records use ``schema_version=retrace_bench_general_1`` it evaluates the
new neutral English benchmark schema requested for the general release.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.llm_providers import get_provider
from benchmark.retrace_bench.scorers_general import aggregate_metrics, score_prediction


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "scenarios.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


def latest_only(scenario: dict[str, Any]) -> dict[str, Any]:
    events = scenario["public_input"]["event_trace"]
    latest = events[-1]
    memory_state = {m["memory_id"]: "current" for m in scenario["public_input"]["initial_memory"]}
    for mid in latest.get("related_memory_ids", []):
        memory_state[mid] = "current"
    return {
        "answer": latest["text"],
        "decision": "use_current_memory",
        "memory_state": memory_state,
        "evidence_event_ids": [latest["event_id"]],
        "failure_diagnosis": "stale_memory_reuse",
    }


def retrieve_all(scenario: dict[str, Any]) -> dict[str, Any]:
    events = scenario["public_input"]["event_trace"]
    text = " ".join(e["text"] for e in events)
    state = {m["memory_id"]: "current" for m in scenario["public_input"]["initial_memory"]}
    evidence = [e["event_id"] for e in events if e["trust_level"] in {"verified", "trusted"}]
    return {
        "answer": text,
        "decision": "use_current_memory",
        "memory_state": state,
        "evidence_event_ids": evidence,
        "failure_diagnosis": scenario.get("primary_failure_mode"),
    }


def heuristic_memory_state(scenario: dict[str, Any]) -> dict[str, Any]:
    events = scenario["public_input"]["event_trace"]
    verified = [e for e in events if e["trust_level"] == "verified"]
    chosen = verified[-1] if verified else events[-1]
    state = {m["memory_id"]: ("out_of_scope" if m.get("is_distractor") else "current") for m in scenario["public_input"]["initial_memory"]}
    for mid in chosen.get("related_memory_ids", []):
        state[mid] = "current"
    mode = scenario.get("primary_failure_mode")
    decision = {
        "policy_violation": "refuse_due_to_policy",
        "conflict_collapse": "mark_unresolved",
        "scope_leakage": "escalate",
        "memory_hallucination": "ask_clarification",
    }.get(mode, "use_current_memory")
    return {
        "answer": chosen["text"] if decision == "use_current_memory" else f"{decision}: {chosen['text']}",
        "decision": decision,
        "memory_state": state,
        "evidence_event_ids": [chosen["event_id"]],
        "failure_diagnosis": mode,
    }


def llm_json_answerer(scenario: dict[str, Any], provider: Any) -> dict[str, Any]:
    if provider is None:
        raise ValueError("llm_json_answerer requires --provider and --model")
    prompt = {
        "instruction": "Answer as strict JSON with keys answer, decision, memory_state, evidence_event_ids, failure_diagnosis.",
        "workflow_context": scenario["workflow_context"],
        "public_input": scenario["public_input"],
        "tasks": scenario["tasks"],
    }
    raw = provider.generate(json.dumps(prompt, ensure_ascii=False), temperature=0)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"answer": raw, "decision": None, "memory_state": {}, "evidence_event_ids": [], "failure_diagnosis": None}
    return parsed


BASELINES = {
    "latest_only": latest_only,
    "retrieve_all": retrieve_all,
    "heuristic_memory_state": heuristic_memory_state,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ReTrace-Bench baseline evaluation.")
    parser.add_argument("--data", required=True, help="Path to scenarios.jsonl or a directory containing it")
    parser.add_argument("--baseline", required=True, choices=sorted([*BASELINES, "llm_json_answerer"]))
    parser.add_argument("--out", required=True, help="Path to output predictions JSONL")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args(argv)

    scenarios = read_jsonl(resolve_data_path(args.data))
    provider = None
    if args.baseline == "llm_json_answerer":
        api_key = args.api_key
        if not api_key and args.provider:
            env_name = {
                "deepseek": "DEEPSEEK_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }.get(args.provider.lower())
            api_key = os.getenv(env_name or "")
        provider = get_provider(args.provider or "openai", args.model or "gpt-4o-mini", api_key=api_key)

    predictions: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for scenario in scenarios:
        if args.baseline == "llm_json_answerer":
            response = llm_json_answerer(scenario, provider)
        else:
            response = BASELINES[args.baseline](scenario)
        pred = {
            "scenario_id": scenario["scenario_id"],
            "baseline": args.baseline,
            "response": response,
        }
        pred["metrics"] = score_prediction(scenario, pred)
        predictions.append(pred)
        scored.append(pred)

    out = Path(args.out)
    write_jsonl(out, predictions)
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(aggregate_metrics(scored), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {out}")
    print(f"Wrote metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

