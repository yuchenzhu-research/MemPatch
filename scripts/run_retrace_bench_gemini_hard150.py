#!/usr/bin/env python3
"""Run Gemini on ReTrace-Bench hard150 scenarios."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.scorers_general import aggregate_metrics, score_prediction
from scripts.gemini_api import call_gemini_generate, redact_secrets, require_api_key, resolve_model
from scripts.run_retrace_bench_baseline import (
    _parse_llm_json_response,
    append_jsonl,
    read_jsonl,
    read_resume_jsonl,
)

ALLOWED_DECISIONS = (
    "use_current_memory",
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
)

ALLOWED_MEMORY_STATUSES = (
    "current",
    "outdated",
    "blocked",
    "unresolved",
    "out_of_scope",
    "deleted",
    "should_not_store",
    "restored",
)

ALLOWED_FAILURE_DIAGNOSIS = (
    "stale_memory_reuse",
    "under_update",
    "over_update",
    "conflict_collapse",
    "scope_leakage",
    "policy_violation",
    "wrong_source_attribution",
    "memory_hallucination",
    "unnecessary_memory_write",
    "failure_to_forget",
    "failure_to_release_or_restore",
)

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


def collect_memory_ids(scenario: dict[str, Any]) -> list[str]:
    memory_ids = [m["memory_id"] for m in scenario["public_input"]["initial_memory"]]
    for event in scenario["public_input"]["event_trace"]:
        for mid in event.get("related_memory_ids", []):
            if mid not in memory_ids:
                memory_ids.append(mid)
    return memory_ids


def collect_tasks(scenario: dict[str, Any]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for key in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task"):
        if key in scenario:
            tasks[key] = scenario[key]
    if not tasks and scenario.get("tasks"):
        tasks = dict(scenario["tasks"])
    return tasks


def build_prompt(scenario: dict[str, Any]) -> str:
    memory_ids = collect_memory_ids(scenario)
    payload = {
        "instruction": (
            "Answer as strict JSON only. Do not use Markdown fences. "
            "Use only the visible scenario content below. Do not use external knowledge. "
            "Cite only minimal supporting event IDs from public_input.event_trace. "
            "Do not cite every event. "
            "Respect scope, branch, version, release state, CI status, actor_role, trust_level, "
            "reviewer/maintainer authority, and policy constraints. "
            "If evidence is conflicting or insufficient, choose ask_clarification, mark_unresolved, or escalate. "
            "The memory_state object must include every memory_id from public_input.initial_memory."
        ),
        "required_output_schema": {
            "decision": list(ALLOWED_DECISIONS),
            "answer": "short final answer/action text",
            "memory_state": {mid: list(ALLOWED_MEMORY_STATUSES) for mid in memory_ids},
            "evidence_event_ids": "minimal list of event_id strings from public_input.event_trace",
            "failure_diagnosis": list(ALLOWED_FAILURE_DIAGNOSIS),
        },
        "scenario_id": scenario["scenario_id"],
        "domain": scenario.get("domain"),
        "difficulty": scenario.get("difficulty") or scenario.get("difficulty_level"),
        "workflow_context": scenario.get("workflow_context", ""),
        "public_input": scenario.get("public_input", {}),
        "tasks": collect_tasks(scenario),
    }
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


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["scenario_id"]: row for row in read_resume_jsonl(path)}


def load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["scenario_id"]: row for row in read_jsonl(path)}


def write_progress(
    path: Path,
    *,
    completed: int,
    total: int,
    errors: int,
    status: str,
) -> None:
    path.write_text(
        json.dumps(
            {"completed": completed, "total": total, "errors": errors, "status": status},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_predictions_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_api_check(model: str, api_key: str) -> bool:
    try:
        raw = call_gemini_generate(
            prompt='Return exactly this JSON: {"ok": true}',
            model=model,
            temperature=0.0,
        )
        return json.loads(raw).get("ok") is True
    except Exception as exc:
        print(json.dumps({"ok": False, "error": redact_secrets(str(exc), api_key)}), flush=True)
        return False


def write_summary(
    *,
    out_dir: Path,
    model: str,
    api_check_ok: bool,
    success_count: int,
    error_count: int,
    format_failure_count: int,
    metrics: dict[str, float],
    deepseek_metrics: dict[str, float] | None,
) -> None:
    lines = [
        "# Gemini Hard150 Benchmark Summary",
        "",
        "## A. API check",
        f"- Status: **{'PASS' if api_check_ok else 'FAIL'}**",
        "",
        "## B. Model",
        f"- `{model}`",
        "",
        "## C. Run counts",
        f"- Success: **{success_count}**",
        f"- Errors: **{error_count}**",
        f"- Format failures: **{format_failure_count}**",
        "",
        "## D. Metrics",
        "",
        "| metric | Gemini |",
        "| --- | --- |",
    ]
    for key in REPORT_METRICS:
        val = metrics.get(key)
        lines.append(f"| {key} | {val:.3f} |" if val is not None else f"| {key} | n/a |")

    lines.extend(["", "## E. Gemini vs DeepSeek-V4-Pro", ""])
    if deepseek_metrics:
        ds_joint = deepseek_metrics.get("joint_revision_success")
        gm_joint = metrics.get("joint_revision_success")
        ds_diag = deepseek_metrics.get("failure_diagnosis_accuracy")
        gm_diag = metrics.get("failure_diagnosis_accuracy")
        stronger = "Gemini" if (gm_joint or 0) > (ds_joint or 0) else "DeepSeek"
        if abs((gm_joint or 0) - (ds_joint or 0)) < 0.01:
            stronger = "roughly tied"
        lines.extend(
            [
                f"- joint_revision_success: Gemini **{gm_joint:.3f}** vs DeepSeek **{ds_joint:.3f}**",
                f"- failure_diagnosis_accuracy: Gemini **{gm_diag:.3f}** vs DeepSeek **{ds_diag:.3f}**",
                f"- Overall on joint metric: **{stronger}**",
            ]
        )
    else:
        lines.append("- DeepSeek baseline metrics not found; comparison skipped.")

    lines.extend(["", "## F. Hard150 discriminability", ""])
    latest_weak = metrics.get("joint_revision_success", 1.0) < 0.60
    diag_not_trivial = metrics.get("failure_diagnosis_accuracy", 1.0) < 0.85
    if latest_weak and diag_not_trivial:
        lines.append(
            "- Hard150 still shows discrimination: Gemini joint remains below 0.60 "
            "and failure_diagnosis is not trivially perfect."
        )
    else:
        lines.append(
            "- Hard150 discrimination signal is weaker than expected on Gemini "
            "(joint >= 0.60 and/or diagnosis too high)."
        )

    lines.extend(["", "## G. Scale to hard_500?", ""])
    if api_check_ok and success_count >= 140 and latest_weak and diag_not_trivial:
        lines.append(
            "- **Conditional yes**: generator/oracle pipeline looks stable and Gemini "
            "is not saturating joint metrics; fix decision skew first, then scale."
        )
    else:
        lines.append(
            "- **Not yet**: resolve API/run quality issues or weak discrimination before hard_500."
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/retrace_bench_hard150/hard_150_en/scenarios.jsonl",
    )
    parser.add_argument("--out-dir", default="outputs/retrace_bench_gemini_hard150")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args(argv)

    api_key = require_api_key()
    model = resolve_model()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "gemini_predictions.jsonl"
    pred_json_path = out_dir / "gemini_predictions.json"
    raw_path = out_dir / "gemini_raw_responses.jsonl"
    err_path = out_dir / "gemini_errors.jsonl"
    metrics_path = out_dir / "gemini_metrics.json"
    progress_path = out_dir / "progress.json"

    scenarios = read_jsonl(Path(args.data))
    total = len(scenarios)

    api_check_ok = run_api_check(model, api_key)
    if not api_check_ok:
        write_progress(progress_path, completed=0, total=total, errors=1, status="api_check_failed")
        return 1
    completed = load_completed(pred_path) if args.resume else {}
    raw_rows = load_jsonl_map(raw_path) if args.resume else {}
    err_rows = load_jsonl_map(err_path) if args.resume else {}

    predictions: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    error_count = len(err_rows)
    format_failure_count = 0

    for sid, row in completed.items():
        scenario = next((s for s in scenarios if s["scenario_id"] == sid), None)
        if scenario is None:
            continue
        pred = {
            "scenario_id": sid,
            "model": model,
            "response": row.get("response", {}),
        }
        pred["metrics"] = score_prediction(scenario, pred)
        predictions.append(row)
        scored_rows.append(
            {
                "scenario_id": sid,
                "expected_decision": scenario["hidden_gold"]["expected_decision"],
                "response": pred["response"],
                "metrics": pred["metrics"],
            }
        )
        if float(pred["metrics"].get("format_failure_rate", 0.0)) >= 1.0:
            format_failure_count += 1

    write_progress(
        progress_path,
        completed=len(predictions),
        total=total,
        errors=error_count,
        status="running",
    )

    for idx, scenario in enumerate(scenarios, start=1):
        sid = scenario["scenario_id"]
        if sid in completed:
            continue
        if idx % 10 == 0 or idx == 1 or idx == len(scenarios):
            print(f"[{idx}/{total}] Gemini :: {sid}", flush=True)

        prompt = build_prompt(scenario)
        try:
            raw = call_gemini_generate(prompt=prompt, model=model, temperature=0.0)
            raw_row = {"scenario_id": sid, "model": model, "raw_response": raw}
            raw_rows[sid] = raw_row
            append_jsonl(raw_path, raw_row)
            parsed = normalize_response(_parse_llm_json_response(raw))
            response = {
                "decision": parsed.get("decision"),
                "answer": parsed.get("answer"),
                "memory_state": parsed.get("memory_state", {}),
                "evidence_event_ids": parsed.get("evidence_event_ids", []),
                "failure_diagnosis": parsed.get("failure_diagnosis"),
            }
            pred = {"scenario_id": sid, "model": model, "response": response}
            pred["metrics"] = score_prediction(scenario, pred)
            row = {"scenario_id": sid, "response": response, "metrics": pred["metrics"]}
            append_jsonl(pred_path, row)
            predictions.append(row)
            scored_rows.append(
                {
                    "scenario_id": sid,
                    "expected_decision": scenario["hidden_gold"]["expected_decision"],
                    "response": response,
                    "metrics": pred["metrics"],
                }
            )
            if float(pred["metrics"].get("format_failure_rate", 0.0)) >= 1.0:
                format_failure_count += 1
        except Exception as exc:
            error_count += 1
            err_row = {
                "scenario_id": sid,
                "model": model,
                "error": redact_secrets(str(exc), api_key),
            }
            err_rows[sid] = err_row
            append_jsonl(err_path, err_row)
            time.sleep(min(2 ** min(error_count, 4), 8))

        write_progress(
            progress_path,
            completed=len(predictions),
            total=total,
            errors=error_count,
            status="running",
        )

    write_predictions_json(pred_json_path, predictions)
    aggregate = aggregate_metrics(scored_rows)
    report_metrics = {k: float(aggregate.get("all_metrics", {}).get(k, 0.0)) for k in REPORT_METRICS}
    metrics_payload = {
        "model": model,
        "count": len(scenarios),
        "success_count": len(predictions),
        "error_count": error_count,
        "format_failure_count": format_failure_count,
        "all_metrics": aggregate.get("all_metrics", {}),
        "report_metrics": report_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    deepseek_metrics_path = Path("outputs/retrace_bench_hard150/api_models/deepseek-ai__DeepSeek-V4-Pro.predictions.metrics.json")
    deepseek_metrics = None
    if deepseek_metrics_path.exists():
        ds_payload = json.loads(deepseek_metrics_path.read_text(encoding="utf-8"))
        ds_metrics = ds_payload.get("report_metrics") or ds_payload.get("all_metrics") or {}
        deepseek_metrics = {k: float(ds_metrics[k]) for k in REPORT_METRICS if k in ds_metrics}

    write_summary(
        out_dir=out_dir,
        model=model,
        api_check_ok=api_check_ok,
        success_count=len(predictions),
        error_count=error_count,
        format_failure_count=format_failure_count,
        metrics=report_metrics,
        deepseek_metrics=deepseek_metrics,
    )
    write_progress(
        progress_path,
        completed=len(predictions),
        total=total,
        errors=error_count,
        status="done" if len(predictions) == total else "running",
    )
    print(json.dumps({"success": len(predictions), "errors": error_count, "metrics": report_metrics}, indent=2))
    return 0 if len(predictions) == len(scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
