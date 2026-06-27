#!/usr/bin/env python3
"""Validate MemPatch-Bench v1.4 raw or public JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools._root import bootstrap_from

bootstrap_from(__file__)

from mempatch.benchmark.general_taxonomy import (
    BENCH_SCHEMA_VERSION,
    DECISIONS,
    DIFFICULTIES,
    DOMAINS,
    FAILURE_MODES,
    MEMORY_OPERATIONS,
    MEMORY_STATUSES,
    PATTERNS,
    TASK_TYPES,
    TRUST_LEVELS,
    canonical_hidden_gold_fields,
    normalize_difficulty,
)
from mempatch.benchmark.leakage import audit_public_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
    return rows


def public_text(scenario: dict[str, Any]) -> str:
    pieces = [scenario.get("workflow_context", "")]
    public = scenario.get("public_input", {})
    for event in public.get("event_trace") or public.get("events") or []:
        pieces.append(event.get("text") or event.get("content") or "")
        pieces.append(event.get("source", ""))
    for memory in public.get("initial_memory") or public.get("initial_memories") or []:
        pieces.append(memory.get("text") or memory.get("content") or "")
    tasks = scenario.get("tasks", [])
    if isinstance(tasks, dict):
        task_iter = tasks.values()
    elif isinstance(tasks, list):
        task_iter = tasks
    else:
        task_iter = []
    for task in task_iter:
        if isinstance(task, dict):
            pieces.append(task.get("prompt") or task.get("query") or "")
    for tkey in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task"):
        task = scenario.get(tkey) or {}
        pieces.append(task.get("prompt", ""))
    return "\n".join(pieces).lower()


def _is_background_event(event: dict[str, Any]) -> bool:
    eid = str(event.get("event_id", ""))
    text = str(event.get("text", "")).lower()
    return "-bg-" in eid or "routine status synchronization" in text


def _infer_split(scenario: dict[str, Any], data_path: Path | None) -> str | None:
    split = scenario.get("public_split_name") or scenario.get("split")
    if split:
        return split
    if data_path is not None:
        name = data_path.parent.name
        for candidate in ("train", "test"):
            if name.startswith(f"{candidate}_"):
                return candidate
    return scenario.get("metadata", {}).get("split")


def _public_events(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    public = scenario.get("public_input") or {}
    return [event for event in (public.get("event_trace") or public.get("events") or []) if isinstance(event, dict)]


def _public_memories(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    public = scenario.get("public_input") or {}
    return [memory for memory in (public.get("initial_memory") or public.get("initial_memories") or []) if isinstance(memory, dict)]


def validate_one(
    scenario: dict[str, Any],
    *,
    data_path: Path | None = None,
    smoke: bool = False,
    packaging_final: bool = False,
) -> tuple[list[str], list[str]]:
    sid = scenario.get("scenario_id", "<missing>")
    errors: list[str] = []
    warnings: list[str] = []

    split = _infer_split(scenario, data_path)

    if scenario.get("domain") is not None and scenario.get("domain") not in DOMAINS:
        errors.append(f"{sid}: invalid domain '{scenario.get('domain')}'")
    if scenario.get("primary_failure_mode") is not None and scenario.get("primary_failure_mode") not in FAILURE_MODES:
        errors.append(f"{sid}: invalid primary_failure_mode '{scenario.get('primary_failure_mode')}'")

    pattern = scenario.get("pattern") or scenario.get("metadata", {}).get("pattern")
    if pattern is not None and pattern not in PATTERNS:
        errors.append(f"{sid}: invalid pattern '{pattern}'")

    raw_diff = scenario.get("difficulty")
    raw_diff_level = scenario.get("difficulty_level")
    diff = normalize_difficulty(raw_diff or raw_diff_level)
    diff_level = normalize_difficulty(raw_diff_level) if raw_diff_level else diff
    if (raw_diff or raw_diff_level) and diff not in DIFFICULTIES:
        errors.append(f"{sid}: invalid difficulty level '{raw_diff or raw_diff_level}'")
    if raw_diff and raw_diff_level and diff != diff_level:
        errors.append(
            f"{sid}: difficulty '{raw_diff}' does not match difficulty_level '{raw_diff_level}'"
        )

    has_new_tasks = any(
        k in scenario
        for k in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task")
    )
    if has_new_tasks:
        for tkey in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task", "followup_task"):
            if tkey not in scenario:
                errors.append(f"{sid}: missing required new schema task '{tkey}'")
    elif "tasks" in scenario:
        tasks = scenario.get("tasks", {})
        if isinstance(tasks, dict):
            missing = [key for key in TASK_TYPES if key not in tasks]
            if missing:
                errors.append(f"{sid}: tasks dict missing {missing}")
        elif isinstance(tasks, list):
            if len(tasks) != 4:
                errors.append(f"{sid}: tasks list must include four task entries")
        else:
            errors.append(f"{sid}: tasks must be dict or list")

    events = _public_events(scenario)
    memories = _public_memories(scenario)
    gold_raw = scenario.get("hidden_gold", {})
    gold = canonical_hidden_gold_fields(gold_raw)

    if gold_raw and not gold.get("expected_decision"):
        errors.append(f"{sid}: hidden_gold.expected_decision is missing")
    elif gold_raw and gold["expected_decision"] not in DECISIONS:
        errors.append(f"{sid}: hidden_gold.expected_decision '{gold['expected_decision']}' not in DECISIONS")

    expected_operation = gold.get("expected_memory_operation")
    if gold_raw and not expected_operation:
        errors.append(f"{sid}: hidden_gold.expected_memory_operation is missing")
    elif gold_raw and expected_operation not in MEMORY_OPERATIONS:
        errors.append(
            f"{sid}: hidden_gold.expected_memory_operation '{expected_operation}' not in MEMORY_OPERATIONS"
        )

    if gold_raw and not gold.get("expected_followup_answer"):
        errors.append(f"{sid}: hidden_gold.expected_followup_answer is missing")

    expected_state = gold["expected_memory_state"]
    if gold_raw and not expected_state and gold.get("expected_decision") != "refuse_due_to_policy":
        errors.append(f"{sid}: hidden_gold.expected_memory_state is missing or empty")
    bad_statuses = sorted({s for s in expected_state.values() if s not in MEMORY_STATUSES})
    if bad_statuses:
        errors.append(f"{sid}: invalid memory statuses in expected_memory_state: {bad_statuses}")

    expected_diag = gold.get("expected_failure_diagnosis")
    if gold_raw and not expected_diag:
        errors.append(f"{sid}: hidden_gold.expected_failure_diagnosis is missing")
    elif gold_raw and expected_diag not in FAILURE_MODES:
        errors.append(
            f"{sid}: hidden_gold.expected_failure_diagnosis '{expected_diag}' not in FAILURE_MODES"
        )

    if len(events) < 2:
        errors.append(f"{sid}: expected at least 2 events")

    event_ids = [e.get("event_id") for e in events]
    memory_ids = [m.get("memory_id") for m in memories]

    introduced = gold_raw.get("rubric", {}).get("introduced_memories", {}) or gold_raw.get("introduced_memories", {})
    all_memory_ids = set(memory_ids) | set(introduced.keys()) | set(expected_state.keys())

    if len(event_ids) != len(set(event_ids)):
        errors.append(f"{sid}: duplicate event_id values")
    if len(memory_ids) != len(set(memory_ids)):
        errors.append(f"{sid}: duplicate memory_id values")

    for mid in expected_state:
        if mid not in all_memory_ids:
            errors.append(f"{sid}: expected_memory_state references missing memory {mid}")

    for event in events:
        if event.get("trust_level") not in TRUST_LEVELS and event.get("trust_level") is not None:
            errors.append(f"{sid}: invalid trust_level in {event.get('event_id')}")
        for mid in event.get("related_memory_ids", []):
            if mid not in all_memory_ids:
                errors.append(f"{sid}: event {event.get('event_id')} references missing memory {mid}")

    for memory in memories:
        for eid in memory.get("source_event_ids", []):
            if eid not in event_ids and eid != "e-init":
                errors.append(f"{sid}: memory {memory.get('memory_id')} references missing event {eid}")

    gold_ev = set(gold["expected_evidence_event_ids"])
    if gold_raw and not gold_ev and gold.get("expected_decision") != "refuse_due_to_policy":
        errors.append(f"{sid}: expected non-empty minimal gold evidence list")
    for eid in gold_ev:
        if eid not in event_ids:
            errors.append(f"{sid}: hidden evidence_event_id {eid} missing from event_trace")

    gold_counter = set(gold["counterevidence_event_ids"])
    for eid in gold_counter:
        if eid not in event_ids:
            errors.append(f"{sid}: counterevidence_event_id {eid} missing from event_trace")

    if packaging_final:
        schema_version = scenario.get("schema_version") or (scenario.get("metadata") or {}).get("schema_version")
        if schema_version != BENCH_SCHEMA_VERSION:
            errors.append(
                f"{sid}: schema_version {schema_version!r} != {BENCH_SCHEMA_VERSION!r}"
            )

    return errors, warnings


def validate_dataset(
    rows: list[dict[str, Any]],
    *,
    data_path: Path | None = None,
    smoke: bool = False,
    packaging_final: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    ids = [r.get("scenario_id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate scenario_id values")
    counters = {
        "events_ge_3": 0,
        "memories_ge_2": 0,
        "has_hidden_gold": 0,
    }
    for row in rows:
        row_errors, row_warnings = validate_one(
            row,
            data_path=data_path,
            smoke=smoke,
            packaging_final=packaging_final,
        )
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        counters["events_ge_3"] += int(len(_public_events(row)) >= 3)
        counters["memories_ge_2"] += int(len(_public_memories(row)) >= 2)
        counters["has_hidden_gold"] += int(bool(row.get("hidden_gold")))
    n = len(rows) or 1
    if packaging_final:
        for violation in audit_public_rows(rows):
            errors.append(f"{violation['scenario_id']}: public release leakage paths {violation['paths']}")
    return {
        "count": len(rows),
        "rates": {k: counters[k] / n for k in sorted(counters)},
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--smoke", action="store_true", help="Relax dataset-rate gates for small local samples")
    parser.add_argument(
        "--packaging-final",
        action="store_true",
        help="Strict checks for final public release packaging",
    )
    args = parser.parse_args(argv)
    data_path = Path(args.data)
    rows = read_jsonl(data_path)
    report = validate_dataset(
        rows,
        data_path=data_path,
        smoke=args.smoke,
        packaging_final=args.packaging_final,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
