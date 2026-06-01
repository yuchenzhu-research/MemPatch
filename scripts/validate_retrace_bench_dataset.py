#!/usr/bin/env python3
"""Validate general English ReTrace-Bench JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import (
    DIFFICULTIES,
    DOMAINS,
    FAILURE_MODES,
    MEMORY_STATUSES,
    PUBLIC_FORBIDDEN_TERMS,
    TASK_TYPES,
    TRUST_LEVELS,
)


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
    for event in public.get("event_trace", []):
        pieces.append(event.get("text", ""))
        pieces.append(event.get("source", ""))
    for memory in public.get("initial_memory", []):
        pieces.append(memory.get("text", ""))
    for task in scenario.get("tasks", []):
        pieces.append(task.get("prompt", ""))
    return "\n".join(pieces).lower()


def validate_one(scenario: dict[str, Any]) -> list[str]:
    sid = scenario.get("scenario_id", "<missing>")
    errors: list[str] = []
    if scenario.get("domain") not in DOMAINS:
        errors.append(f"{sid}: invalid domain")
    if scenario.get("primary_failure_mode") not in FAILURE_MODES:
        errors.append(f"{sid}: invalid primary_failure_mode")
    if scenario.get("difficulty") not in DIFFICULTIES:
        errors.append(f"{sid}: invalid difficulty")
    secondary = scenario.get("secondary_failure_modes", [])
    if not isinstance(secondary, list) or len(secondary) > 3 or any(m not in FAILURE_MODES for m in secondary):
        errors.append(f"{sid}: invalid secondary_failure_modes")
    public = scenario.get("public_input", {})
    events = public.get("event_trace", [])
    memories = public.get("initial_memory", [])
    tasks = scenario.get("tasks", [])
    gold = scenario.get("hidden_gold", {})
    if len(events) < 4:
        errors.append(f"{sid}: expected at least 4 events")
    if len(tasks) != 4 or {t.get("task_type") for t in tasks} != set(TASK_TYPES):
        errors.append(f"{sid}: must include exactly the four task types")
    event_ids = [e.get("event_id") for e in events]
    memory_ids = [m.get("memory_id") for m in memories]
    introduced = gold.get("rubric", {}).get("introduced_memories", {})
    all_memory_ids = set(memory_ids) | set(introduced.keys())
    if len(event_ids) != len(set(event_ids)):
        errors.append(f"{sid}: duplicate event_id")
    if len(memory_ids) != len(set(memory_ids)):
        errors.append(f"{sid}: duplicate memory_id")
    for event in events:
        if event.get("trust_level") not in TRUST_LEVELS:
            errors.append(f"{sid}: invalid trust_level in {event.get('event_id')}")
        if not event.get("visibility_scope"):
            errors.append(f"{sid}: missing visibility_scope in {event.get('event_id')}")
        for mid in event.get("related_memory_ids", []):
            if mid not in all_memory_ids:
                errors.append(f"{sid}: event {event.get('event_id')} references missing memory {mid}")
    for memory in memories:
        for eid in memory.get("source_event_ids", []):
            if eid not in event_ids:
                errors.append(f"{sid}: memory {memory.get('memory_id')} references missing event {eid}")
    for eid in gold.get("expected_evidence_event_ids", []):
        if eid not in event_ids:
            errors.append(f"{sid}: hidden evidence_event_id {eid} missing from event_trace")
    for mid, status in gold.get("expected_memory_state", {}).items():
        if mid not in all_memory_ids:
            errors.append(f"{sid}: hidden memory state references missing memory {mid}")
        if status not in MEMORY_STATUSES:
            errors.append(f"{sid}: invalid memory status {status} for {mid}")
    text = public_text(scenario)
    for term in PUBLIC_FORBIDDEN_TERMS:
        if term in text:
            errors.append(f"{sid}: public text contains forbidden term '{term}'")
    return errors


def validate_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [r.get("scenario_id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate scenario_id values")
    counters = Counter()
    for row in rows:
        errors.extend(validate_one(row))
        counters["events_ge_7"] += int(len(row.get("public_input", {}).get("event_trace", [])) >= 7)
        counters["memories_ge_3"] += int(len(row.get("public_input", {}).get("initial_memory", [])) >= 3)
        meta = row.get("metadata", {})
        counters["distractors"] += int(meta.get("has_distractor"))
        counters["cross_scope"] += int(meta.get("has_cross_scope_trap"))
        counters["verified_over_trusted"] += int(meta.get("verified_contradicts_trusted_note"))
        counters["false_premise"] += int(meta.get("requires_rejecting_false_premise"))
        counters["non_answer"] += int(meta.get("requires_non_answer_action"))
    n = len(rows) or 1
    thresholds = {
        "events_ge_7": 0.80,
        "memories_ge_3": 0.50,
        "distractors": 0.40,
        "cross_scope": 0.30,
        "verified_over_trusted": 0.25,
        "false_premise": 0.20,
        "non_answer": 0.20,
    }
    for key, threshold in thresholds.items():
        rate = counters[key] / n
        if rate < threshold:
            errors.append(f"dataset rate {key}={rate:.3f} below {threshold:.2f}")
    return {"count": len(rows), "rates": {k: counters[k] / n for k in sorted(thresholds)}, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args(argv)
    rows = read_jsonl(Path(args.data))
    report = validate_dataset(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

