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
    DECISIONS,
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
    
    # 1. Domain and difficulty validations
    if scenario.get("domain") not in DOMAINS:
        errors.append(f"{sid}: invalid domain '{scenario.get('domain')}'")
    if scenario.get("primary_failure_mode") not in FAILURE_MODES:
        errors.append(f"{sid}: invalid primary_failure_mode '{scenario.get('primary_failure_mode')}'")
        
    diff = scenario.get("difficulty") or scenario.get("difficulty_level")
    # Map L1-L4 difficulties to general validation if needed
    if diff not in DIFFICULTIES and diff not in ("L1", "L2", "L3", "L4"):
        errors.append(f"{sid}: invalid difficulty level '{diff}'")
        
    # 2. Schema compatibility check for Tasks
    has_new_tasks = any(k in scenario for k in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task"))
    if has_new_tasks:
        for tkey in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task"):
            if tkey not in scenario:
                errors.append(f"{sid}: missing required new schema task '{tkey}'")
    else:
        tasks = scenario.get("tasks", [])
        if len(tasks) != 4 or {t.get("task_type") for t in tasks} != set(TASK_TYPES):
            errors.append(f"{sid}: must include exactly the four task types in tasks list")

    public = scenario.get("public_input", {})
    events = public.get("event_trace", [])
    memories = public.get("initial_memory", [])
    gold = scenario.get("hidden_gold", {})
    
    # 3. Minimum event and memory checks
    if len(events) < 2:
        errors.append(f"{sid}: expected at least 2 events")
        
    event_ids = [e.get("event_id") for e in events]
    memory_ids = [m.get("memory_id") for m in memories]
    
    introduced = gold.get("rubric", {}).get("introduced_memories", {}) or gold.get("introduced_memories", {})
    all_memory_ids = set(memory_ids) | set(introduced.keys())
    
    if len(event_ids) != len(set(event_ids)):
        errors.append(f"{sid}: duplicate event_id values")
    if len(memory_ids) != len(set(memory_ids)):
        errors.append(f"{sid}: duplicate memory_id values")
        
    # 4. Reference checks
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
                
    # 5. Gold evidence and counterevidence checks
    gold_ev = set(gold.get("expected_evidence_event_ids", []) or gold.get("minimal_evidence_event_ids", []) or [])
    if not gold_ev and gold.get("expected_decision") != "refuse_due_to_policy":
        errors.append(f"{sid}: expected non-empty minimal gold evidence list")
    for eid in gold_ev:
        if eid not in event_ids:
            errors.append(f"{sid}: hidden evidence_event_id {eid} missing from event_trace")
            
    gold_counter = set(gold.get("counterevidence_event_ids", []) or [])
    for eid in gold_counter:
        if eid not in event_ids:
            errors.append(f"{sid}: counterevidence_event_id {eid} missing from event_trace")

    # 6. Leakage Audit
    # Ensure public text does not contain forbidden words or decision enum strings
    text = public_text(scenario)
    for term in PUBLIC_FORBIDDEN_TERMS:
        if term in text:
            errors.append(f"{sid}: public text contains forbidden term '{term}'")
            
    # Decision leak check
    for dec in DECISIONS:
        # Avoid checking 'use_current_memory' or single words like 'escalate' in normal natural speech
        # but check for exact action verb combinations
        if dec in text and dec not in ("escalate", "mark_unresolved"):
            errors.append(f"{sid}: public text leaks decision verb phrase '{dec}'")
            
    # Check for no-latest-event shortcut in L3/L4 difficulty
    is_hard_or_l34 = diff in ("L3", "L4", "L3_conditional_validity", "L4_cross_scope_adversarial_audit")
    if is_hard_or_l34 and events:
        sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
        latest_event_id = sorted_events[-1].get("event_id") if sorted_events else None
        # If latest event matches expected evidence alone, it is a shortcut
        if latest_event_id and latest_event_id in gold_ev and len(gold_ev) == 1:
            errors.append(f"{sid}: L3/L4 has latest-event shortcut (latest event is the sole minimal evidence)")

    # 7. Split-specific properties
    manifest_split = scenario.get("public_split_name") or scenario.get("split")
    if manifest_split == "realistic" and scenario.get("annotation_status") != "reviewed":
        # Switched to warning or error as required
        errors.append(f"{sid}: realistic split scenarios must have annotation_status='reviewed'")
        
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

