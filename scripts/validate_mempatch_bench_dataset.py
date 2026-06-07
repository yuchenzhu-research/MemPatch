#!/usr/bin/env python3
"""Validate MemPatch-Bench scenario JSONL files (canonical v1.1 schema)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.general_taxonomy import (
    DECISIONS,
    DIFFICULTIES,
    DOMAINS,
    FAILURE_MODES,
    MEMORY_STATUSES,
    PUBLIC_FORBIDDEN_TERMS,
    TASK_TYPES,
    TRUST_LEVELS,
    canonical_hidden_gold_fields,
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
        for candidate in ("train", "validation", "test", "main", "hard", "realistic", "calibration", "private_hidden"):
            if name.startswith(f"{candidate}_"):
                return candidate
    return scenario.get("metadata", {}).get("split")


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

    if scenario.get("domain") not in DOMAINS:
        errors.append(f"{sid}: invalid domain '{scenario.get('domain')}'")
    if scenario.get("primary_failure_mode") not in FAILURE_MODES:
        errors.append(f"{sid}: invalid primary_failure_mode '{scenario.get('primary_failure_mode')}'")

    diff = scenario.get("difficulty") or scenario.get("difficulty_level")
    if diff not in DIFFICULTIES and diff not in ("L1", "L2", "L3", "L4"):
        errors.append(f"{sid}: invalid difficulty level '{diff}'")

    has_new_tasks = any(
        k in scenario
        for k in ("black_box_task", "memory_state_task", "evidence_retrieval_task", "diagnostic_task")
    )
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
    gold_raw = scenario.get("hidden_gold", {})
    gold = canonical_hidden_gold_fields(gold_raw)

    if not gold.get("expected_decision"):
        errors.append(f"{sid}: hidden_gold.expected_decision is missing")
    elif gold["expected_decision"] not in DECISIONS:
        errors.append(f"{sid}: hidden_gold.expected_decision '{gold['expected_decision']}' not in DECISIONS")

    expected_state = gold["expected_memory_state"]
    if not expected_state and gold.get("expected_decision") != "refuse_due_to_policy":
        errors.append(f"{sid}: hidden_gold.expected_memory_state is missing or empty")
    bad_statuses = sorted({s for s in expected_state.values() if s not in MEMORY_STATUSES})
    if bad_statuses:
        errors.append(f"{sid}: invalid memory statuses in expected_memory_state: {bad_statuses}")

    expected_diag = gold.get("expected_failure_diagnosis")
    if not expected_diag:
        errors.append(f"{sid}: hidden_gold.expected_failure_diagnosis is missing")
    elif expected_diag not in FAILURE_MODES:
        errors.append(f"{sid}: hidden_gold.expected_failure_diagnosis '{expected_diag}' not in FAILURE_MODES")

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
    if not gold_ev and gold.get("expected_decision") != "refuse_due_to_policy":
        errors.append(f"{sid}: expected non-empty minimal gold evidence list")
    for eid in gold_ev:
        if eid not in event_ids:
            errors.append(f"{sid}: hidden evidence_event_id {eid} missing from event_trace")

    gold_counter = set(gold["counterevidence_event_ids"])
    for eid in gold_counter:
        if eid not in event_ids:
            errors.append(f"{sid}: counterevidence_event_id {eid} missing from event_trace")

    text = public_text(scenario)
    for term in PUBLIC_FORBIDDEN_TERMS:
        if term in text:
            errors.append(f"{sid}: public text contains forbidden term '{term}'")

    for dec in DECISIONS:
        if dec in text and dec not in ("escalate", "mark_unresolved"):
            errors.append(f"{sid}: public text leaks decision verb phrase '{dec}'")

    # hidden_gold leakage into public_input (decision enums checked above)
    answer = gold.get("expected_answer") or ""
    if len(answer) > 100 and answer.lower() in text:
        errors.append(f"{sid}: public text leaks full hidden_gold.expected_answer")

    is_hard_or_l34 = diff in ("L3", "L4", "L3_conditional_validity", "L4_cross_scope_adversarial_audit")
    if is_hard_or_l34 and events:
        sorted_events = sorted(events, key=lambda e: e.get("timestamp", "") or str(e.get("timestamp_order", "")))
        latest_event_id = sorted_events[-1].get("event_id") if sorted_events else None
        if latest_event_id and latest_event_id in gold_ev and len(gold_ev) == 1:
            errors.append(f"{sid}: L3/L4 has latest-event shortcut (latest event is the sole minimal evidence)")

    if split in ("test", "hard") or diff in ("L3", "L4"):
        non_bg = [e for e in events if not _is_background_event(e)]
        if len(non_bg) < 3:
            errors.append(f"{sid}: hard/L3/L4 scenario has too few non-background events ({len(non_bg)} < 3)")
        bg_in_gold = [eid for eid in gold_ev if any(e.get("event_id") == eid and _is_background_event(e) for e in events)]
        if bg_in_gold:
            errors.append(f"{sid}: background filler events appear in gold evidence: {bg_in_gold}")

    annotation_status = scenario.get("annotation_status") or scenario.get("metadata", {}).get("annotation_status")
    if split == "realistic":
        if annotation_status == "reviewed" and scenario.get("source_type") == "github_realistic":
            errors.append(
                f"{sid}: realistic github_realistic scenario cannot be auto-marked reviewed "
                "(requires manual validation script)"
            )
        if annotation_status not in ("reviewed", "synthetic_gold_unreviewed", "pending"):
            warnings.append(f"{sid}: realistic annotation_status={annotation_status!r} is non-standard")
        if annotation_status != "reviewed":
            msg = f"{sid}: realistic split is not manually reviewed (annotation_status={annotation_status!r})"
            warnings.append(msg)

    if split == "calibration" and packaging_final:
        warnings.append(
            f"{sid}: calibration row is smoke/quickstart only; exclude from headline table generation"
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
    counters = Counter()
    for row in rows:
        row_errors, row_warnings = validate_one(
            row,
            data_path=data_path,
            smoke=smoke,
            packaging_final=packaging_final,
        )
        errors.extend(row_errors)
        warnings.extend(row_warnings)
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
    auto_smoke = smoke
    if data_path is not None and any(x in data_path.parent.name for x in ("_30_en", "_20_en", "smoke")):
        auto_smoke = True
    v13_release = rows and all(
        (row.get("benchmark_version") == "v1.3")
        or (row.get("metadata") or {}).get("renderer") == "unified_renderer_v13"
        for row in rows
    )
    if not auto_smoke and not v13_release:
        for key, threshold in thresholds.items():
            rate = counters[key] / n
            if rate < threshold:
                errors.append(f"dataset rate {key}={rate:.3f} below {threshold:.2f}")
    return {
        "count": len(rows),
        "rates": {k: counters[k] / n for k in sorted(thresholds)},
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--smoke", action="store_true", help="Relax dataset-rate gates; warn on unreviewed realistic")
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
