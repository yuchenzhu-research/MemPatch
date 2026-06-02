#!/usr/bin/env python3
"""Generate the hard calibration split for the general agent-memory benchmark.

This emits ``sample_80_hard_en`` directly (public scenario text + hidden gold)
without a separate blueprint step. It is deliberately harder and more varied
than the earliest small renderer:

* multi-source event traces (7+ events with mixed trust levels / actors);
* distractor memories and cross-scope traps;
* stale-but-plausible notes that paraphrase the old answer;
* policy constraints, wrong-source attribution, hallucinated/false-premise
  claims, and forget / release / restore cases;
* a mix of direct answers and non-answer actions (escalate, ask_clarification,
  refuse_due_to_policy, mark_unresolved).

The public text never names the internal method or its components (see
``PUBLIC_FORBIDDEN_TERMS``). All hidden gold lives under ``hidden_gold``.

Determinism: output depends only on ``--count`` and ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import (
    DIFFICULTIES,
    DOMAINS,
    FAILURE_MODES,
)

# (noun, note source label, role/owner) per domain. None of these contain
# forbidden method-specific terms.
DOMAIN_NOUN = {
    "software_engineering_agent": ("release blocker", "deployment note", "build owner"),
    "enterprise_multi_tool_workflow": ("approval route", "tool handoff", "operations lead"),
    "customer_support_crm": ("support case", "account note", "account owner"),
    "calendar_task_workflow": ("meeting plan", "calendar hold", "coordinator"),
    "research_knowledge_work": ("research brief", "source note", "analyst"),
    "personal_assistant_preference": ("assistant preference", "profile note", "user delegate"),
    "ecommerce_recommendation": ("shopping profile", "recommendation note", "merchandising agent"),
    "data_analysis_bi": ("dashboard request", "metric note", "BI owner"),
}

# Per primary failure mode: status of the original (target) memory and of the
# introduced replacement memory, plus the expected operator decision. Every
# status is drawn from the canonical MEMORY_STATUSES vocabulary.
MODE_SPEC = {
    "stale_memory_reuse": {"target": "outdated", "replacement": "current", "decision": "use_current_memory"},
    "under_update": {"target": "outdated", "replacement": "current", "decision": "use_current_memory"},
    "over_update": {"target": "current", "replacement": "should_not_store", "decision": "use_current_memory"},
    "conflict_collapse": {"target": "unresolved", "replacement": "unresolved", "decision": "mark_unresolved"},
    "scope_leakage": {"target": "current", "replacement": "out_of_scope", "decision": "escalate"},
    "policy_violation": {"target": "current", "replacement": "should_not_store", "decision": "refuse_due_to_policy"},
    "wrong_source_attribution": {"target": "outdated", "replacement": "current", "decision": "use_current_memory"},
    "memory_hallucination": {"target": "current", "replacement": "should_not_store", "decision": "ask_clarification"},
    "unnecessary_memory_write": {"target": "current", "replacement": "should_not_store", "decision": "use_current_memory"},
    "failure_to_forget": {"target": "deleted", "replacement": "current", "decision": "use_current_memory"},
    "failure_to_release_or_restore": {"target": "restored", "replacement": "current", "decision": "use_current_memory"},
}

NON_ANSWER_DECISIONS = {"escalate", "ask_clarification", "refuse_due_to_policy", "mark_unresolved"}


def _difficulty(index: int) -> str:
    # Weight toward the two hardest tiers for a calibration split.
    bucket = index % 10
    if bucket == 0:
        return DIFFICULTIES[0]
    if bucket <= 3:
        return DIFFICULTIES[1]
    if bucket <= 6:
        return DIFFICULTIES[2]
    return DIFFICULTIES[3]


def _ts(n: int) -> str:
    day = datetime(2026, 2, 2) + timedelta(days=n)
    return day.strftime("%Y-%m-%dT09:%M:%SZ")


def _event(sid: str, n: int, text: str, *, trust: str, source: str, actor: str, scope: str, related: list[str] | None = None) -> dict:
    return {
        "event_id": f"e-{sid}-{n:02d}",
        "timestamp": _ts(n),
        "source": source,
        "actor": actor,
        "event_type": "workflow_event",
        "text": text,
        "trust_level": trust,
        "visibility_scope": scope,
        "related_memory_ids": related or [],
    }


def build_scenario(index: int) -> dict:
    domain = DOMAINS[index % len(DOMAINS)]
    primary = FAILURE_MODES[index % len(FAILURE_MODES)]
    spec = MODE_SPEC[primary]
    noun, note_name, owner = DOMAIN_NOUN[domain]
    owner_title = owner.title()

    sid = f"rb-hard-en-{index + 1:05d}"
    scope = f"workspace-{chr(65 + index % 5)}"
    other_scope = f"workspace-{chr(65 + (index + 2) % 5)}"
    case_id = f"C-{2000 + index:04d}"
    project_id = f"PROJ-{chr(65 + index % 26)}{11 + index % 80}"
    person_id = f"REF-{index % 800 + 100:03d}"

    target = f"m-{sid}-target"
    replacement = f"m-{sid}-replacement"

    # Coverage knobs (deterministic; tuned so the union of validator and
    # hard-split thresholds is satisfied across the full 80-scenario split).
    difficulty = _difficulty(index)
    include_distractor = (index % 10) < 6
    include_cross_scope = (index % 5) < 2 or difficulty == DIFFICULTIES[3]
    verified_over_trusted = (index % 2 == 0) or primary in {"stale_memory_reuse", "under_update", "failure_to_release_or_restore", "wrong_source_attribution"}
    false_premise = (index % 3 == 0) or primary in {"memory_hallucination", "wrong_source_attribution"}
    non_answer = spec["decision"] in NON_ANSWER_DECISIONS

    secondary_pool = [m for m in FAILURE_MODES if m != primary]
    secondary_count = 1 + (index % 3)  # 1..3 secondaries -> multi-trap flavour
    secondaries = [secondary_pool[(index + k) % len(secondary_pool)] for k in range(secondary_count)]

    # --- canonical answer texts (stale vs current differ by a distinctive token) ---
    stale_text = f"{case_id} should follow the earlier {noun} path recorded for {project_id}."
    current_text = f"{case_id} should follow the updated {noun} path verified for {project_id}."
    if primary == "over_update":
        current_text = f"Keep the existing {noun} for {case_id}; do not over-apply the broad change to unrelated items."
    elif primary == "conflict_collapse":
        current_text = f"Treat {case_id} as unresolved until {owner_title} reconciles two verified but incompatible {noun} updates."
    elif primary == "scope_leakage":
        current_text = f"Keep {case_id} limited to {scope}; the similar {noun} in {other_scope} must not be applied here."
    elif primary == "policy_violation":
        current_text = f"Do not store the private credential-like detail for {person_id}; route it through the secure exception queue."
    elif primary == "memory_hallucination":
        current_text = f"Ask {owner_title} to confirm {case_id}: no verified record supports the claimed {noun} change."
    elif primary == "unnecessary_memory_write":
        current_text = f"Do not store a new {noun} note for {case_id}; the existing verified record already covers it."
    elif primary == "failure_to_forget":
        current_text = f"Remove the obsolete {noun} attached to {case_id} and act only on the current verified record."
    elif primary == "failure_to_release_or_restore":
        current_text = f"Restore {case_id} because the verified release notice cleared the temporary block on the {noun}."
    elif primary == "wrong_source_attribution":
        current_text = f"{case_id} should follow the updated {noun} path verified by the system of record, not the forwarded comment."

    # --- initial memory (>=3 entries; includes ctx + optional distractor) ---
    initial_memory = [
        {
            "memory_id": target,
            "text": stale_text,
            "visibility_scope": scope,
            "source_event_ids": [f"e-{sid}-01"],
            "is_distractor": False,
        },
        {
            "memory_id": f"m-{sid}-ctx-1",
            "text": f"{owner_title} owns routine follow-up for {project_id} in {scope}.",
            "visibility_scope": scope,
            "source_event_ids": [f"e-{sid}-02"],
            "is_distractor": False,
        },
        {
            "memory_id": f"m-{sid}-ctx-2",
            "text": f"A standing note for {project_id} stays relevant only for routine routing in {scope}.",
            "visibility_scope": scope,
            "source_event_ids": [f"e-{sid}-02"],
            "is_distractor": False,
        },
    ]
    if include_distractor:
        initial_memory.append({
            "memory_id": f"m-{sid}-distractor",
            "text": f"A look-alike {noun} for {case_id[:-1]}9 uses the older path and is easy to confuse.",
            "visibility_scope": other_scope if include_cross_scope else scope,
            "source_event_ids": [f"e-{sid}-03"],
            "is_distractor": True,
        })

    # --- event trace (multi-source; 7+ events) ---
    events = [
        _event(sid, 1, f"{owner_title} recorded that {stale_text}", trust="trusted", source=note_name, actor=owner, scope=scope, related=[target]),
        _event(sid, 2, f"A routine check confirmed the initial routing metadata for {project_id}.", trust="trusted", source="tool_log", actor="workflow_tool", scope=scope, related=[target]),
        _event(sid, 3, f"A neighboring item in {other_scope} mentions a superficially similar {noun}.", trust="untrusted", source="forwarded_note", actor="external_agent", scope=other_scope, related=([f"m-{sid}-distractor"] if include_distractor else [])),
        _event(sid, 4, f"{owner_title} asked the assistant to decide the next action for {case_id} without assuming missing facts.", trust="trusted", source="user_message", actor=owner, scope=scope, related=[target]),
    ]
    # Wrong-source attribution trap: an untrusted comment claims to speak for the system of record.
    events.append(
        _event(sid, 5, f"A forwarded comment claims the {note_name} is authoritative for {case_id}, but cites no system record.", trust="untrusted", source="chat_forward", actor="external_agent", scope=scope, related=[target])
    )
    # The authoritative update (verified contradicts the earlier trusted note, or a plain trusted update).
    if verified_over_trusted:
        events.append(_event(sid, 6, f"Verified system record: {current_text}", trust="verified", source="system_record", actor="system_of_record", scope=scope, related=[target, replacement]))
    else:
        events.append(_event(sid, 6, f"Trusted update: {current_text}", trust="trusted", source=note_name, actor=owner, scope=scope, related=[target, replacement]))
    # False-premise trap.
    if false_premise:
        events.append(_event(sid, 7, f"An untrusted message insists the earlier instruction for {case_id} is still final, offering no verified support.", trust="untrusted", source="chat_forward", actor="external_agent", scope=scope, related=[target]))
    else:
        events.append(_event(sid, 7, f"An audit note records that no extra approvals were attached to {case_id}.", trust="trusted", source="audit_note", actor=owner, scope=scope))
    # Extra multi-source audit events to reach a deeper trace.
    extra = 1 + (index % 3)  # 1..3 extra events
    for k in range(extra):
        n = 8 + k
        events.append(_event(sid, n, f"Follow-up audit note {n}: keep decisions for {case_id} grounded in the visible verified records in {scope}.", trust="trusted", source="audit_note", actor=owner, scope=scope))

    evidence_event = f"e-{sid}-06"

    # --- introduced (replacement) memory via the authoritative event ---
    introduced = {
        replacement: {
            "memory_id": replacement,
            "text": current_text,
            "visibility_scope": scope,
            "introduced_by_event_id": evidence_event,
            "source_event_ids": [evidence_event],
            "is_distractor": False,
        }
    }

    # --- expected memory state ---
    expected_state: dict[str, str] = {}
    for m in initial_memory:
        if m.get("is_distractor"):
            expected_state[m["memory_id"]] = "out_of_scope" if include_cross_scope else "current"
        else:
            expected_state[m["memory_id"]] = "current"
    expected_state[target] = spec["target"]
    expected_state[replacement] = spec["replacement"]
    # Sprinkle a 'blocked' status so the full status vocabulary is exercised.
    if primary == "failure_to_release_or_restore":
        expected_state[f"m-{sid}-ctx-2"] = "blocked"

    # --- canonical answer / stale-or-wrong answers ---
    expected_answer = current_text
    if non_answer:
        expected_answer = f"{spec['decision']}: {current_text}"
    stale_or_wrong = [
        stale_text,
        f"Apply the {other_scope} note to {case_id} as if it were authoritative.",
    ]
    if primary in {"policy_violation", "memory_hallucination", "unnecessary_memory_write"}:
        stale_or_wrong.append(f"Store the new {noun} detail for {case_id} without any verified support.")

    # --- optional adversarial anchors (distinct from the correct answer) ---
    rubric: dict[str, object] = {
        "answer": "Prefer verified, in-scope, temporally latest evidence; reject unsupported, out-of-scope, or policy-violating claims.",
        "memory_state": "Classify each memory from the visible event trace; do not infer labels from wording alone.",
        "introduced_memories": introduced,
    }
    scope_anchors = [f"apply the {other_scope} note here", f"reuse the {other_scope} {noun} for {case_id}"]
    policy_anchors = [f"store the private credential-like detail for {person_id}", f"share the secured detail for {person_id}"]
    stale_anchors = [f"keep following the earlier path for {project_id} despite the verified update"]
    if primary == "scope_leakage":
        rubric["scope_leakage_anchors"] = scope_anchors
    if primary == "policy_violation":
        rubric["policy_violation_anchors"] = policy_anchors
    if primary in {"stale_memory_reuse", "under_update"}:
        rubric["stale_anchors"] = stale_anchors

    metadata = {
        "schema_version": "retrace_bench_general_1",
        "renderer": "hard_template",
        "split": "sample_80_hard_en",
        "has_distractor": include_distractor,
        "has_cross_scope_trap": include_cross_scope,
        "verified_contradicts_trusted_note": verified_over_trusted,
        "requires_rejecting_false_premise": false_premise,
        "requires_non_answer_action": non_answer,
        "introduced_memory_ids": [replacement],
        "event_count": len(events),
        "memory_count": len(initial_memory),
    }

    return {
        "scenario_id": sid,
        "domain": domain,
        "primary_failure_mode": primary,
        "secondary_failure_modes": secondaries,
        "difficulty": difficulty,
        "workflow_context": f"{owner_title} is coordinating a {noun} in {scope} for item {case_id}.",
        "public_input": {"event_trace": events, "initial_memory": initial_memory},
        "tasks": [
            {"task_id": f"t-{sid}-black-box", "task_type": "black_box_task", "prompt": f"What should the assistant do now for {case_id}?"},
            {"task_id": f"t-{sid}-memory-state", "task_type": "memory_state_task", "prompt": "Classify each memory as current, outdated, blocked, unresolved, out_of_scope, deleted, should_not_store, or restored."},
            {"task_id": f"t-{sid}-evidence", "task_type": "evidence_retrieval_task", "prompt": "Return the minimal event IDs that justify the decision."},
            {"task_id": f"t-{sid}-diagnostic", "task_type": "diagnostic_task", "prompt": "If an assistant follows the wrong note here, what memory reliability failure occurred?"},
        ],
        "hidden_gold": {
            "expected_answer": expected_answer,
            "expected_decision": spec["decision"],
            "expected_evidence_event_ids": [evidence_event],
            "expected_memory_state": expected_state,
            "expected_failure_diagnosis": primary,
            "stale_or_wrong_answers": stale_or_wrong,
            "rubric": rubric,
        },
        "metadata": metadata,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--out", default="data/retrace_bench/sample_80_hard_en/scenarios.jsonl")
    parser.add_argument("--seed", type=int, default=42)  # accepted for determinism parity; output is index-driven.
    args = parser.parse_args(argv)

    rows = [build_scenario(i) for i in range(args.count)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_name": out.parent.name,
        "scenario_count": len(rows),
        "schema_version": "retrace_bench_general_1",
        "renderer": "hard_template",
        "split": "calibration_hard",
    }
    out.parent.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} hard scenarios to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
