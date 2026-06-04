#!/usr/bin/env python3
"""Render general benchmark blueprints into public scenarios plus hidden gold."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DOMAIN_NOUN = {
    "software_engineering_agent": ("release blocker", "deployment note", "build owner"),
    "enterprise_multi_tool_workflow": ("approval route", "tool handoff", "operations lead"),
    "customer_support_crm": ("support case", "CRM note", "account owner"),
    "calendar_task_workflow": ("meeting plan", "calendar hold", "coordinator"),
    "research_knowledge_work": ("research brief", "source note", "analyst"),
    "personal_assistant_preference": ("assistant preference", "profile note", "user delegate"),
    "ecommerce_recommendation": ("shopping profile", "recommendation note", "merchandising agent"),
    "data_analysis_bi": ("dashboard request", "metric note", "BI owner"),
}


def article_for(phrase: str) -> str:
    return "an" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def event(bp: dict, n: int, text: str, *, trust: str = "trusted", source: str = "agent_note", actor: str = "workflow_agent", scope: str | None = None, related: list[str] | None = None) -> dict:
    day = datetime(2026, 1, 3) + timedelta(days=n)
    return {
        "event_id": f"e-{bp['scenario_id']}-{n:02d}",
        "timestamp": day.strftime("%Y-%m-%dT09:%M:%SZ"),
        "source": source,
        "actor": actor,
        "event_type": "workflow_event",
        "text": text,
        "trust_level": trust,
        "visibility_scope": scope or bp["scope"],
        "related_memory_ids": related or [],
    }


def render_one(bp: dict, rng: random.Random) -> dict:
    noun, note_name, owner = DOMAIN_NOUN[bp["domain"]]
    sid = bp["scenario_id"]
    target = bp["target_memory_id"]
    replacement = bp["replacement_memory_id"]
    stale_text = f"{bp['case_id']} should follow the earlier {noun} path for {bp['project_id']}."
    current_text = f"{bp['case_id']} should follow the updated {noun} path for {bp['project_id']}."
    if bp["primary_failure_mode"] == "policy_violation":
        current_text = f"Do not store the private credential-like detail for {bp['employee_id']}; handle it through the secure exception queue."
    elif bp["primary_failure_mode"] == "failure_to_forget":
        current_text = f"Remove the obsolete preference attached to {bp['employee_id']} and do not use it in future actions."
    elif bp["primary_failure_mode"] == "failure_to_release_or_restore":
        current_text = f"Restore {bp['case_id']} because the verified release notice cleared the temporary block."
    elif bp["primary_failure_mode"] == "scope_leakage":
        current_text = f"Keep {bp['case_id']} limited to {bp['scope']}; the similar instruction in {bp['other_scope']} must not be used here."
    elif bp["primary_failure_mode"] == "conflict_collapse":
        current_text = f"Treat {bp['case_id']} as unresolved pending {owner} reconciliation."

    initial_memory = [
        {
            "memory_id": target,
            "text": stale_text,
            "visibility_scope": bp["scope"],
            "source_event_ids": [f"e-{sid}-01"],
            "is_distractor": False,
        },
        {
            "memory_id": f"m-{sid}-ctx-0",
            "text": f"{owner.title()} handles routine follow-up for {bp['project_id']} in {bp['scope']}.",
            "visibility_scope": bp["scope"],
            "source_event_ids": [f"e-{sid}-02"],
            "is_distractor": False,
        },
    ]
    while len(initial_memory) < max(3, bp["memory_count"] - 1):
        idx = len(initial_memory)
        initial_memory.append({
            "memory_id": f"m-{sid}-ctx-{idx}",
            "text": f"Context note {idx} for {bp['project_id']} remains useful only for routine routing.",
            "visibility_scope": bp["scope"],
            "source_event_ids": [f"e-{sid}-02"],
            "is_distractor": False,
        })
    if bp["include_distractor"]:
        initial_memory.append({
            "memory_id": f"m-{sid}-distractor",
            "text": f"A similar {noun} for PROJ-{chr(90 - (len(sid) % 10))}2 uses the older path.",
            "visibility_scope": bp["scope"],
            "source_event_ids": [f"e-{sid}-03"],
            "is_distractor": True,
        })

    events = [
        event(bp, 1, f"{owner.title()} recorded that {stale_text}", trust="trusted", source=note_name, actor=owner, related=[target]),
        event(bp, 2, f"A routine check confirmed the initial routing metadata for {bp['project_id']}.", trust="trusted", source="tool_log", actor="workflow_tool", related=[target]),
        event(bp, 3, f"A neighboring item in {bp['other_scope']} mentions a superficially similar {noun}.", trust="untrusted" if bp["include_cross_scope_trap"] else "trusted", source="forwarded_note", actor="external_agent", scope=bp["other_scope"]),
        event(bp, 4, f"{owner.title()} asked the assistant to decide the next action for {bp['case_id']} without assuming missing facts.", trust="trusted", source="user_message", actor=owner, related=[target]),
    ]
    evidence_n = 5 if bp["verified_contradicts_trusted_note"] else 4
    if bp["verified_contradicts_trusted_note"]:
        events.append(event(bp, 5, f"Verified system record: {current_text}", trust="verified", source="system_record", actor="system_of_record", related=[target, replacement]))
    else:
        events.append(event(bp, 5, f"Trusted update: {current_text}", trust="trusted", source=note_name, actor=owner, related=[target, replacement]))
    if bp["requires_rejecting_false_premise"]:
        events.append(event(bp, 6, f"An untrusted comment claims the old instruction is still final, but it cites no system record.", trust="untrusted", source="chat_forward", actor="external_agent", related=[target]))
    else:
        events.append(event(bp, 6, f"The workflow tool logged that no additional approvals were attached to this item.", trust="trusted", source="tool_log", actor="workflow_tool"))
    while len(events) < bp["event_count"]:
        n = len(events) + 1
        events.append(event(bp, n, f"Follow-up audit note {n}: keep decisions limited to {bp['case_id']} and the visible records above.", trust="trusted", source="audit_note", actor=owner))

    introduced = {
        replacement: {
            "memory_id": replacement,
            "text": current_text,
            "visibility_scope": bp["scope"],
            "introduced_by_event_id": f"e-{sid}-{evidence_n:02d}",
            "source_event_ids": [f"e-{sid}-{evidence_n:02d}"],
            "is_distractor": False,
        }
    }
    expected_state = {m["memory_id"]: "current" for m in initial_memory if not m.get("is_distractor")}
    expected_state[target] = bp["expected_status"] if bp["expected_status"] != "restored" else "outdated"
    expected_state[replacement] = bp["expected_status"] if bp["expected_status"] == "restored" else "current"
    if bp["primary_failure_mode"] in {"policy_violation", "memory_hallucination", "unnecessary_memory_write"}:
        expected_state[replacement] = "should_not_store"
    if bp["primary_failure_mode"] == "scope_leakage":
        expected_state[replacement] = "out_of_scope"
    if bp["primary_failure_mode"] == "conflict_collapse":
        expected_state[replacement] = "unresolved"
    for m in initial_memory:
        if m.get("is_distractor"):
            expected_state[m["memory_id"]] = "out_of_scope" if bp["include_cross_scope_trap"] else "current"

    expected_answer = current_text
    if bp["expected_decision"] in {"refuse_due_to_policy", "ask_clarification", "escalate", "mark_unresolved"}:
        expected_answer = f"{bp['expected_decision']}: {current_text}"

    tasks = [
        {"task_id": f"t-{sid}-black-box", "task_type": "black_box_task", "prompt": f"What should the assistant do now for {bp['case_id']}?"},
        {"task_id": f"t-{sid}-memory-state", "task_type": "memory_state_task", "prompt": "Classify each memory as current, outdated, blocked, unresolved, out_of_scope, deleted, should_not_store, or restored."},
        {"task_id": f"t-{sid}-evidence", "task_type": "evidence_retrieval_task", "prompt": "Return the minimal event IDs that support the decision."},
        {"task_id": f"t-{sid}-diagnostic", "task_type": "diagnostic_task", "prompt": "If an assistant follows the wrong note here, what memory reliability failure occurred?"},
    ]
    return {
        "scenario_id": sid,
        "domain": bp["domain"],
        "primary_failure_mode": bp["primary_failure_mode"],
        "secondary_failure_modes": bp["secondary_failure_modes"],
        "difficulty": bp["difficulty"],
        "workflow_context": f"{owner.title()} is coordinating {article_for(noun)} {noun} in {bp['scope']} for synthetic item {bp['case_id']}.",
        "public_input": {"event_trace": events, "initial_memory": initial_memory},
        "tasks": tasks,
        "hidden_gold": {
            "expected_answer": expected_answer,
            "expected_decision": bp["expected_decision"],
            "expected_evidence_event_ids": [f"e-{sid}-{evidence_n:02d}"],
            "expected_memory_state": expected_state,
            "expected_failure_diagnosis": bp["primary_failure_mode"],
            "stale_or_wrong_answers": [stale_text, f"Use the note from {bp['other_scope']} for {bp['case_id']}."],
            "rubric": {
                "answer": "Prefer verified, in-scope, temporally latest evidence and reject unsupported or out-of-scope claims.",
                "memory_state": "Use blueprint-derived statuses; do not infer labels from wording alone.",
                "introduced_memories": introduced,
            },
        },
        "metadata": {
            "schema_version": "retrace_bench_general_1",
            "renderer": "template",
            "pattern": bp.get("pattern"),
            "blueprint_seed_marker": bp["seed_marker"],
            "has_distractor": bp["include_distractor"],
            "has_cross_scope_trap": bp["include_cross_scope_trap"],
            "verified_contradicts_trusted_note": bp["verified_contradicts_trusted_note"],
            "requires_rejecting_false_premise": bp["requires_rejecting_false_premise"],
            "requires_non_answer_action": bp["requires_non_answer_action"],
            "introduced_memory_ids": [replacement],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--renderer", default="template", choices=["template"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    rows = [render_one(bp, rng) for bp in read_jsonl(Path(args.blueprints))]
    write_jsonl(Path(args.out), rows)
    manifest = {
        "dataset_name": Path(args.out).parent.name,
        "scenario_count": len(rows),
        "schema_version": "retrace_bench_general_1",
        "renderer": args.renderer,
    }
    Path(args.out).parent.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} scenarios to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
