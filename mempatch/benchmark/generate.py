"""Deterministic v1.4 synthetic scenario generation.

The generator creates raw internal scenarios.  A separate release step strips
private fields into public rows and scorer-only labels.  This keeps generation
metadata useful for audits without leaking it into model inputs.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mempatch.benchmark.contracts import DECISIONS, FAILURE_MODES
from mempatch.benchmark.release import write_jsonl

DEFAULT_QUOTAS = {
    "dev_calibration": 60,
    "main_test_synthetic": 240,
    "challenge_test_hard": 120,
}

SPLIT_DIFFICULTY = {
    "dev_calibration": "medium",
    "main_test_synthetic": "hard",
    "challenge_test_hard": "challenge",
}

DOMAINS = (
    "software_release",
    "customer_support",
    "calendar_coordination",
    "research_notes",
    "business_intelligence",
    "personal_assistant",
)


@dataclass(frozen=True)
class ScenarioSpec:
    pattern: str
    domain: str
    failure_mode: str
    decision: str
    target_status: str
    topic: str
    trusted_source: str
    distractor_source: str
    cue: str


SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        pattern="temporal_supersession",
        domain="software_release",
        failure_mode="stale_memory_reuse",
        decision="use_current_memory",
        target_status="current",
        topic="release gate",
        trusted_source="release manager",
        distractor_source="old planning note",
        cue="a newer verified release note supersedes an older memory",
    ),
    ScenarioSpec(
        pattern="partial_update_required",
        domain="business_intelligence",
        failure_mode="under_update",
        decision="use_current_memory",
        target_status="current",
        topic="dashboard metric",
        trusted_source="auditor",
        distractor_source="stale chart annotation",
        cue="two linked memories must be kept consistent",
    ),
    ScenarioSpec(
        pattern="cross_scope_distractor",
        domain="calendar_coordination",
        failure_mode="scope_leakage",
        decision="mark_unresolved",
        target_status="unresolved",
        topic="room booking",
        trusted_source="workspace calendar",
        distractor_source="other team's calendar",
        cue="the latest-looking update belongs to a different workspace",
    ),
    ScenarioSpec(
        pattern="verified_conflict",
        domain="customer_support",
        failure_mode="conflict_collapse",
        decision="mark_unresolved",
        target_status="unresolved",
        topic="support entitlement",
        trusted_source="billing system",
        distractor_source="support macro",
        cue="two verified sources disagree and no authority rule resolves it",
    ),
    ScenarioSpec(
        pattern="policy_blocks_storage",
        domain="personal_assistant",
        failure_mode="policy_violation",
        decision="refuse_due_to_policy",
        target_status="should_not_store",
        topic="sensitive credential",
        trusted_source="security policy",
        distractor_source="user request",
        cue="the user asks to store content that policy forbids",
    ),
    ScenarioSpec(
        pattern="authority_misattribution",
        domain="research_notes",
        failure_mode="wrong_source_attribution",
        decision="use_current_memory",
        target_status="current",
        topic="literature claim",
        trusted_source="paper erratum",
        distractor_source="forwarded summary",
        cue="a forwarded summary conflicts with the system-of-record source",
    ),
    ScenarioSpec(
        pattern="unsupported_memory_claim",
        domain="customer_support",
        failure_mode="memory_hallucination",
        decision="ask_clarification",
        target_status="blocked",
        topic="refund preference",
        trusted_source="CRM record",
        distractor_source="assistant guess",
        cue="the requested memory has no supporting visible event",
    ),
    ScenarioSpec(
        pattern="overbroad_write",
        domain="software_release",
        failure_mode="over_update",
        decision="ask_clarification",
        target_status="blocked",
        topic="feature flag",
        trusted_source="maintainer",
        distractor_source="neighbor service",
        cue="an update is valid for one target but ambiguous across siblings",
    ),
    ScenarioSpec(
        pattern="forget_request",
        domain="personal_assistant",
        failure_mode="failure_to_forget",
        decision="use_current_memory",
        target_status="deleted",
        topic="user preference",
        trusted_source="explicit user request",
        distractor_source="previous profile",
        cue="a valid forget request should deactivate the old memory",
    ),
    ScenarioSpec(
        pattern="release_after_hold",
        domain="software_release",
        failure_mode="failure_to_release_or_restore",
        decision="use_current_memory",
        target_status="restored",
        topic="deployment hold",
        trusted_source="change advisory board",
        distractor_source="previous hold notice",
        cue="a later release event lifts a temporary block",
    ),
    ScenarioSpec(
        pattern="write_not_needed",
        domain="research_notes",
        failure_mode="unnecessary_memory_write",
        decision="use_current_memory",
        target_status="current",
        topic="one-off citation lookup",
        trusted_source="current request",
        distractor_source="note-taking habit",
        cue="the evidence supports a one-shot answer but not durable storage",
    ),
)


def _seed(split: str, index: int, seed_namespace: str) -> int:
    payload = f"{seed_namespace}:{split}:{index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _case_id(split: str, index: int) -> str:
    prefix = {
        "dev_calibration": "mp_dev",
        "main_test_synthetic": "mp_syn",
        "challenge_test_hard": "mp_hard",
    }.get(split, "mp")
    return f"{prefix}_{index:05d}"


def _event(
    scenario_id: str,
    suffix: str,
    order: int,
    *,
    content: str,
    actor_role: str,
    trust_level: str,
    scope: str,
    event_type: str,
    related_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": f"{scenario_id}_e{suffix}",
        "timestamp_order": order,
        "timestamp": f"2026-04-{(order % 28) + 1:02d}T10:{order:02d}:00Z",
        "actor_role": actor_role,
        "trust_level": trust_level,
        "visibility_scope": scope,
        "event_type": event_type,
        "content": content,
        "related_memory_ids": related_memory_ids or [],
    }


def _memory_state(spec: ScenarioSpec, target_id: str, condition_id: str, distractor_id: str) -> dict[str, str]:
    return {
        target_id: spec.target_status,
        condition_id: "current",
        distractor_id: "out_of_scope",
    }


def _answer(spec: ScenarioSpec, case_ref: str) -> str:
    if spec.decision == "ask_clarification":
        return f"Ask for the missing target before changing the {spec.topic} memory for {case_ref}."
    if spec.decision == "refuse_due_to_policy":
        return f"Refuse to store the {spec.topic} for {case_ref}; the policy source forbids durable memory."
    if spec.target_status == "deleted":
        return f"The old {spec.topic} memory for {case_ref} should be treated as deleted."
    if spec.target_status == "restored":
        return f"The {spec.topic} memory for {case_ref} is restored after the verified release event."
    if spec.decision == "mark_unresolved":
        return f"Mark the {spec.topic} memory for {case_ref} unresolved until authority is clarified."
    return f"Use the verified {spec.trusted_source} evidence for the current {spec.topic} state of {case_ref}."


def build_scenario(split: str, index: int, *, seed_namespace: str = "mempatch_v14") -> dict[str, Any]:
    rng = random.Random(_seed(split, index, seed_namespace))
    spec = SPECS[index % len(SPECS)]
    scenario_id = _case_id(split, index)
    case_ref = f"CASE-{10000 + index}"
    target_id = f"{scenario_id}_m_target"
    condition_id = f"{scenario_id}_m_condition"
    distractor_id = f"{scenario_id}_m_distractor"
    hard_extra = split == "challenge_test_hard"

    memories = [
        {
            "memory_id": target_id,
            "content": f"Prior memory: {case_ref} keeps an earlier {spec.topic} state.",
            "scope": "workspace-main",
            "source_event_ids": [f"{scenario_id}_e0"],
            "memory_type": "operational_fact",
            "tags": [spec.domain],
        },
        {
            "memory_id": condition_id,
            "content": f"Condition rule: {case_ref} memory writes require the most authoritative in-scope source.",
            "scope": "workspace-main",
            "source_event_ids": [f"{scenario_id}_e0"],
            "memory_type": "revision_rule",
            "tags": ["condition"],
        },
        {
            "memory_id": distractor_id,
            "content": f"Distractor memory: another scope has a different {spec.topic} state for {case_ref}.",
            "scope": "workspace-side",
            "source_event_ids": [],
            "memory_type": "distractor",
            "tags": ["distractor"],
            "is_distractor": True,
        },
    ]
    events = [
        _event(
            scenario_id,
            "0",
            0,
            content=f"Initial import recorded {case_ref} before the current revision window.",
            actor_role="system",
            trust_level="trusted",
            scope="workspace-main",
            event_type="import",
        ),
        _event(
            scenario_id,
            "1",
            1,
            content=f"{spec.trusted_source.title()} states that {spec.cue} for {case_ref}.",
            actor_role=spec.trusted_source.replace(" ", "_"),
            trust_level="verified",
            scope="workspace-main",
            event_type="evidence",
            related_memory_ids=[target_id],
        ),
        _event(
            scenario_id,
            "2",
            2,
            content=f"{spec.distractor_source.title()} gives a tempting but non-authoritative note about {case_ref}.",
            actor_role=spec.distractor_source.replace(" ", "_"),
            trust_level="trusted" if spec.failure_mode != "wrong_source_attribution" else "untrusted",
            scope="workspace-side" if spec.failure_mode == "scope_leakage" else "workspace-main",
            event_type="distractor",
            related_memory_ids=[distractor_id],
        ),
        _event(
            scenario_id,
            "3",
            3,
            content=f"Current task asks whether the durable memory for {case_ref} should be revised or reused.",
            actor_role="user",
            trust_level="trusted",
            scope="workspace-main",
            event_type="request",
            related_memory_ids=[target_id],
        ),
    ]
    if hard_extra:
        events.append(
            _event(
                scenario_id,
                "4",
                4,
                content=f"A later audit mentions {case_ref} but does not resolve the specific {spec.topic} state.",
                actor_role="auditor",
                trust_level=rng.choice(["trusted", "verified"]),
                scope="workspace-main",
                event_type="audit",
            )
        )

    evidence = [events[1]["event_id"], events[3]["event_id"]]
    if spec.failure_mode in {"conflict_collapse", "scope_leakage"}:
        evidence = [events[1]["event_id"], events[2]["event_id"]]
    if spec.failure_mode == "policy_violation":
        evidence = [events[1]["event_id"]]

    expected_answer = _answer(spec, case_ref)
    stale_answer = f"Keep the earlier {spec.topic} state for {case_ref} without revision."
    return {
        "scenario_id": scenario_id,
        "public_split_name": split,
        "domain": spec.domain,
        "difficulty": SPLIT_DIFFICULTY.get(split, "hard"),
        "primary_failure_mode": spec.failure_mode,
        "pattern": spec.pattern,
        "workflow_context": f"A persistent-memory agent maintains {spec.domain} records across sessions.",
        "source_type": "controlled_synthetic",
        "source_pointers": [
            {
                "kind": "synthetic_generator",
                "url_or_id": "mempatch.benchmark.generate:v1.4",
                "license_or_terms_note": "Generated benchmark scenario.",
            }
        ],
        "public_input": {
            "initial_memories": memories,
            "events": events,
        },
        "black_box_task": {"prompt": f"Resolve the current memory status for {case_ref}."},
        "memory_state_task": {"prompt": "Return a status for every visible memory."},
        "evidence_retrieval_task": {"prompt": "Return minimal event IDs supporting the decision."},
        "diagnostic_task": {"prompt": "Name the primary memory failure mode being tested."},
        "hidden_gold": {
            "expected_decision": spec.decision,
            "expected_answer": expected_answer,
            "expected_memory_state": _memory_state(spec, target_id, condition_id, distractor_id),
            "expected_evidence_event_ids": evidence,
            "counterevidence_event_ids": [events[2]["event_id"]],
            "expected_failure_diagnosis": spec.failure_mode,
            "stale_or_wrong_answers": [stale_answer],
            "rubric": {
                "must_include": [case_ref, spec.topic],
                "must_not_include": [stale_answer],
            },
        },
        "metadata": {
            "generator": "mempatch.benchmark.generate",
            "seed_namespace": seed_namespace,
            "pattern": spec.pattern,
            "resolver_trace": {
                "rule": "spec_table_v1.4",
                "expected_decision": spec.decision,
                "expected_failure_mode": spec.failure_mode,
            },
        },
    }


def generate_split(split: str, count: int, *, seed_namespace: str = "mempatch_v14") -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return [build_scenario(split, index, seed_namespace=seed_namespace) for index in range(count)]


def generate_raw_files(
    output_dir: Path,
    quotas: dict[str, int] | None = None,
    *,
    seed_namespace: str = "mempatch_v14",
) -> dict[str, Path]:
    quotas = quotas or DEFAULT_QUOTAS
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split, count in quotas.items():
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, generate_split(split, count, seed_namespace=seed_namespace))
        out[split] = path
    return out


def validate_generated_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if row.get("primary_failure_mode") not in FAILURE_MODES:
        errors.append(f"{row.get('scenario_id')}: invalid failure mode")
    if (row.get("hidden_gold") or {}).get("expected_decision") not in DECISIONS:
        errors.append(f"{row.get('scenario_id')}: invalid expected decision")
    public = row.get("public_input") or {}
    if len(public.get("initial_memories") or []) < 2:
        errors.append(f"{row.get('scenario_id')}: expected at least two memories")
    if len(public.get("events") or []) < 3:
        errors.append(f"{row.get('scenario_id')}: expected at least three events")
    return errors
