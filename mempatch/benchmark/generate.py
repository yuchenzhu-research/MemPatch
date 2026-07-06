"""Deterministic final synthetic scenario generation.

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

from mempatch.benchmark.contracts import DECISIONS, FAILURE_MODES, MEMORY_OPERATIONS
from mempatch.benchmark.release import write_jsonl

DEFAULT_QUOTAS = {
    "dev_calibration": 500,
    "main_test_synthetic": 3000,
    "challenge_test_hard": 500,
}

SPLIT_DIFFICULTY_MIX = {
    "dev_calibration": ("medium", "medium", "hard", "hard", "challenge"),
    "main_test_synthetic": ("medium", "hard", "hard", "challenge"),
    "challenge_test_hard": ("medium", "hard", "challenge", "challenge", "challenge"),
}

STRUCTURE_CHOICES = {
    "medium": {
        "events": (4, 6, 8),
        "memories": (2, 3, 5),
        "distractors": (0, 1, 2),
    },
    "hard": {
        "events": (6, 8, 12),
        "memories": (3, 5, 8),
        "distractors": (1, 2, 4),
    },
    "challenge": {
        "events": (8, 12, 20),
        "memories": (5, 8, 12),
        "distractors": (2, 4, 6),
    },
}

TIMESTAMP_STYLES = ("iso", "date_only", "relative_turn", "unordered_evidence")
ID_STYLES = ("compact", "zero_padded", "issue_like", "release_note_like")
PROMPT_STYLES = ("compact", "verbose", "ticket", "table", "chat_log", "changelog")

DOMAINS = (
    "software_release",
    "customer_support",
    "calendar_coordination",
    "research_notes",
    "business_intelligence",
    "personal_assistant",
)

DOMAIN_PROFILES = {
    "software_release": {
        "topic": "release gate",
        "trusted_source": "release manager",
        "distractor_source": "old planning note",
        "context": "A persistent-memory agent maintains release records across engineering sessions.",
    },
    "customer_support": {
        "topic": "support entitlement",
        "trusted_source": "billing system",
        "distractor_source": "support macro",
        "context": "A persistent-memory agent maintains customer support records across tickets.",
    },
    "calendar_coordination": {
        "topic": "room booking",
        "trusted_source": "workspace calendar",
        "distractor_source": "other team's calendar",
        "context": "A persistent-memory agent maintains scheduling records across workspaces.",
    },
    "research_notes": {
        "topic": "literature claim",
        "trusted_source": "paper erratum",
        "distractor_source": "forwarded summary",
        "context": "A persistent-memory agent maintains research notes across reading sessions.",
    },
    "business_intelligence": {
        "topic": "dashboard metric",
        "trusted_source": "auditor",
        "distractor_source": "stale chart annotation",
        "context": "A persistent-memory agent maintains business metrics across reporting cycles.",
    },
    "personal_assistant": {
        "topic": "user preference",
        "trusted_source": "explicit user request",
        "distractor_source": "previous profile",
        "context": "A persistent-memory agent maintains personal assistant records across sessions.",
    },
}


@dataclass(frozen=True)
class ScenarioSpec:
    pattern: str
    failure_mode: str
    decision: str
    memory_operation: str
    target_status: str
    cue: str


SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        pattern="temporal_supersession",
        failure_mode="stale_memory_reuse",
        decision="use_current_memory",
        memory_operation="REVISE",
        target_status="current",
        cue="a newer verified record supersedes the older durable memory",
    ),
    ScenarioSpec(
        pattern="partial_update_required",
        failure_mode="under_update",
        decision="use_current_memory",
        memory_operation="REVISE",
        target_status="current",
        cue="two linked records must be kept consistent after the verified update",
    ),
    ScenarioSpec(
        pattern="cross_scope_distractor",
        failure_mode="scope_leakage",
        decision="mark_unresolved",
        memory_operation="RESTRICT_SCOPE",
        target_status="unresolved",
        cue="the latest-looking update belongs to a different workspace",
    ),
    ScenarioSpec(
        pattern="verified_conflict",
        failure_mode="conflict_collapse",
        decision="mark_unresolved",
        memory_operation="MARK_UNRESOLVED",
        target_status="unresolved",
        cue="two verified sources disagree and no authority rule resolves them",
    ),
    ScenarioSpec(
        pattern="policy_blocks_storage",
        failure_mode="policy_violation",
        decision="refuse_due_to_policy",
        memory_operation="BLOCK",
        target_status="should_not_store",
        cue="the requested durable storage is disallowed by the governing policy",
    ),
    ScenarioSpec(
        pattern="authority_misattribution",
        failure_mode="wrong_source_attribution",
        decision="use_current_memory",
        memory_operation="REVISE",
        target_status="current",
        cue="the system-of-record source overrides the forwarded summary",
    ),
    ScenarioSpec(
        pattern="unsupported_memory_claim",
        failure_mode="memory_hallucination",
        decision="ask_clarification",
        memory_operation="REJECT_NEW_MEMORY",
        target_status="blocked",
        cue="the requested durable memory has no supporting visible event",
    ),
    ScenarioSpec(
        pattern="overbroad_write",
        failure_mode="over_update",
        decision="ask_clarification",
        memory_operation="ESCALATE",
        target_status="blocked",
        cue="an update is valid for one target but ambiguous across siblings",
    ),
    ScenarioSpec(
        pattern="forget_request",
        failure_mode="failure_to_forget",
        decision="use_current_memory",
        memory_operation="DELETE_OR_FORGET",
        target_status="deleted",
        cue="a valid forget request should deactivate the old durable memory",
    ),
    ScenarioSpec(
        pattern="release_after_hold",
        failure_mode="failure_to_release_or_restore",
        decision="use_current_memory",
        memory_operation="RESTORE_OR_RELEASE",
        target_status="restored",
        cue="a later verified release event lifts a temporary block",
    ),
    ScenarioSpec(
        pattern="write_not_needed",
        failure_mode="unnecessary_memory_write",
        decision="use_current_memory",
        memory_operation="NO_WRITE",
        target_status="current",
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


def _difficulty_for(split: str, index: int) -> str:
    mix = SPLIT_DIFFICULTY_MIX.get(split, ("hard",))
    return mix[index % len(mix)]


def _domain_for(index: int) -> str:
    return DOMAINS[(index // len(SPECS) + index) % len(DOMAINS)]


def _event_id(scenario_id: str, suffix: str, order: int, id_style: str) -> str:
    if id_style == "zero_padded":
        return f"{scenario_id}_ev_{order + 1:03d}"
    if id_style == "issue_like":
        return f"{scenario_id}-comment-{order + 1:03d}"
    if id_style == "release_note_like":
        return f"{scenario_id}-note-{order + 1:03d}"
    return f"{scenario_id}_e{suffix}"


def _memory_id(scenario_id: str, role: str, order: int, id_style: str) -> str:
    if id_style == "zero_padded":
        return f"{scenario_id}_mem_{order:03d}"
    if id_style == "issue_like":
        return f"{scenario_id}-record-{order:02d}"
    if id_style == "release_note_like":
        return f"{scenario_id}/record/{order:02d}"
    return f"{scenario_id}_m{order:02d}"


def _timestamp(order: int, timestamp_style: str) -> str:
    if timestamp_style == "date_only":
        return f"2026-04-{(order % 28) + 1:02d}"
    if timestamp_style == "relative_turn":
        return f"T+{order}"
    if timestamp_style == "unordered_evidence":
        day = ((order * 7 + 3) % 28) + 1
        return f"2026-04-{day:02d}T{(9 + order) % 24:02d}:00:00Z"
    return f"2026-04-{(order % 28) + 1:02d}T10:{order % 60:02d}:00Z"


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
    id_style: str,
    timestamp_style: str,
    related_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": _event_id(scenario_id, suffix, order, id_style),
        "timestamp_order": order,
        "timestamp": _timestamp(order, timestamp_style),
        "actor_role": actor_role,
        "trust_level": trust_level,
        "visibility_scope": scope,
        "event_type": event_type,
        "content": content,
        "related_memory_ids": related_memory_ids or [],
    }


def _structure_profile(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    difficulty = _difficulty_for(split, index)
    choices = STRUCTURE_CHOICES[difficulty]
    return {
        "difficulty": difficulty,
        "num_events": rng.choice(choices["events"]),
        "num_memories": rng.choice(choices["memories"]),
        "num_distractors": rng.choice(choices["distractors"]),
        "timestamp_style": TIMESTAMP_STYLES[index % len(TIMESTAMP_STYLES)],
        "id_style": ID_STYLES[(index // len(TIMESTAMP_STYLES)) % len(ID_STYLES)],
        "prompt_style": PROMPT_STYLES[(index // (len(TIMESTAMP_STYLES) * len(ID_STYLES))) % len(PROMPT_STYLES)],
    }


def _task_prompt(style: str, case_ref: str, topic: str) -> str:
    if style == "ticket":
        return f"Ticket {case_ref}: resolve the durable-memory operation for the {topic} record."
    if style == "table":
        return f"Fill the final row for {case_ref}: memory record, operation, status, and evidence."
    if style == "chat_log":
        return f"In the current chat turn, decide how the agent should handle the {topic} memory for {case_ref}."
    if style == "changelog":
        return f"From the change log, identify the current memory lifecycle action for {case_ref}."
    if style == "verbose":
        return f"Review all visible memories and events, then decide the safe durable-memory update for {case_ref}'s {topic}."
    return f"Resolve the current memory status for {case_ref}."


def _followup_prompt(style: str, case_ref: str, topic: str) -> str:
    base = f"After applying your memory operation, answer the next user query about {case_ref}'s {topic}."
    if style == "ticket":
        return f"Follow-up ticket for {case_ref}: what should the agent now say about the {topic}?"
    if style == "table":
        return f"Follow-up row: final user-facing fact for {case_ref} / {topic}."
    if style == "chat_log":
        return f"User asks later: what is the current {topic} for {case_ref}?"
    if style == "changelog":
        return f"Next release-note query: what durable {topic} state applies to {case_ref}?"
    return base


def _answer(spec: ScenarioSpec, case_ref: str, topic: str, trusted_source: str) -> str:
    if spec.memory_operation == "REJECT_NEW_MEMORY":
        return f"Ask for support before creating a durable {topic} memory for {case_ref}."
    if spec.memory_operation == "BLOCK":
        return f"Refuse to store the {topic} for {case_ref}; the policy source forbids durable memory."
    if spec.memory_operation == "DELETE_OR_FORGET":
        return f"The old {topic} memory for {case_ref} should be treated as deleted."
    if spec.memory_operation == "RESTORE_OR_RELEASE":
        return f"The {topic} memory for {case_ref} is restored after the verified release event."
    if spec.memory_operation == "MARK_UNRESOLVED":
        return f"Mark the {topic} memory for {case_ref} unresolved until authority is clarified."
    if spec.memory_operation == "RESTRICT_SCOPE":
        return f"Keep the {topic} memory for {case_ref} unresolved in this scope and do not reuse the side-scope update."
    if spec.memory_operation == "ESCALATE":
        return f"Escalate before changing the {topic} memory for {case_ref}; the target is ambiguous."
    if spec.memory_operation == "NO_WRITE":
        return f"Answer the one-shot {topic} request for {case_ref} without writing a new durable memory."
    if spec.memory_operation == "PRESERVE":
        return f"Preserve the existing durable {topic} memory for {case_ref}."
    return f"Use the verified {trusted_source} evidence for the current {topic} state of {case_ref}."


def _followup_answer(spec: ScenarioSpec, case_ref: str, topic: str) -> str:
    if spec.memory_operation == "DELETE_OR_FORGET":
        return f"{case_ref} has no active durable {topic} memory after the forget request."
    if spec.memory_operation == "RESTORE_OR_RELEASE":
        return f"For {case_ref}, the {topic} is active again after the release event."
    if spec.memory_operation == "BLOCK":
        return f"No durable {topic} memory should be stored for {case_ref}."
    if spec.memory_operation == "REJECT_NEW_MEMORY":
        return f"The {topic} for {case_ref} remains unsupported until more evidence is provided."
    if spec.memory_operation in {"MARK_UNRESOLVED", "RESTRICT_SCOPE", "ESCALATE"}:
        return f"The {topic} for {case_ref} remains unresolved for this scope."
    if spec.memory_operation == "NO_WRITE":
        return f"There is no new durable {topic} memory to reuse for {case_ref}."
    return f"For {case_ref}, the durable {topic} should follow the verified current record."


def _memory_status(memory: dict[str, Any], spec: ScenarioSpec, target_id: str, condition_id: str) -> str:
    memory_id = memory["memory_id"]
    if memory_id == target_id:
        return spec.target_status
    if memory_id == condition_id:
        return "current"
    if memory.get("scope") != "workspace-main":
        return "out_of_scope"
    return "current"


def _memory_state(memories: list[dict[str, Any]], spec: ScenarioSpec, target_id: str, condition_id: str) -> dict[str, str]:
    return {
        str(memory["memory_id"]): _memory_status(memory, spec, target_id, condition_id)
        for memory in memories
    }


def _extra_event_content(case_ref: str, topic: str, order: int) -> str:
    variants = (
        f"Routine status synchronization mentions {case_ref} but adds no new authority for the {topic}.",
        f"A workspace handoff repeats a prior note about {case_ref} without changing the {topic}.",
        f"A side comment references {case_ref} while discussing a neighboring record.",
        f"An audit log asks reviewers to verify {case_ref} before changing durable memory.",
        f"A meeting note records background context for {case_ref} but no final decision.",
    )
    return variants[order % len(variants)]


def build_scenario(split: str, index: int, *, seed_namespace: str = "mempatch_final") -> dict[str, Any]:
    rng = random.Random(_seed(split, index, seed_namespace))
    spec = SPECS[index % len(SPECS)]
    scenario_id = _case_id(split, index)
    case_ref = f"CASE-{10000 + index}"
    domain = _domain_for(index)
    profile = DOMAIN_PROFILES[domain]
    structure = _structure_profile(split, index, rng)
    difficulty = structure["difficulty"]
    id_style = structure["id_style"]
    timestamp_style = structure["timestamp_style"]
    prompt_style = structure["prompt_style"]
    topic = str(profile["topic"])
    trusted_source = str(profile["trusted_source"])
    distractor_source = str(profile["distractor_source"])

    target_id = _memory_id(scenario_id, "target", 1, id_style)
    condition_id = _memory_id(scenario_id, "condition", 2, id_style)
    side_id = _memory_id(scenario_id, "side", 3, id_style)
    init_event_id = _event_id(scenario_id, "0", 0, id_style)

    min_memories = 3 if spec.failure_mode in {"scope_leakage", "conflict_collapse"} else 2
    memory_target = max(structure["num_memories"], min_memories)
    side_memory_needed = memory_target >= 3
    memories: list[dict[str, Any]] = [
        {
            "memory_id": target_id,
            "content": f"Prior memory: {case_ref} keeps an earlier {topic} state.",
            "scope": "workspace-main",
            "source_event_ids": [init_event_id],
            "memory_type": "operational_fact",
            "tags": [domain],
        },
        {
            "memory_id": condition_id,
            "content": f"Condition rule: {case_ref} memory writes require the most authoritative in-scope source.",
            "scope": "workspace-main",
            "source_event_ids": [init_event_id],
            "memory_type": "revision_rule",
            "tags": ["condition"],
        },
    ]
    if side_memory_needed:
        memories.append(
            {
                "memory_id": side_id,
                "content": f"Workspace note: another workspace has a different {topic} state for {case_ref}.",
                "scope": "workspace-side",
                "source_event_ids": [],
                "memory_type": "context_note",
                "tags": ["side_scope"],
                "is_distractor": True,
            }
        )
    while len(memories) < memory_target:
        order = len(memories) + 1
        scope = "workspace-side" if order <= structure["num_distractors"] + 3 else "workspace-main"
        memories.append(
            {
                "memory_id": _memory_id(scenario_id, f"aux{order}", order, id_style),
                "content": f"Auxiliary memory {order}: background context for {case_ref}'s {topic}.",
                "scope": scope,
                "source_event_ids": [],
                "memory_type": "context_note",
                "tags": ["background"],
            }
        )

    related_side_ids = [side_id] if side_memory_needed else []
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
            id_style=id_style,
            timestamp_style=timestamp_style,
        ),
        _event(
            scenario_id,
            "1",
            1,
            content=f"{trusted_source.title()} states that {spec.cue} for {case_ref}.",
            actor_role=trusted_source.replace(" ", "_"),
            trust_level="verified",
            scope="workspace-main",
            event_type="evidence",
            id_style=id_style,
            timestamp_style=timestamp_style,
            related_memory_ids=[target_id],
        ),
        _event(
            scenario_id,
            "2",
            2,
            content=f"{distractor_source.title()} gives a tempting but non-authoritative note about {case_ref}.",
            actor_role=distractor_source.replace(" ", "_"),
            trust_level="trusted" if spec.failure_mode != "wrong_source_attribution" else "untrusted",
            scope="workspace-side" if spec.failure_mode == "scope_leakage" else "workspace-main",
            event_type="side_note",
            id_style=id_style,
            timestamp_style=timestamp_style,
            related_memory_ids=related_side_ids,
        ),
        _event(
            scenario_id,
            "3",
            3,
            content=f"Current task asks whether the durable memory for {case_ref} should be revised, preserved, blocked, or scoped.",
            actor_role="user",
            trust_level="trusted",
            scope="workspace-main",
            event_type="request",
            id_style=id_style,
            timestamp_style=timestamp_style,
            related_memory_ids=[target_id],
        ),
    ]

    while len(events) < structure["num_events"]:
        order = len(events)
        related = [rng.choice(memories)["memory_id"]] if memories and rng.random() < 0.65 else []
        events.append(
            _event(
                scenario_id,
                str(order),
                order,
                content=_extra_event_content(case_ref, topic, order),
                actor_role=rng.choice(["auditor", "operator", "reviewer", "teammate"]),
                trust_level=rng.choice(["trusted", "verified", "untrusted"]),
                scope=rng.choice(["workspace-main", "workspace-side"]),
                event_type=rng.choice(["audit", "status_sync", "handoff", "comment", "meeting_note"]),
                id_style=id_style,
                timestamp_style=timestamp_style,
                related_memory_ids=related,
            )
        )

    evidence = [events[1]["event_id"], events[3]["event_id"]]
    if spec.failure_mode in {"conflict_collapse", "scope_leakage"}:
        evidence = [events[1]["event_id"], events[2]["event_id"]]
    if spec.failure_mode == "policy_violation":
        evidence = [events[1]["event_id"]]

    expected_answer = _answer(spec, case_ref, topic, trusted_source)
    expected_followup_answer = _followup_answer(spec, case_ref, topic)
    stale_answer = f"Keep the earlier {topic} state for {case_ref} without revision."
    return {
        "scenario_id": scenario_id,
        "public_split_name": split,
        "domain": domain,
        "difficulty": difficulty,
        "primary_failure_mode": spec.failure_mode,
        "pattern": spec.pattern,
        "workflow_context": profile["context"],
        "source_type": "controlled_synthetic",
        "source_pointers": [
            {
                "kind": "synthetic_generator",
                "url_or_id": "mempatch.benchmark.generate:final",
                "license_or_terms_note": "Generated benchmark scenario.",
            }
        ],
        "public_input": {
            "initial_memories": memories,
            "events": events,
        },
        "black_box_task": {"prompt": _task_prompt(prompt_style, case_ref, topic)},
        "memory_state_task": {"prompt": "Return a status for every visible memory."},
        "evidence_retrieval_task": {"prompt": "Return minimal event IDs supporting the memory operation."},
        "diagnostic_task": {"prompt": "Name the primary memory failure mode being tested."},
        "followup_task": {"prompt": _followup_prompt(prompt_style, case_ref, topic)},
        "hidden_gold": {
            "expected_decision": spec.decision,
            "expected_memory_operation": spec.memory_operation,
            "expected_answer": expected_answer,
            "expected_followup_answer": expected_followup_answer,
            "expected_followup_answer_key_facts": [case_ref, topic],
            "unsafe_reuse_patterns": [stale_answer, f"earlier {topic} state"],
            "expected_memory_state": _memory_state(memories, spec, target_id, condition_id),
            "expected_evidence_event_ids": evidence,
            "counterevidence_event_ids": [events[2]["event_id"]],
            "expected_failure_diagnosis": spec.failure_mode,
            "stale_or_wrong_answers": [stale_answer],
            "rubric": {
                "must_include": [case_ref, topic],
                "must_not_include": [stale_answer],
            },
        },
        "metadata": {
            "generator": "mempatch.benchmark.generate",
            "seed_namespace": seed_namespace,
            "pattern": spec.pattern,
            "structure": structure,
            "resolver_trace": {
                "rule": "spec_table_final",
                "expected_decision": spec.decision,
                "expected_memory_operation": spec.memory_operation,
                "expected_failure_mode": spec.failure_mode,
            },
        },
    }


def generate_split(split: str, count: int, *, seed_namespace: str = "mempatch_final") -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return [build_scenario(split, index, seed_namespace=seed_namespace) for index in range(count)]


def generate_raw_files(
    output_dir: Path,
    quotas: dict[str, int] | None = None,
    *,
    seed_namespace: str = "mempatch_final",
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
    gold = row.get("hidden_gold") or {}
    if row.get("primary_failure_mode") not in FAILURE_MODES:
        errors.append(f"{row.get('scenario_id')}: invalid failure mode")
    if gold.get("expected_decision") not in DECISIONS:
        errors.append(f"{row.get('scenario_id')}: invalid expected decision")
    if gold.get("expected_memory_operation") not in MEMORY_OPERATIONS:
        errors.append(f"{row.get('scenario_id')}: invalid expected memory operation")
    public = row.get("public_input") or {}
    if len(public.get("initial_memories") or []) < 2:
        errors.append(f"{row.get('scenario_id')}: expected at least two memories")
    if len(public.get("events") or []) < 4:
        errors.append(f"{row.get('scenario_id')}: expected at least four events")
    if "followup_task" not in row:
        errors.append(f"{row.get('scenario_id')}: missing followup task")
    if not gold.get("expected_followup_answer"):
        errors.append(f"{row.get('scenario_id')}: missing expected followup answer")
    return errors
