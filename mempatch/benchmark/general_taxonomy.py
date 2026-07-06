"""Taxonomy for the MemPatch-Bench final release."""

from __future__ import annotations

DOMAINS = (
    "software_release",
    "customer_support",
    "calendar_coordination",
    "research_notes",
    "business_intelligence",
    "personal_assistant",
)

FAILURE_MODES = (
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

DIFFICULTIES = ("medium", "hard", "challenge")

DIFFICULTY_DEFINITIONS = {
    "medium": "single revision hazard with explicit evidence",
    "hard": "multiple records or distractors with authority/scope reasoning",
    "challenge": "hard case with extra non-resolving evidence or lifecycle state",
}

DIFFICULTY_ALIASES = {
    "L1": "medium",
    "L2": "medium",
    "L3": "hard",
    "L4": "challenge",
    "L1_single_hop_update": "medium",
    "L2_multi_hop_with_distractor": "medium",
    "L3_conditional_validity": "hard",
    "L4_cross_scope_adversarial_audit": "challenge",
}

PATTERNS = (
    "temporal_supersession",
    "partial_update_required",
    "cross_scope_distractor",
    "verified_conflict",
    "policy_blocks_storage",
    "authority_misattribution",
    "unsupported_memory_claim",
    "overbroad_write",
    "forget_request",
    "release_after_hold",
    "write_not_needed",
)

MEMORY_STATUSES = (
    "current",
    "blocked",
    "unresolved",
    "out_of_scope",
    "should_not_store",
    "outdated",
    "deleted",
    "restored",
)

TASK_TYPES = (
    "black_box_task",
    "memory_state_task",
    "evidence_retrieval_task",
    "diagnostic_task",
    "followup_task",
)

TRUST_LEVELS = ("verified", "trusted", "untrusted")

DECISIONS = (
    "use_current_memory",
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
)

MEMORY_OPERATIONS = (
    "PRESERVE",
    "REVISE",
    "RESTRICT_SCOPE",
    "BLOCK",
    "MARK_UNRESOLVED",
    "DELETE_OR_FORGET",
    "RESTORE_OR_RELEASE",
    "REJECT_NEW_MEMORY",
    "NO_WRITE",
    "ESCALATE",
)

NON_ANSWER_DECISIONS = (
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
)

BENCH_SCHEMA_VERSION = "mempatch_bench_final"

PUBLIC_FORBIDDEN_TERMS = (
    "hidden_gold",
    "expected_decision",
    "expected_answer",
    "expected_memory_state",
    "resolver_trace",
    "primary_failure_mode",
)

FAILURE_MODE_DEFINITIONS = {
    "stale_memory_reuse": "A superseded memory remains the basis for the current answer or state.",
    "under_update": "A valid revision should propagate to linked memory state but only part is updated.",
    "over_update": "A valid revision is applied too broadly to records outside its intended target.",
    "conflict_collapse": "Credible in-scope evidence conflicts and is collapsed into an unsupported answer.",
    "scope_leakage": "Out-of-scope workspace/user/session evidence is treated as applicable.",
    "policy_violation": "The proposed memory action would store content that policy forbids.",
    "wrong_source_attribution": "A non-authoritative or forwarded source is treated as the system of record.",
    "memory_hallucination": "The assistant invents or accepts a memory claim without visible support.",
    "unnecessary_memory_write": "Durable memory is written when evidence supports only a one-shot answer.",
    "failure_to_forget": "A valid delete/forget request is ignored and the memory remains active.",
    "failure_to_release_or_restore": "A temporary hold/block is not correctly released or restored.",
}


def normalize_difficulty(value: object) -> str:
    text = str(value or "").strip()
    return DIFFICULTY_ALIASES.get(text, text)


def _state_list_to_map(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        out = {}
        for item in value:
            if isinstance(item, dict) and "memory_id" in item and "status" in item:
                out[item["memory_id"]] = item["status"]
        return out
    return {}


def canonical_hidden_gold_fields(gold: dict) -> dict:
    """Read canonical final hidden_gold fields, accepting legacy aliases."""
    return {
        "expected_decision": gold.get("expected_decision"),
        "expected_memory_operation": gold.get("expected_memory_operation"),
        "expected_answer": gold.get("expected_answer"),
        "expected_followup_answer": gold.get("expected_followup_answer"),
        "expected_followup_answer_key_facts": list(gold.get("expected_followup_answer_key_facts") or []),
        "unsafe_reuse_patterns": list(gold.get("unsafe_reuse_patterns") or []),
        "expected_memory_state": _state_list_to_map(
            gold.get("expected_memory_state") or gold.get("expected_memory_states") or {}
        ),
        "expected_failure_diagnosis": gold.get("expected_failure_diagnosis"),
        "expected_evidence_event_ids": list(gold.get("expected_evidence_event_ids") or []),
        "counterevidence_event_ids": list(gold.get("counterevidence_event_ids") or []),
        "rubric": gold.get("rubric") or {},
        "decision_aliases": gold.get("decision_aliases"),
        "stale_or_wrong_answers": gold.get("stale_or_wrong_answers") or [],
    }
