"""Reporting taxonomy for final MemPatch-Bench aggregate exports."""

from __future__ import annotations

from typing import Any


MEMORY_CAPABILITIES = (
    "factual_recall",
    "temporal_reasoning",
    "update_handling",
    "conflict_resolution",
    "multi_hop_memory_use",
    "abstention",
)

PATTERN_TO_CAPABILITY = {
    "temporal_supersession": "temporal_reasoning",
    "partial_update_required": "multi_hop_memory_use",
    "cross_scope_distractor": "conflict_resolution",
    "verified_conflict": "conflict_resolution",
    "policy_blocks_storage": "abstention",
    "authority_misattribution": "factual_recall",
    "unsupported_memory_claim": "abstention",
    "overbroad_write": "update_handling",
    "forget_request": "update_handling",
    "release_after_hold": "temporal_reasoning",
    "write_not_needed": "abstention",
}

FAILURE_MODE_TO_CAPABILITY = {
    "stale_memory_reuse": "update_handling",
    "under_update": "multi_hop_memory_use",
    "scope_leakage": "conflict_resolution",
    "conflict_collapse": "conflict_resolution",
    "policy_violation": "abstention",
    "wrong_source_attribution": "factual_recall",
    "memory_hallucination": "factual_recall",
    "over_update": "update_handling",
    "failure_to_forget": "update_handling",
    "failure_to_release_or_restore": "temporal_reasoning",
    "unnecessary_memory_write": "abstention",
}

OPERATION_TO_CAPABILITY = {
    "PRESERVE": "factual_recall",
    "REVISE": "update_handling",
    "RESTRICT_SCOPE": "conflict_resolution",
    "MARK_UNRESOLVED": "conflict_resolution",
    "BLOCK": "abstention",
    "REJECT_NEW_MEMORY": "abstention",
    "ESCALATE": "abstention",
    "DELETE_OR_FORGET": "update_handling",
    "RESTORE_OR_RELEASE": "temporal_reasoning",
    "NO_WRITE": "abstention",
}

METHOD_BASELINE_FAMILIES = {
    "direct_json": "no_memory_vanilla",
    "full_context_json": "full_or_truncated_context",
    "summary_memory_json": "summary_memory",
    "bm25_rag_json": "raw_rag",
    "dense_rag_json": "raw_rag",
    "time_aware_rag_json": "raw_rag",
    "mempatch_noguard": "mempatch_ablation",
    "mempatch": "mempatch",
}


def _string(value: Any) -> str:
    return str(value or "").strip()


def capability_for_score(row: dict[str, Any]) -> str:
    """Map a score row to a reviewer-facing memory capability category."""
    pattern = _string(row.get("pattern"))
    if pattern in PATTERN_TO_CAPABILITY:
        return PATTERN_TO_CAPABILITY[pattern]

    failure_mode = _string(row.get("failure_mode") or row.get("primary_failure_mode"))
    if failure_mode in FAILURE_MODE_TO_CAPABILITY:
        return FAILURE_MODE_TO_CAPABILITY[failure_mode]

    operation = _string(row.get("operation") or row.get("expected_memory_operation") or row.get("memory_operation"))
    if operation in OPERATION_TO_CAPABILITY:
        return OPERATION_TO_CAPABILITY[operation]

    return "unknown"


def baseline_family_for_method(method: str) -> str:
    return METHOD_BASELINE_FAMILIES.get(method, "unknown")
