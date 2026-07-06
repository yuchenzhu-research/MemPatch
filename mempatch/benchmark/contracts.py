"""MemPatch-Bench final public contracts.

This module is the small interface shared by generation, release export,
prediction validation, and scoring.  It intentionally uses JSON-compatible
plain dictionaries so dataset rows stay easy to inspect and publish.
"""

from __future__ import annotations

from typing import Any

DECISIONS = {
    "use_current_memory",
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
}

MEMORY_OPERATIONS = {
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
}

MEMORY_STATUSES = {
    "current",
    "blocked",
    "unresolved",
    "out_of_scope",
    "should_not_store",
    "outdated",
    "deleted",
    "restored",
}

FAILURE_MODES = {
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
}

REQUIRED_RESPONSE_FIELDS = (
    "answer",
    "decision",
    "memory_state",
    "evidence_event_ids",
    "failure_diagnosis",
)

OPTIONAL_RESPONSE_FIELDS = (
    "memory_operation",
    "followup_answer",
)

RESPONSE_FIELDS = REQUIRED_RESPONSE_FIELDS + OPTIONAL_RESPONSE_FIELDS


def state_list_to_map(rows: Any) -> dict[str, str]:
    """Normalize list-of-state records or legacy dicts to memory_id -> status."""
    if isinstance(rows, dict):
        return {str(key): str(value) for key, value in rows.items()}
    if not isinstance(rows, list):
        return {}
    out: dict[str, str] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        memory_id = item.get("memory_id")
        status = item.get("status")
        if memory_id is not None and status is not None:
            out[str(memory_id)] = str(status)
    return out


def state_map_to_list(state_map: Any) -> list[dict[str, str]]:
    """Normalize memory_id -> status maps to HF-friendly records."""
    if isinstance(state_map, list):
        return [
            {"memory_id": str(item["memory_id"]), "status": str(item["status"])}
            for item in state_map
            if isinstance(item, dict) and "memory_id" in item and "status" in item
        ]
    if isinstance(state_map, dict):
        return [
            {"memory_id": str(memory_id), "status": str(status)}
            for memory_id, status in state_map.items()
        ]
    return []


def normalize_prediction(row: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical final prediction row with a nested parsed object.

    For compatibility with older experiment outputs, rows with ``response`` are
    accepted and normalized to ``parsed``.
    """
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        parsed = row.get("response")
    if not isinstance(parsed, dict):
        parsed = {field: row[field] for field in RESPONSE_FIELDS if field in row}
    parsed = dict(parsed)
    followup = row.get("followup_response")
    if "followup_answer" not in parsed and isinstance(followup, dict) and "answer" in followup:
        parsed["followup_answer"] = followup["answer"]
    return {
        "scenario_id": row.get("scenario_id"),
        "method": row.get("method"),
        "model": row.get("model"),
        "parsed": parsed,
        "raw_response": row.get("raw_response"),
    }


def validate_prediction(prediction: dict[str, Any]) -> list[str]:
    """Return schema errors for the final prediction contract."""
    errors: list[str] = []
    parsed = prediction.get("parsed")
    if not isinstance(parsed, dict):
        return ["missing parsed response object"]
    for field in REQUIRED_RESPONSE_FIELDS:
        if field not in parsed:
            errors.append(f"missing parsed.{field}")
    decision = parsed.get("decision")
    if decision not in DECISIONS:
        errors.append(f"invalid decision: {decision!r}")
    operation = parsed.get("memory_operation")
    if operation is not None and operation not in MEMORY_OPERATIONS:
        errors.append(f"invalid memory_operation: {operation!r}")
    diagnosis = parsed.get("failure_diagnosis")
    if diagnosis not in FAILURE_MODES:
        errors.append(f"invalid failure_diagnosis: {diagnosis!r}")
    if not isinstance(parsed.get("evidence_event_ids", []), list):
        errors.append("parsed.evidence_event_ids must be a list")
    bad_statuses = sorted(
        {
            status
            for status in state_list_to_map(parsed.get("memory_state")).values()
            if status not in MEMORY_STATUSES
        }
    )
    if bad_statuses:
        errors.append(f"invalid memory statuses: {bad_statuses}")
    return errors
