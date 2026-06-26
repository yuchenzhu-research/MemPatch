"""Public-release leakage audit for MemPatch-Bench v1.4."""

from __future__ import annotations

import re
from typing import Any

from mempatch.benchmark.general_taxonomy import DIFFICULTIES, FAILURE_MODES, PATTERNS

FORBIDDEN_FIELDS = {
    "canonical_failure_mode",
    "decision_triggers",
    "difficulty",
    "difficulty_level",
    "expected_answer",
    "expected_decision",
    "expected_evidence_event_ids",
    "expected_failure_diagnosis",
    "expected_followup_answer",
    "expected_followup_answer_key_facts",
    "expected_memory_state",
    "expected_memory_states",
    "expected_memory_operation",
    "failure_mode",
    "generation_metadata",
    "hidden_gold",
    "is_distractor",
    "pattern",
    "pattern_trap_type",
    "primary_failure_mode",
    "resolver_trace",
    "source_pointers",
    "template_family_id",
    "template_instance_id",
    "unsafe_reuse_patterns",
    "validation_notes",
}

FORBIDDEN_KEY_ALIASES = {
    "answerkey",
    "answerkeys",
    "canonicalfailuremode",
    "decisiontrigger",
    "decisiontriggers",
    "difficulty",
    "difficultylevel",
    "expectedanswer",
    "expecteddecision",
    "expectedevidenceeventids",
    "expectedfailurediagnosis",
    "expectedfollowupanswer",
    "expectedfollowupanswerkeyfacts",
    "expectedmemorystate",
    "expectedmemorystates",
    "expectedmemoryoperation",
    "failuremode",
    "generationmetadata",
    "gold",
    "hidden",
    "hiddengold",
    "isdistractor",
    "oracle",
    "pattern",
    "patterntraptype",
    "primaryfailuremode",
    "resolvertrace",
    "sourcepointers",
    "templatefamilyid",
    "templateinstanceid",
    "unsafereusepatterns",
    "validationnotes",
}

FORBIDDEN_VALUES = frozenset(
    str(value).lower()
    for value in (*FAILURE_MODES, *PATTERNS, *DIFFICULTIES, "l1", "l2", "l3", "l4")
)


def _canonical_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _is_forbidden_key(key: object) -> bool:
    key_text = str(key)
    canonical = _canonical_key(key_text)
    return (
        key_text in FORBIDDEN_FIELDS
        or key_text.startswith("expected_")
        or canonical.startswith("expected")
        or canonical in FORBIDDEN_KEY_ALIASES
    )


def _is_forbidden_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if text in FORBIDDEN_VALUES:
        return True
    return any("_" in term and term in text for term in FORBIDDEN_VALUES)


def forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if _is_forbidden_key(key_text):
                paths.append(path)
            paths.extend(forbidden_paths(item, path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            paths.extend(forbidden_paths(item, f"{prefix}[{idx}]"))
    elif _is_forbidden_value(value):
        paths.append(prefix)
    return paths


def sanitize_public_value(value: Any) -> Any:
    """Drop internal keys and hidden taxonomy values from public payloads."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if _is_forbidden_key(key):
                continue
            sanitized = sanitize_public_value(item)
            if sanitized is not None and sanitized != [] and sanitized != {}:
                cleaned[key] = sanitized
        return cleaned
    if isinstance(value, list):
        return [
            sanitized
            for item in value
            if (sanitized := sanitize_public_value(item)) is not None
        ]
    if _is_forbidden_value(value):
        return None
    return value


def audit_public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        paths = forbidden_paths(row)
        if paths:
            violations.append({"scenario_id": row.get("scenario_id"), "paths": paths})
    return violations
