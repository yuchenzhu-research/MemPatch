"""Resolve expected_decision from rendered public events and blueprint params.

Gold labels must be derivable from visible public_input — never assigned after
the fact from hidden templates.
"""

from __future__ import annotations

import re
from typing import Any

from benchmark.generation.blueprints import (
    ASK_TRIGGERS,
    ESCALATE_TRIGGERS,
    MARK_CI_TRIGGERS,
    MARK_NON_CI_TRIGGERS,
    REFUSE_TRIGGERS,
    USE_TRIGGERS,
    V13BlueprintInstance,
    V13DecisionVariant,
)

# Marker phrases embedded by unified_renderer_v13 in event text (see design doc).
TRIGGER_PHRASES: dict[str, re.Pattern[str]] = {
    "missing_target_scope": re.compile(
        r"\[trigger:missing_target_scope\]|update (it|the memory) without specifying",
        re.I,
    ),
    "ambiguous_user_intent": re.compile(
        r"\[trigger:ambiguous_user_intent\]|could mean (update|delete|ignore)",
        re.I,
    ),
    "ambiguous_workspace": re.compile(
        r"\[trigger:ambiguous_workspace\]|stable and beta both",
        re.I,
    ),
    "multiple_candidate_memories": re.compile(
        r"\[trigger:multiple_candidate_memories\]|multiple candidate memories",
        re.I,
    ),
    "explicit_hold_for_user": re.compile(
        r"\[trigger:explicit_hold_for_user\]|ask user to confirm before updating",
        re.I,
    ),
    "policy_human_review_required": re.compile(
        r"\[trigger:policy_human_review_required\]|human review required before",
        re.I,
    ),
    "verified_reviewer_hold": re.compile(
        r"\[trigger:verified_reviewer_hold\]|verified reviewer hold",
        re.I,
    ),
    "security_compliance_block": re.compile(
        r"\[trigger:security_compliance_block\]|compliance block",
        re.I,
    ),
    "protected_production_memory": re.compile(
        r"\[trigger:protected_production_memory\]|protected production memory",
        re.I,
    ),
    "evidence_sufficient_but_policy_blocks": re.compile(
        r"\[trigger:evidence_sufficient_but_policy_blocks\]|policy blocks automatic",
        re.I,
    ),
    "mark_verified_conflict": re.compile(
        r"\[trigger:mark_verified_conflict\]|verified sources directly conflict",
        re.I,
    ),
    "mark_insufficient_passive": re.compile(
        r"\[trigger:mark_insufficient_passive\]|passive monitor gap",
        re.I,
    ),
    "mark_stalemate_no_authority": re.compile(
        r"\[trigger:mark_stalemate_no_authority\]|no authority path",
        re.I,
    ),
    "assume_duplicate_means_fixed": re.compile(
        r"\[trigger:assume_duplicate_means_fixed\]|duplicate means fixed",
        re.I,
    ),
    "treat_open_pr_as_merged": re.compile(
        r"\[trigger:treat_open_pr_as_merged\]|open pr treated as merged",
        re.I,
    ),
    "trust_unverified_over_verified": re.compile(
        r"\[trigger:trust_unverified_over_verified\]|unverified over verified",
        re.I,
    ),
    "ci_second_verified_contradiction": re.compile(
        r"\[trigger:ci_second_verified_contradiction\]|second verified contradiction",
        re.I,
    ),
    "ci_passive_monitor_gap": re.compile(
        r"\[trigger:ci_passive_monitor_gap\]|ci passive monitor gap",
        re.I,
    ),
    "ci_no_authority_path": re.compile(
        r"\[trigger:ci_no_authority_path\]|ci no authority path",
        re.I,
    ),
    "verified_maintainer_confirms": re.compile(
        r"\[trigger:verified_maintainer_confirms\]|verified maintainer confirms",
        re.I,
    ),
    "verified_ci_release_confirms": re.compile(
        r"\[trigger:verified_ci_release_confirms\]|verified (ci|release) confirms",
        re.I,
    ),
    "verified_auditor_confirms": re.compile(
        r"\[trigger:verified_auditor_confirms\]|verified auditor confirms",
        re.I,
    ),
    "stable_scope_matches_target": re.compile(
        r"\[trigger:stable_scope_matches_target\]|stable scope matches target",
        re.I,
    ),
    "distractor_beta_out_of_scope": re.compile(
        r"\[trigger:distractor_beta_out_of_scope\]|workspace-beta distractor",
        re.I,
    ),
    "credential_secret_forbidden": re.compile(
        r"\[trigger:credential_secret_forbidden\]|(?:api )?secret.*forbidden",
        re.I,
    ),
    "security_policy_forbids_storage": re.compile(
        r"\[trigger:security_policy_forbids_storage\]|security policy forbids storage",
        re.I,
    ),
    "compliance_do_not_store": re.compile(
        r"\[trigger:compliance_do_not_store\]|compliance.*do not store",
        re.I,
    ),
}


def collect_public_text(public_input: dict[str, Any]) -> str:
    chunks: list[str] = []
    for event in public_input.get("event_trace") or []:
        chunks.append(str(event.get("text") or ""))
        chunks.append(str(event.get("source") or ""))
    for memory in public_input.get("initial_memory") or []:
        chunks.append(str(memory.get("text") or ""))
    return "\n".join(chunks)


def detect_triggers(public_input: dict[str, Any]) -> set[str]:
    text = collect_public_text(public_input)
    found: set[str] = set()
    for trigger_id, pattern in TRIGGER_PHRASES.items():
        if pattern.search(text):
            found.add(trigger_id)
    return found


def _priority_decision(detected: set[str]) -> str | None:
    """Deterministic precedence when multiple trigger families appear."""
    if detected & set(REFUSE_TRIGGERS):
        return "refuse_due_to_policy"
    if detected & set(ESCALATE_TRIGGERS):
        return "escalate"
    if detected & set(ASK_TRIGGERS):
        return "ask_clarification"
    if detected & set(MARK_NON_CI_TRIGGERS) or detected & set(MARK_CI_TRIGGERS):
        return "mark_unresolved"
    if detected & set(USE_TRIGGERS):
        return "use_current_memory"
    return None


def resolve_expected_decision(
    *,
    blueprint: V13BlueprintInstance,
    variant: V13DecisionVariant,
    public_input: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return (expected_decision, resolver_trace).

    Raises ValueError when rendered public_input does not support the blueprint
    variant (trigger mismatch or cross-decision ambiguity).
    """
    detected = detect_triggers(public_input)
    expected_triggers = set(variant.triggers)
    missing = expected_triggers - detected
    if missing:
        raise ValueError(
            f"{blueprint.scenario_id}: missing visible triggers {sorted(missing)} "
            f"for variant {variant.variant_id}"
        )

    resolved = _priority_decision(detected)
    if resolved is None:
        raise ValueError(f"{blueprint.scenario_id}: no decision triggers detected in public_input")

    if resolved != variant.decision:
        cross = {
            "ask_clarification": ASK_TRIGGERS,
            "escalate": ESCALATE_TRIGGERS,
            "mark_unresolved": tuple(MARK_NON_CI_TRIGGERS) + tuple(MARK_CI_TRIGGERS),
        }
        for other_decision, other_triggers in cross.items():
            if other_decision == variant.decision:
                continue
            overlap = detected & set(other_triggers)
            if overlap:
                raise ValueError(
                    f"{blueprint.scenario_id}: cross-decision trigger overlap "
                    f"{sorted(overlap)} ({variant.decision} vs {other_decision})"
                )
        raise ValueError(
            f"{blueprint.scenario_id}: resolver got {resolved!r}, expected {variant.decision!r}"
        )

    trace = {
        "detected_triggers": sorted(detected),
        "expected_triggers": sorted(expected_triggers),
        "resolver": "v13_priority_decision",
        "mark_ci_derived": variant.mark_ci_derived,
    }
    return variant.decision, trace
