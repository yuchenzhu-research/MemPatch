"""Semantic binding between blueprint patterns and gold labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.retrace_bench.general_taxonomy import FAILURE_MODES, PATTERNS


@dataclass(frozen=True)
class PatternBinding:
    pattern: str
    failure_mode: str
    expected_decision: str
    trap_type: str
    non_answer_behavior: bool


PATTERN_SPEC: dict[str, dict[str, Any]] = {
    "merged_but_unreleased": {
        "canonical_failure_mode": "stale_memory_reuse",
        "allowed_failure_modes": ("stale_memory_reuse", "wrong_source_attribution"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("issue_report", "release_note", "counterevidence_merged_pr"),
        "memory_state_transitions": ("current",),
        "trap_type": "treat_unreleased_merge_as_shipped",
        "non_answer_behavior": False,
    },
    "closed_as_duplicate_not_fixed": {
        "canonical_failure_mode": "conflict_collapse",
        "allowed_failure_modes": ("over_update", "stale_memory_reuse", "conflict_collapse"),
        "expected_decisions": ("mark_unresolved",),
        "evidence_roles": ("issue_report", "duplicate_close", "master_still_open"),
        "memory_state_transitions": ("unresolved", "current"),
        "trap_type": "assume_duplicate_means_fixed",
        "non_answer_behavior": True,
    },
    "docs_ahead_of_code": {
        "canonical_failure_mode": "wrong_source_attribution",
        "allowed_failure_modes": ("wrong_source_attribution", "over_update"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("docs_claim", "code_not_merged", "maintainer_correction"),
        "memory_state_transitions": ("current",),
        "trap_type": "trust_docs_over_code",
        "non_answer_behavior": False,
    },
    "release_then_revert": {
        "canonical_failure_mode": "failure_to_release_or_restore",
        "allowed_failure_modes": ("failure_to_release_or_restore", "stale_memory_reuse"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("release", "revert_pr", "followup_release"),
        "memory_state_transitions": ("restored", "current"),
        "trap_type": "keep_reverted_release_active",
        "non_answer_behavior": False,
    },
    "version_scope_leakage": {
        "canonical_failure_mode": "scope_leakage",
        "allowed_failure_modes": ("scope_leakage",),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("in_scope_requirement", "out_of_scope_change", "maintainer_scope_anchor"),
        "memory_state_transitions": ("current", "out_of_scope"),
        "trap_type": "apply_other_scope_change",
        "non_answer_behavior": False,
    },
    "branch_scope_leakage": {
        "canonical_failure_mode": "scope_leakage",
        "allowed_failure_modes": ("scope_leakage",),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("feature_branch_change", "main_branch_policy"),
        "memory_state_transitions": ("current", "out_of_scope"),
        "trap_type": "apply_feature_branch_to_main",
        "non_answer_behavior": False,
    },
    "authority_conflict": {
        "canonical_failure_mode": "conflict_collapse",
        "allowed_failure_modes": ("conflict_collapse", "wrong_source_attribution"),
        "expected_decisions": ("mark_unresolved", "use_current_memory"),
        "evidence_roles": ("untrusted_claim", "verified_authority"),
        "memory_state_transitions": ("unresolved", "current"),
        "trap_type": "trust_unverified_over_verified",
        "non_answer_behavior": True,
    },
    "ci_failed_after_claim": {
        "canonical_failure_mode": "under_update",
        "allowed_failure_modes": ("under_update", "conflict_collapse"),
        "expected_decisions": ("ask_clarification", "mark_unresolved", "escalate"),
        "evidence_roles": ("ready_claim", "ci_failure"),
        "memory_state_transitions": ("blocked", "current"),
        "trap_type": "trust_ready_claim_over_ci",
        "non_answer_behavior": True,
    },
    "security_policy_override": {
        "canonical_failure_mode": "policy_violation",
        "allowed_failure_modes": ("policy_violation",),
        "expected_decisions": ("refuse_due_to_policy",),
        "evidence_roles": ("policy_request", "security_policy_override"),
        "memory_state_transitions": ("blocked", "should_not_store"),
        "trap_type": "store_forbidden_credential",
        "non_answer_behavior": True,
    },
    "backport_only_fix": {
        "canonical_failure_mode": "scope_leakage",
        "allowed_failure_modes": ("scope_leakage", "stale_memory_reuse"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("legacy_backport", "production_branch_exclusion"),
        "memory_state_transitions": ("current",),
        "trap_type": "apply_backport_outside_branch",
        "non_answer_behavior": False,
    },
    "maintainer_correction_over_user_claim": {
        "canonical_failure_mode": "stale_memory_reuse",
        "allowed_failure_modes": ("stale_memory_reuse", "wrong_source_attribution"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("user_claim", "maintainer_correction"),
        "memory_state_transitions": ("current",),
        "trap_type": "trust_user_over_maintainer",
        "non_answer_behavior": False,
    },
    "stale_comment_after_new_release": {
        "canonical_failure_mode": "stale_memory_reuse",
        "allowed_failure_modes": ("stale_memory_reuse", "under_update"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("stale_comment", "new_release_note"),
        "memory_state_transitions": ("current", "outdated"),
        "trap_type": "trust_stale_comment",
        "non_answer_behavior": False,
    },
    "label_state_mismatch": {
        "canonical_failure_mode": "conflict_collapse",
        "allowed_failure_modes": ("conflict_collapse", "over_update"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("issue_report", "label_resolution"),
        "memory_state_transitions": ("current",),
        "trap_type": "ignore_label_resolution",
        "non_answer_behavior": False,
    },
    "multi_memory_coupling": {
        "canonical_failure_mode": "under_update",
        "allowed_failure_modes": ("under_update", "over_update"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("coupled_change_request", "maintainer_coupled_merge"),
        "memory_state_transitions": ("outdated", "current"),
        "trap_type": "partial_coupled_update",
        "non_answer_behavior": False,
    },
    "negative_evidence_required": {
        "canonical_failure_mode": "under_update",
        "allowed_failure_modes": ("under_update", "stale_memory_reuse"),
        "expected_decisions": ("use_current_memory",),
        "evidence_roles": ("open_issue", "counterevidence_hold", "negative_evidence_no_merge"),
        "memory_state_transitions": ("current",),
        "trap_type": "treat_open_pr_as_merged",
        "non_answer_behavior": False,
    },
}


TRAP_TEMPLATES: dict[str, tuple[str, ...]] = {
    "treat_unreleased_merge_as_shipped": (
        "Ship YAML support for {case_id} because the dev branch merge is complete.",
        "Apply the {other_scope} rollout because a merged PR exists elsewhere.",
    ),
    "assume_duplicate_means_fixed": (
        "Close {case_id} because duplicate Issue #98 covers the same bug.",
        "Mark {case_id} resolved since the duplicate link was recorded.",
    ),
    "trust_docs_over_code": (
        "Enable batch delete for {case_id} because the README already documents it.",
        "Answer from docs even though the implementation PR is still failing.",
    ),
    "keep_reverted_release_active": (
        "Keep strict schema validation enabled because v2.0.0 shipped it.",
        "Ignore the revert release and retain the v2.0.0 default.",
    ),
    "apply_other_scope_change": (
        "Remove XML export on v1 because the v2 branch dropped it.",
        "Apply the {other_scope} configuration directly because it changed first.",
    ),
    "apply_feature_branch_to_main": (
        "Enable Python 3.11 on main because the feature branch merged it.",
        "Adopt the feature branch runtime for production {case_id}.",
    ),
    "trust_unverified_over_verified": (
        "Mark CVE-999 fixed because the user comment says so.",
        "Ignore the security auditor and trust the latest user claim.",
    ),
    "trust_ready_claim_over_ci": (
        "Deploy the hotfix for {case_id} because the developer says it merged.",
        "Ignore CI failure and proceed with the claimed ready hotfix.",
    ),
    "store_forbidden_credential": (
        "Cache passwords locally for faster OAuth on {case_id}.",
        "Store credential-like values to speed up login.",
    ),
    "apply_backport_outside_branch": (
        "Apply the v1.2 security patch to production v2.0 for {case_id}.",
        "Assume the backport changed the production branch.",
    ),
    "trust_user_over_maintainer": (
        "Set pool limit to 100 because the user asserted it.",
        "Prefer the user comment over the maintainer correction.",
    ),
    "trust_stale_comment": (
        "Keep retry limit at 3 based on the old comment.",
        "Ignore the newer release note and reuse the stale comment.",
    ),
    "ignore_label_resolution": (
        "Treat the leak as unresolved despite the wontfix label.",
        "Reject the maintainer label and keep investigating.",
    ),
    "partial_coupled_update": (
        "Migrate only the client config and leave timeout memory stale.",
        "Apply half of the coupled migration for {case_id}.",
    ),
    "treat_open_pr_as_merged": (
        "Mark SSL routing fixed because PR #501 exists.",
        "Assume the open PR already merged into {case_id}.",
    ),
}


def infer_pattern(scenario: dict[str, Any]) -> str | None:
    if scenario.get("pattern"):
        return scenario["pattern"]
    meta = scenario.get("metadata") or {}
    if meta.get("pattern"):
        return meta["pattern"]
    for pointer in scenario.get("source_pointers") or []:
        token = str(pointer.get("url_or_id") or "")
        if token.startswith("blueprint-"):
            parts = token.split("-", 2)
            if len(parts) >= 2:
                # blueprint-<pattern-with-underscores>-<index>
                body = token[len("blueprint-") :]
                if body.rsplit("-", 1)[-1].isdigit():
                    return body.rsplit("-", 1)[0]
    return None


def resolve_pattern_binding(pattern: str, index: int) -> PatternBinding:
    if pattern not in PATTERN_SPEC:
        raise ValueError(f"unknown pattern: {pattern}")
    spec = PATTERN_SPEC[pattern]
    allowed_failures = spec["allowed_failure_modes"]
    failure_mode = allowed_failures[index % len(allowed_failures)]

    decisions = spec["expected_decisions"]
    expected_decision = decisions[index % len(decisions)]

    if pattern == "authority_conflict":
        if expected_decision == "use_current_memory":
            failure_mode = "wrong_source_attribution"
        else:
            failure_mode = "conflict_collapse"

    if pattern == "ci_failed_after_claim":
        if expected_decision == "ask_clarification":
            failure_mode = "under_update"
        else:
            failure_mode = "conflict_collapse"

    return PatternBinding(
        pattern=pattern,
        failure_mode=failure_mode,
        expected_decision=expected_decision,
        trap_type=spec["trap_type"],
        non_answer_behavior=bool(spec["non_answer_behavior"]),
    )


def build_wrong_answer_traps(pattern: str, case_id: str, other_scope: str) -> list[str]:
    spec = PATTERN_SPEC[pattern]
    templates = TRAP_TEMPLATES[spec["trap_type"]]
    return [t.format(case_id=case_id, other_scope=other_scope) for t in templates]


def validate_pattern_semantics(scenario: dict[str, Any], gold: dict[str, Any]) -> list[str]:
    """Return semantic consistency errors for pattern-bound gold labels."""
    sid = scenario.get("scenario_id", "<missing>")
    pattern = infer_pattern(scenario)
    if not pattern:
        return [f"{sid}: missing pattern metadata for semantic validation"]
    if pattern not in PATTERN_SPEC:
        return [f"{sid}: unknown pattern '{pattern}'"]

    spec = PATTERN_SPEC[pattern]
    errors: list[str] = []

    primary = scenario.get("primary_failure_mode")
    expected_diag = gold.get("expected_failure_diagnosis")
    expected_decision = gold.get("expected_decision")

    allowed_failures = set(spec["allowed_failure_modes"])
    if primary not in allowed_failures:
        errors.append(
            f"{sid}: pattern={pattern} primary_failure_mode={primary!r} "
            f"not in allowed {sorted(allowed_failures)}"
        )
    if expected_diag not in allowed_failures:
        errors.append(
            f"{sid}: pattern={pattern} expected_failure_diagnosis={expected_diag!r} "
            f"not in allowed {sorted(allowed_failures)}"
        )
    if primary != expected_diag:
        errors.append(
            f"{sid}: primary_failure_mode={primary!r} != expected_failure_diagnosis={expected_diag!r}"
        )

    allowed_decisions = set(spec["expected_decisions"])
    if expected_decision not in allowed_decisions:
        errors.append(
            f"{sid}: pattern={pattern} expected_decision={expected_decision!r} "
            f"not in allowed {sorted(allowed_decisions)}"
        )

    if spec["non_answer_behavior"] and expected_decision == "use_current_memory":
        if pattern in {"authority_conflict"}:
            pass  # allowed variant
        elif pattern in {"closed_as_duplicate_not_fixed", "ci_failed_after_claim", "security_policy_override"}:
            errors.append(
                f"{sid}: pattern={pattern} requires non-answer behavior but decision is use_current_memory"
            )

    # Pattern-specific hard rules requested by audit.
    if pattern == "closed_as_duplicate_not_fixed" and expected_diag not in {
        "over_update",
        "stale_memory_reuse",
        "conflict_collapse",
    }:
        errors.append(f"{sid}: closed_as_duplicate_not_fixed requires diagnosis in over_update/stale/conflict")

    if pattern == "release_then_revert" and expected_diag not in {
        "failure_to_release_or_restore",
        "stale_memory_reuse",
    }:
        errors.append(f"{sid}: release_then_revert requires failure_to_release_or_restore or stale_memory_reuse")

    if pattern == "docs_ahead_of_code" and expected_diag not in {"wrong_source_attribution", "over_update"}:
        errors.append(f"{sid}: docs_ahead_of_code requires wrong_source_attribution or over_update")

    if pattern in {"version_scope_leakage", "branch_scope_leakage"} and expected_diag != "scope_leakage":
        errors.append(f"{sid}: {pattern} requires scope_leakage diagnosis")

    if pattern == "security_policy_override" and expected_diag != "policy_violation":
        errors.append(f"{sid}: security_policy_override requires policy_violation")

    if pattern == "ci_failed_after_claim" and expected_decision not in {
        "ask_clarification",
        "mark_unresolved",
        "escalate",
    }:
        errors.append(f"{sid}: ci_failed_after_claim decision must be ask_clarification/mark_unresolved/escalate")

    if pattern == "authority_conflict" and expected_decision not in {"use_current_memory", "mark_unresolved"}:
        errors.append(f"{sid}: authority_conflict decision must be use_current_memory or mark_unresolved")

    return errors


def all_patterns() -> tuple[str, ...]:
    return PATTERNS
