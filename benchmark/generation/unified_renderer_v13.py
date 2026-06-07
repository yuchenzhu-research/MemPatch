"""MemPatch v1.3 unified scenario renderer.

Renders public_input with visible decision triggers; gold decision is assigned
by ``decision_resolver`` after render, not here.
"""

from __future__ import annotations

import random
from typing import Any

from benchmark.generation.blueprints import (
    V13BlueprintInstance,
    V13DecisionVariant,
    V13PatternFamily,
)

DOMAIN_AGENTS: dict[str, str] = {
    "software_engineering_agent": "Release Engineer",
    "enterprise_multi_tool_workflow": "Workflow Coordinator",
    "customer_support_crm": "Support Agent",
    "research_knowledge_work": "Research Librarian",
    "calendar_task_workflow": "Calendar Coordinator",
    "data_analysis_bi": "BI Analyst",
}

PATTERN_TOPICS: dict[str, str] = {
    "maintainer_correction_over_user_claim": "API migration",
    "authority_conflict": "access policy",
    "security_policy_override": "credential handling",
    "closed_as_duplicate_not_fixed": "support ticket",
    "negative_evidence_required": "literature index",
    "ci_failed_after_claim": "deployment gate",
    "version_scope_leakage": "room booking",
    "label_state_mismatch": "dashboard metric",
}


def _case_ref(blueprint: V13BlueprintInstance) -> str:
    return f"CASE-{blueprint.scenario_num - 1}"


def _sid(blueprint: V13BlueprintInstance) -> str:
    return blueprint.scenario_id.replace("case-", "")


def _event(
    *,
    blueprint: V13BlueprintInstance,
    suffix: str,
    order: int,
    actor_role: str,
    trust_level: str,
    scope: str,
    event_type: str,
    text: str,
    related: list[str] | None = None,
) -> dict[str, Any]:
    eid = f"e-case-{_sid(blueprint)}-{suffix}"
    return {
        "event_id": eid,
        "timestamp_order": order,
        "actor_role": actor_role,
        "trust_level": trust_level,
        "visibility_scope": scope,
        "event_type": event_type,
        "text": text,
        "related_memory_ids": related or [],
        "timestamp": f"2027-01-{(order % 28) + 1:02d}T09:{order:02d}:00Z",
    }


def _memories(blueprint: V13BlueprintInstance, case_ref: str, topic: str) -> list[dict[str, Any]]:
    sid = _sid(blueprint)
    target = f"m-case-{sid}-target"
    condition = f"m-case-{sid}-condition"
    distractor = f"m-case-{sid}-distractor"
    return [
        {
            "memory_id": target,
            "text": f"Prior state: {case_ref} uses default {topic} configuration on stable v1.",
            "scope": "workspace-stable",
            "source_event_ids": ["e-init"],
            "is_distractor": False,
        },
        {
            "memory_id": condition,
            "text": f"Condition rule: Any update to {case_ref} requires verified approval on stable.",
            "scope": "workspace-stable",
            "source_event_ids": ["e-init"],
            "is_distractor": False,
        },
        {
            "memory_id": distractor,
            "text": f"Distractor info: {case_ref} has separate config for workspace-beta.",
            "scope": "workspace-beta",
            "source_event_ids": [f"e-case-{sid}-distractor-ver"],
            "is_distractor": True,
        },
    ]


def _memory_ids(blueprint: V13BlueprintInstance) -> dict[str, str]:
    sid = _sid(blueprint)
    return {
        "target": f"m-case-{sid}-target",
        "condition": f"m-case-{sid}-condition",
        "distractor": f"m-case-{sid}-distractor",
    }


def _tasks(case_ref: str, topic: str) -> dict[str, Any]:
    return {
        "black_box_task": {
            "prompt": f"What is the final authorized status and value of {case_ref} according to stable rules?",
            "output_schema": "JSON object with 'decision' and 'answer'",
        },
        "memory_state_task": {
            "prompt": "Classify active memory statuses for current entries.",
            "output_schema": "JSON object mapping memory_id -> status",
        },
        "evidence_retrieval_task": {
            "prompt": "Cite exact event IDs for minimal required evidence.",
            "output_schema": "JSON list of event_ids",
        },
        "diagnostic_task": {
            "prompt": "Diagnose the primary memory failure mode.",
            "output_schema": "JSON object with 'failure_diagnosis'",
        },
    }


def _difficulty_factors(*, events: int, policy: bool = False, ci: bool = False) -> dict[str, Any]:
    return {
        "num_events": events,
        "num_memories": 3,
        "authority_conflict": policy or ci,
        "scope_collision": False,
        "version_or_release_chain": ci,
        "branch_scope_conflict": False,
        "ci_or_test_state_required": ci,
        "multi_memory_coupling": False,
        "negative_evidence_required": False,
        "rollback_or_restore": False,
        "policy_or_security_constraint": policy,
        "minimal_evidence_required": True,
        "adversarial_distractors": 2,
    }


def _inject_triggers(events: list[dict[str, Any]], triggers: tuple[str, ...]) -> None:
    """Prefix core events with explicit visible trigger markers."""
    for idx, trigger in enumerate(triggers):
        if idx < len(events):
            events[idx]["text"] = f"[trigger:{trigger}] {events[idx]['text']}"


def _build_core_events(
    blueprint: V13BlueprintInstance,
    variant: V13DecisionVariant,
    *,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[str], list[str], str]:
    """Return (events, evidence_ids, counter_ids, expected_answer)."""
    case_ref = _case_ref(blueprint)
    topic = PATTERN_TOPICS.get(blueprint.pattern, "configuration")
    mids = _memory_ids(blueprint)
    vid = variant.variant_id

    specs: dict[str, tuple[list[dict[str, Any]], list[str], list[str], str]] = {}

    # --- use_current_memory variants ---
    specs["verified_maintainer_overrides_user"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text=f"Verified maintainer confirms {case_ref} {topic} on workspace-stable remains authoritative.",
                related=[mids["target"]],
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="auditor", trust_level="verified", scope="workspace-stable",
                event_type="audit",
                text=f"Audit record: stable scope matches target memory for {case_ref}.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="user", trust_level="untrusted", scope="workspace-stable",
                event_type="comment",
                text=f"Unverified user claims {case_ref} {topic} already changed (distractor only).",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [f"e-case-{_sid(blueprint)}-3"],
        f"Verified maintainer confirms the current {topic} for {case_ref} on workspace-stable.",
    )

    specs["verified_release_confirms_stable"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="release_note", trust_level="verified", scope="workspace-stable",
                event_type="release",
                text=f"Verified release confirms {case_ref} {topic} on stable v2 is active.",
                related=[mids["target"]],
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="ci", trust_level="verified", scope="workspace-stable",
                event_type="ci",
                text=f"Verified CI confirms stable build passed for {case_ref}.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="user", trust_level="trusted", scope="workspace-beta",
                event_type="comment",
                text=f"workspace-beta distractor: nightly {case_ref} differs from stable (out of scope).",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [f"e-case-{_sid(blueprint)}-3"],
        f"Verified release and CI confirm current {topic} for {case_ref} on workspace-stable.",
    )

    specs["verified_auditor_signoff"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="auditor", trust_level="verified", scope="workspace-stable",
                event_type="audit",
                text=f"Verified auditor confirms {case_ref} dashboard metric matches authorized baseline.",
                related=[mids["target"]],
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text=f"Stable scope matches target memory for {case_ref} after auditor review.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="reviewer", trust_level="trusted", scope="workspace-stable",
                event_type="review",
                text=f"Reviewer note: no pending changes for {case_ref} on stable.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Verified auditor sign-off keeps current {topic} for {case_ref} on workspace-stable.",
    )

    # --- ask_clarification ---
    specs["missing_target_on_update_request"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="user", trust_level="trusted", scope="workspace-stable",
                event_type="request",
                text=f"User: update the memory without specifying workspace, version, or target for {case_ref}.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text="Team lead: ask user to confirm before updating memory state.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="bot", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"System logged ambiguous update request for {case_ref}; no target memory selected.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Clarify with the user which workspace, version, and target memory should be updated for {case_ref}.",
    )

    specs["ambiguous_scope_no_verified_ruling"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="user", trust_level="trusted", scope="workspace-stable",
                event_type="request",
                text=f"User request for {case_ref} could mean update or ignore the {topic} change.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="maintainer", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"Stable and beta both have active {topic} notes for {case_ref}; no verified ruling yet.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="user", trust_level="trusted", scope="workspace-beta",
                event_type="comment",
                text=f"Beta workspace shows alternate {topic} for {case_ref} (unverified for stable).",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Ask the user to confirm whether {case_ref} changes apply to stable or beta workspace.",
    )

    specs["multiple_matching_memories"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="user", trust_level="trusted", scope="workspace-stable",
                event_type="request",
                text=f"User references ticket {case_ref}; multiple candidate memories match this request.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="support_bot", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"CRM shows two open records tied to {case_ref} {topic}.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="agent", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text="Support agent cannot pick a unique memory without user confirmation.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Ask the user which CRM memory record for {case_ref} should be updated.",
    )

    specs["user_intent_ambiguous_action"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="user", trust_level="trusted", scope="workspace-stable",
                event_type="request",
                text=f"User: apply the fix for {case_ref} — could mean update, delete, or ignore.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="calendar_bot", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"Calendar bot logged ambiguous_user_intent on {case_ref} {topic}.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="maintainer", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"No target memory specified for {case_ref} fix request.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Clarify whether the user wants to update, delete, or ignore the {topic} for {case_ref}.",
    )

    # --- escalate ---
    specs["human_review_gate_active"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="reviewer", trust_level="verified", scope="workspace-stable",
                event_type="review",
                text=f"Verified reviewer hold: human review required before any {case_ref} memory write.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text=f"Policy gate active for {case_ref}; automatic updates blocked pending approval.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="auditor", trust_level="verified", scope="workspace-stable",
                event_type="audit",
                text=f"Sufficient evidence exists for {case_ref} but policy_human_review_required applies.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Escalate {case_ref} {topic} change — human review required before memory update.",
    )

    specs["compliance_block_with_sufficient_evidence"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="security", trust_level="verified", scope="workspace-stable",
                event_type="policy",
                text=f"Security compliance block: evidence sufficient but policy blocks automatic write for {case_ref}.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text=f"Verified maintainer agrees evidence supports {case_ref} update; compliance block remains.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="compliance", trust_level="verified", scope="workspace-stable",
                event_type="policy",
                text="Compliance block prevents agent from storing revised policy text.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Escalate — compliance block prevents automatic memory update for {case_ref} despite sufficient evidence.",
    )

    specs["protected_prod_memory"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="security", trust_level="verified", scope="workspace-stable",
                event_type="policy",
                text=f"protected production memory: {case_ref} deployment gate cannot be auto-updated.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="ci", trust_level="verified", scope="workspace-stable",
                event_type="ci",
                text=f"CI reports failure on feature branch for {case_ref} (not sufficient for auto-write).",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text=f"human review required before changing production {topic} for {case_ref}.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-3"],
        [f"e-case-{_sid(blueprint)}-2"],
        f"Escalate {case_ref} — protected production memory requires human review before update.",
    )

    # --- mark_unresolved ---
    specs["dual_verified_no_policy_gate"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text=f"Verified maintainer: {case_ref} {topic} is enabled on stable.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="release_note", trust_level="verified", scope="workspace-stable",
                event_type="release",
                text=f"Verified release note: {case_ref} {topic} is disabled on stable — verified sources directly conflict.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="auditor", trust_level="verified", scope="workspace-stable",
                event_type="audit",
                text=f"No policy gate resolves the {case_ref} conflict; mark as unresolved.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Mark unresolved — verified sources directly conflict on {case_ref} {topic} with no authority path.",
    )

    specs["duplicate_ticket_assumed_fixed"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="support_bot", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"Ticket {case_ref} closed as duplicate means fixed without verified resolution.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="user", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"User still reports {topic} issue for {case_ref}; fix unverified.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="agent", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text="No verified maintainer confirmation that duplicate closure equals fix.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Mark unresolved — duplicate closure for {case_ref} does not verify the {topic} is fixed.",
    )

    specs["passive_monitor_gap"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="research_bot", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"Literature index for {case_ref} updated locally; passive monitor gap — no external confirmation.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="monitor", trust_level="trusted", scope="workspace-stable",
                event_type="monitor",
                text=f"Monitoring silent for {case_ref}; mark_insufficient_passive evidence.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="librarian", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"No user clarification path for {case_ref}; cannot determine authoritative state.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Mark unresolved — passive monitor gap prevents determining {case_ref} {topic} state.",
    )

    specs["trust_chain_broken"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="user", trust_level="untrusted", scope="workspace-stable",
                event_type="comment",
                text=f"Untrusted forward claims {case_ref} {topic} changed.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="bot", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"Trust chain broken for {case_ref}; no authority path to verify claim.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="maintainer", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text="No escalation process registered; stalemate on authoritative source.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Mark unresolved — no authority path to resolve {case_ref} {topic} claim.",
    )

    specs["ci_contradiction_independent"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text=f"Maintainer claims {case_ref} {topic} ready for stable.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="ci", trust_level="verified", scope="workspace-stable",
                event_type="ci",
                text=f"Second verified contradiction: CI rejects {case_ref} while maintainer claims ready.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="release_note", trust_level="verified", scope="workspace-stable",
                event_type="release",
                text=f"Release pipeline shows conflicting signals for {case_ref}; not a user-clarify case.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Mark unresolved — second verified contradiction on {case_ref} CI state without policy gate.",
    )

    specs["ci_monitor_gap"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="maintainer", trust_level="verified", scope="workspace-stable",
                event_type="comment",
                text=f"Maintainer merged change for {case_ref} {topic}.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="monitor", trust_level="verified", scope="workspace-stable",
                event_type="monitor",
                text=f"CI passive monitor gap: no post-merge CI result for {case_ref}.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="ci", trust_level="trusted", scope="workspace-stable",
                event_type="ci",
                text=f"Partial CI logs inconclusive for {case_ref}; cannot escalate or ask user.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Mark unresolved — CI passive monitor gap leaves {case_ref} deployment state unknown.",
    )

    # --- refuse ---
    specs["credential_forbidden_write"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="user", trust_level="trusted", scope="workspace-stable",
                event_type="request",
                text=f"User asks to store API secret sk-live-XXXX for {case_ref} — credential forbidden.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="security", trust_level="verified", scope="workspace-stable",
                event_type="policy",
                text="Security policy forbids storage of credential-like values in agent memory.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="compliance", trust_level="verified", scope="workspace-stable",
                event_type="policy",
                text=f"Compliance reminder: do not persist secrets for {case_ref}.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Refuse to store credential-like content for {case_ref}; security policy forbids storage.",
    )

    specs["compliance_do_not_store"] = (
        [
            _event(
                blueprint=blueprint, suffix="1", order=1,
                actor_role="user", trust_level="trusted", scope="workspace-stable",
                event_type="request",
                text=f"User asks to remember sensitive BI aggregate for {case_ref}.",
            ),
            _event(
                blueprint=blueprint, suffix="2", order=2,
                actor_role="compliance", trust_level="verified", scope="workspace-stable",
                event_type="policy",
                text="Compliance policy: do not store regulated BI aggregates in durable memory.",
            ),
            _event(
                blueprint=blueprint, suffix="3", order=3,
                actor_role="security", trust_level="verified", scope="workspace-stable",
                event_type="policy",
                text=f"Data governance block on {case_ref} metric retention.",
            ),
        ],
        [f"e-case-{_sid(blueprint)}-1", f"e-case-{_sid(blueprint)}-2"],
        [],
        f"Refuse — compliance says do not store the sensitive aggregate for {case_ref}.",
    )

    if vid not in specs:
        raise ValueError(f"No event spec for variant {vid}")

    core, evidence, counter, answer = specs[vid]
    _inject_triggers(core, variant.triggers)
    return core, evidence, counter, answer


def _memory_state_for_decision(
    decision: str,
    mids: dict[str, str],
) -> dict[str, str]:
    if decision == "use_current_memory":
        return {mids["target"]: "current", mids["condition"]: "current", mids["distractor"]: "out_of_scope"}
    if decision == "refuse_due_to_policy":
        return {mids["target"]: "should_not_store", mids["condition"]: "current", mids["distractor"]: "out_of_scope"}
    if decision == "ask_clarification":
        return {mids["target"]: "blocked", mids["condition"]: "current", mids["distractor"]: "out_of_scope"}
    if decision == "escalate":
        return {mids["target"]: "blocked", mids["condition"]: "current", mids["distractor"]: "out_of_scope"}
    # mark_unresolved
    return {mids["target"]: "unresolved", mids["condition"]: "current", mids["distractor"]: "out_of_scope"}


class MemPatchUnifiedRendererV13:
    """In-repo v1.3 renderer producing benchmark-valid scenarios."""

    def render(
        self,
        *,
        blueprint: V13BlueprintInstance,
        variant: V13DecisionVariant,
        family: V13PatternFamily,
        seed: int,
    ) -> dict[str, Any]:
        rng = random.Random(seed)
        case_ref = _case_ref(blueprint)
        topic = PATTERN_TOPICS.get(blueprint.pattern, "configuration")
        agent = DOMAIN_AGENTS.get(family.domain, "Coordinator")
        mids = _memory_ids(blueprint)

        core, evidence_ids, counter_ids, answer = _build_core_events(
            blueprint, variant, rng=rng,
        )

        sid = _sid(blueprint)
        distractors = [
            _event(
                blueprint=blueprint, suffix="distractor-ver", order=4,
                actor_role="user", trust_level="trusted", scope="workspace-beta",
                event_type="comment",
                text=f"Beta nightly note for {case_ref} {topic} (out-of-scope distractor).",
            ),
            _event(
                blueprint=blueprint, suffix="distractor-auth", order=5,
                actor_role="user", trust_level="untrusted", scope="workspace-stable",
                event_type="comment",
                text=f"Unverified chat claim about {case_ref} (distractor).",
            ),
        ]
        backgrounds = [
            _event(
                blueprint=blueprint, suffix="bg-1", order=6,
                actor_role="bot", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text=f"Routine status synchronization for PROJ-{sid} confirmed system heartbeat is normal.",
            ),
            _event(
                blueprint=blueprint, suffix="bg-2", order=7,
                actor_role="user", trust_level="trusted", scope="workspace-stable",
                event_type="comment",
                text="Gateway logged successful authentication check.",
            ),
        ]

        events = core + distractors + backgrounds
        memories = _memories(blueprint, case_ref, topic)

        decision = variant.decision
        memory_state = _memory_state_for_decision(decision, mids)

        hidden_gold = {
            "expected_answer": answer,
            "expected_memory_state": memory_state,
            "expected_evidence_event_ids": evidence_ids,
            "counterevidence_event_ids": counter_ids,
            "expected_failure_diagnosis": family.primary_failure_mode,
            "stale_or_wrong_answers": [
                f"Accept unverified user claim for {case_ref}.",
                f"Ignore stable-scope evidence for {case_ref}.",
            ],
            "rubric": {
                "must_include": [case_ref, topic.split()[0]],
                "must_not_include": ["workspace-beta"] if decision != "ask_clarification" else [],
            },
        }

        difficulty = blueprint.difficulty
        is_test = blueprint.split == "test"
        if is_test:
            difficulty = "L4"

        ci = blueprint.pattern == "ci_failed_after_claim"
        policy = decision in ("escalate", "refuse_due_to_policy") or "security" in blueprint.pattern

        return {
            "workflow_context": (
                f"{agent} is auditing {topic} and checking state integrity for {case_ref} "
                f"on {family.domain}."
            ),
            "public_input": {"event_trace": events, "initial_memory": memories},
            "hidden_gold": hidden_gold,
            "difficulty_level": difficulty,
            "difficulty_factors": _difficulty_factors(events=len(events), policy=policy, ci=ci),
            **_tasks(case_ref, topic),
        }


DEFAULT_RENDERER = MemPatchUnifiedRendererV13()
