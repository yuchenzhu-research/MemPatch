"""v1.3 blueprint registry: pattern families, decision variants, and visible triggers.

Each decision label must map to at least one pattern family with mutually exclusive
public triggers. Gold labels are produced by ``decision_resolver`` from rendered
events + blueprint params — never post-hoc relabeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from benchmark.general_taxonomy import DECISIONS

RENDERER = "unified_renderer_v13"
BENCHMARK_VERSION = "v1.3"

DecisionName = Literal[
    "use_current_memory",
    "mark_unresolved",
    "ask_clarification",
    "escalate",
    "refuse_due_to_policy",
]

# Visible trigger tokens emitted in public_input.event_trace text.
ASK_TRIGGERS = (
    "missing_target_scope",
    "ambiguous_user_intent",
    "ambiguous_workspace",
    "multiple_candidate_memories",
    "explicit_hold_for_user",
)

ESCALATE_TRIGGERS = (
    "policy_human_review_required",
    "verified_reviewer_hold",
    "security_compliance_block",
    "protected_production_memory",
    "evidence_sufficient_but_policy_blocks",
)

MARK_NON_CI_TRIGGERS = (
    "assume_duplicate_means_fixed",
    "treat_open_pr_as_merged",
    "trust_unverified_over_verified",
    "mark_verified_conflict",
    "mark_insufficient_passive",
    "mark_stalemate_no_authority",
)

MARK_CI_TRIGGERS = (
    "ci_second_verified_contradiction",
    "ci_passive_monitor_gap",
    "ci_no_authority_path",
)

USE_TRIGGERS = (
    "verified_maintainer_confirms",
    "verified_ci_release_confirms",
    "verified_auditor_confirms",
    "stable_scope_matches_target",
    "distractor_beta_out_of_scope",
)

REFUSE_TRIGGERS = (
    "credential_secret_forbidden",
    "security_policy_forbids_storage",
    "compliance_do_not_store",
)


@dataclass(frozen=True)
class V13DecisionVariant:
    """One learnable decision boundary within a pattern family."""

    variant_id: str
    decision: DecisionName
    triggers: tuple[str, ...]
    pattern_trap_type: str
    description: str
    mark_ci_derived: bool = False


@dataclass(frozen=True)
class V13PatternFamily:
    """Pattern family spanning one or more decision variants."""

    pattern: str
    domain: str
    primary_failure_mode: str
    variants: tuple[V13DecisionVariant, ...]


@dataclass
class V13BlueprintInstance:
    """Fully specified blueprint draw before rendering."""

    blueprint_id: str
    pattern: str
    pattern_trap_type: str
    decision_variant: str
    decision_triggers: list[str]
    split: str
    split_seed_namespace: str
    scenario_num: int
    difficulty: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario_id(self) -> str:
        return f"case-{self.scenario_num:06d}"


def _variant(
    variant_id: str,
    decision: DecisionName,
    triggers: tuple[str, ...],
    trap: str,
    description: str,
    *,
    mark_ci_derived: bool = False,
) -> V13DecisionVariant:
    return V13DecisionVariant(
        variant_id=variant_id,
        decision=decision,
        triggers=triggers,
        pattern_trap_type=trap,
        description=description,
        mark_ci_derived=mark_ci_derived,
    )


# Pilot registry: each decision has ≥2 pattern families; ask/esc/mark have ≥3 variants.
PATTERN_FAMILIES: tuple[V13PatternFamily, ...] = (
    V13PatternFamily(
        pattern="maintainer_correction_over_user_claim",
        domain="software_engineering_agent",
        primary_failure_mode="wrong_source_attribution",
        variants=(
            _variant(
                "verified_maintainer_overrides_user",
                "use_current_memory",
                ("verified_maintainer_confirms", "stable_scope_matches_target"),
                "trust_verified_maintainer",
                "Verified maintainer confirms current memory; user claim is distractor.",
            ),
            _variant(
                "missing_target_on_update_request",
                "ask_clarification",
                ("missing_target_scope", "explicit_hold_for_user"),
                "user_update_without_target",
                "User asks to update memory but omits workspace/version/target.",
            ),
        ),
    ),
    V13PatternFamily(
        pattern="authority_conflict",
        domain="enterprise_multi_tool_workflow",
        primary_failure_mode="conflict_collapse",
        variants=(
            _variant(
                "dual_verified_no_policy_gate",
                "mark_unresolved",
                ("mark_verified_conflict",),
                "verified_sources_disagree",
                "Two verified sources conflict; no policy gate or user clarification path.",
            ),
            _variant(
                "human_review_gate_active",
                "escalate",
                ("policy_human_review_required", "verified_reviewer_hold"),
                "reviewer_hold_blocks_auto_write",
                "Verified reviewer hold; policy forbids automatic memory update.",
            ),
            _variant(
                "ambiguous_scope_no_verified_ruling",
                "ask_clarification",
                ("ambiguous_workspace", "ambiguous_user_intent"),
                "stable_beta_both_active",
                "Stable and beta both have events; user must confirm scope.",
            ),
        ),
    ),
    V13PatternFamily(
        pattern="security_policy_override",
        domain="software_engineering_agent",
        primary_failure_mode="policy_violation",
        variants=(
            _variant(
                "credential_forbidden_write",
                "refuse_due_to_policy",
                ("credential_secret_forbidden", "security_policy_forbids_storage"),
                "secret_in_memory_request",
                "Request embeds credential-like content; storage forbidden.",
            ),
            _variant(
                "compliance_block_with_sufficient_evidence",
                "escalate",
                ("security_compliance_block", "evidence_sufficient_but_policy_blocks"),
                "compliance_blocks_despite_evidence",
                "Evidence supports update but compliance policy blocks automatic write.",
            ),
        ),
    ),
    V13PatternFamily(
        pattern="closed_as_duplicate_not_fixed",
        domain="customer_support_crm",
        primary_failure_mode="stale_memory_reuse",
        variants=(
            _variant(
                "duplicate_ticket_assumed_fixed",
                "mark_unresolved",
                ("assume_duplicate_means_fixed",),
                "duplicate_closure_trap",
                "Duplicate closure treated as fix without verified resolution.",
            ),
            _variant(
                "multiple_matching_memories",
                "ask_clarification",
                ("multiple_candidate_memories",),
                "crm_record_ambiguity",
                "Several CRM memories match the user request.",
            ),
        ),
    ),
    V13PatternFamily(
        pattern="negative_evidence_required",
        domain="research_knowledge_work",
        primary_failure_mode="memory_hallucination",
        variants=(
            _variant(
                "passive_monitor_gap",
                "mark_unresolved",
                ("mark_insufficient_passive",),
                "no_passive_confirmation",
                "No passive monitor or log confirms state; not a user-clarify case.",
            ),
            _variant(
                "trust_chain_broken",
                "mark_unresolved",
                ("mark_stalemate_no_authority",),
                "no_authority_path",
                "Trust chain broken; no authority path and no escalation process.",
            ),
        ),
    ),
    V13PatternFamily(
        pattern="ci_failed_after_claim",
        domain="software_engineering_agent",
        primary_failure_mode="conflict_collapse",
        variants=(
            _variant(
                "ci_contradiction_independent",
                "mark_unresolved",
                ("ci_second_verified_contradiction",),
                "ci_verified_contradiction",
                "Second verified CI/release event contradicts claim; independent mark skeleton.",
                mark_ci_derived=True,
            ),
            _variant(
                "ci_monitor_gap",
                "mark_unresolved",
                ("ci_passive_monitor_gap",),
                "ci_passive_gap",
                "CI state unknown due to passive monitor gap; not ask/escalate.",
                mark_ci_derived=True,
            ),
            _variant(
                "protected_prod_memory",
                "escalate",
                ("protected_production_memory", "policy_human_review_required"),
                "prod_memory_policy_gate",
                "Production memory protected; human review required despite CI signal.",
            ),
        ),
    ),
    V13PatternFamily(
        pattern="version_scope_leakage",
        domain="calendar_task_workflow",
        primary_failure_mode="scope_leakage",
        variants=(
            _variant(
                "verified_release_confirms_stable",
                "use_current_memory",
                ("verified_ci_release_confirms", "distractor_beta_out_of_scope"),
                "beta_distractor_ignored",
                "Verified stable release confirms memory; beta events are distractors.",
            ),
            _variant(
                "user_intent_ambiguous_action",
                "ask_clarification",
                ("ambiguous_user_intent", "missing_target_scope"),
                "apply_fix_unspecified",
                "User says apply the fix without specifying target memory.",
            ),
        ),
    ),
    V13PatternFamily(
        pattern="label_state_mismatch",
        domain="data_analysis_bi",
        primary_failure_mode="under_update",
        variants=(
            _variant(
                "verified_auditor_signoff",
                "use_current_memory",
                ("verified_auditor_confirms", "stable_scope_matches_target"),
                "auditor_confirms_current",
                "Auditor verified sign-off; current memory remains authoritative.",
            ),
            _variant(
                "compliance_do_not_store",
                "refuse_due_to_policy",
                ("compliance_do_not_store",),
                "bi_compliance_refusal",
                "Compliance policy forbids storing sensitive BI aggregate.",
            ),
        ),
    ),
)


def variants_for_decision(decision: str) -> list[V13DecisionVariant]:
    out: list[V13DecisionVariant] = []
    for family in PATTERN_FAMILIES:
        for variant in family.variants:
            if variant.decision == decision:
                out.append(variant)
    return out


def pattern_families_for_decision(decision: str) -> list[str]:
    patterns: set[str] = set()
    for family in PATTERN_FAMILIES:
        if any(v.decision == decision for v in family.variants):
            patterns.add(family.pattern)
    return sorted(patterns)


def all_variants() -> list[tuple[V13PatternFamily, V13DecisionVariant]]:
    pairs: list[tuple[V13PatternFamily, V13DecisionVariant]] = []
    for family in PATTERN_FAMILIES:
        for variant in family.variants:
            pairs.append((family, variant))
    return pairs


def validate_registry() -> list[str]:
    """Return registry errors (empty when pilot-ready)."""
    errors: list[str] = []
    decisions_seen: dict[str, set[str]] = {d: set() for d in DECISIONS}
    variant_ids: set[str] = set()

    for family in PATTERN_FAMILIES:
        for variant in family.variants:
            if variant.variant_id in variant_ids:
                errors.append(f"duplicate variant_id: {variant.variant_id}")
            variant_ids.add(variant.variant_id)
            decisions_seen[variant.decision].add(family.pattern)

            if variant.decision == "ask_clarification":
                if not set(variant.triggers) <= set(ASK_TRIGGERS):
                    errors.append(f"{variant.variant_id}: invalid ask triggers")
                if set(variant.triggers) & set(ESCALATE_TRIGGERS):
                    errors.append(f"{variant.variant_id}: ask variant has escalate triggers")
            elif variant.decision == "escalate":
                if not set(variant.triggers) <= set(ESCALATE_TRIGGERS):
                    errors.append(f"{variant.variant_id}: invalid escalate triggers")
            elif variant.decision == "mark_unresolved":
                allowed = set(MARK_NON_CI_TRIGGERS) | set(MARK_CI_TRIGGERS)
                if not set(variant.triggers) <= allowed:
                    errors.append(f"{variant.variant_id}: invalid mark triggers")

    for decision in DECISIONS:
        if len(decisions_seen[decision]) < 2:
            errors.append(f"{decision}: fewer than 2 pattern families ({decisions_seen[decision]})")

    for decision in ("ask_clarification", "escalate", "mark_unresolved"):
        n_variants = len(variants_for_decision(decision))
        if n_variants < 3:
            errors.append(f"{decision}: fewer than 3 decision variants (have {n_variants})")

    return errors
