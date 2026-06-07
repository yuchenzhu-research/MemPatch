"""Neutral taxonomy for the general English MemPatch-Bench release."""

PRIMARY_DOMAINS = (
    "software_engineering_agent",
    "enterprise_multi_tool_workflow",
    "customer_support_crm",
    "calendar_task_workflow",
    "research_knowledge_work",
    "data_analysis_bi",
)

RESERVED_DOMAINS = (
    "personal_assistant_preference",
    "ecommerce_recommendation",
)

DOMAINS = PRIMARY_DOMAINS + RESERVED_DOMAINS

PRIMARY_FAILURE_MODES = (
    "stale_memory_reuse",
    "under_update",
    "conflict_collapse",
    "scope_leakage",
    "policy_violation",
    "wrong_source_attribution",
    "memory_hallucination",
)

RESERVED_FAILURE_MODES = (
    "over_update",
    "unnecessary_memory_write",
    "failure_to_forget",
    "failure_to_release_or_restore",
)

FAILURE_MODES = PRIMARY_FAILURE_MODES + RESERVED_FAILURE_MODES

PRIMARY_DIFFICULTIES = ("L3", "L4")
RESERVED_DIFFICULTIES = ("L1", "L2")
DIFFICULTIES = RESERVED_DIFFICULTIES + PRIMARY_DIFFICULTIES

DIFFICULTY_DEFINITIONS = {
    "L1": "single-hop update",
    "L2": "multi-hop update with distractor",
    "L3": "conditional validity",
    "L4": "cross-scope adversarial audit",
}

DIFFICULTY_ALIASES = {
    "L1_single_hop_update": "L1",
    "L2_multi_hop_with_distractor": "L2",
    "L3_conditional_validity": "L3",
    "L4_cross_scope_adversarial_audit": "L4",
}

PRIMARY_PATTERNS = (
    "closed_as_duplicate_not_fixed",
    "version_scope_leakage",
    "authority_conflict",
    "ci_failed_after_claim",
    "security_policy_override",
    "maintainer_correction_over_user_claim",
    "label_state_mismatch",
    "negative_evidence_required",
)

RESERVED_PATTERNS = (
    "merged_but_unreleased",
    "docs_ahead_of_code",
    "release_then_revert",
    "branch_scope_leakage",
    "backport_only_fix",
    "stale_comment_after_new_release",
    "multi_memory_coupling",
)

PATTERNS = PRIMARY_PATTERNS + RESERVED_PATTERNS

PRIMARY_MEMORY_STATUSES = (
    "current",
    "blocked",
    "unresolved",
    "out_of_scope",
    "should_not_store",
)

RESERVED_MEMORY_STATUSES = (
    "outdated",
    "deleted",
    "restored",
)

MEMORY_STATUSES = PRIMARY_MEMORY_STATUSES + RESERVED_MEMORY_STATUSES


def normalize_difficulty(value: object) -> str:
    """Normalize short and legacy long difficulty labels to L1-L4."""
    text = str(value or "").strip()
    return DIFFICULTY_ALIASES.get(text, text)


def canonical_hidden_gold_fields(gold: dict) -> dict:
    """Read canonical v1.1 hidden_gold fields."""
    return {
        "expected_decision": gold.get("expected_decision"),
        "expected_answer": gold.get("expected_answer"),
        "expected_memory_state": gold.get("expected_memory_state") or {},
        "expected_failure_diagnosis": gold.get("expected_failure_diagnosis"),
        "expected_evidence_event_ids": list(gold.get("expected_evidence_event_ids") or []),
        "counterevidence_event_ids": list(gold.get("counterevidence_event_ids") or []),
        "rubric": gold.get("rubric") or {},
        "decision_aliases": gold.get("decision_aliases"),
        "stale_or_wrong_answers": gold.get("stale_or_wrong_answers") or [],
    }

TASK_TYPES = (
    "black_box_task",
    "memory_state_task",
    "evidence_retrieval_task",
    "diagnostic_task",
)

TRUST_LEVELS = ("verified", "trusted", "untrusted")

# Canonical five-way revision-decision action space. This is the single source
# of truth for decision labels; scorers, generators, and baselines import it
# rather than redefining their own copies.
DECISIONS = (
    "use_current_memory",
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
)

# Decisions whose correct action is NOT to answer from current memory. Used for
# the non_answer_decision_accuracy headline metric.
NON_ANSWER_DECISIONS = (
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
)

PUBLIC_FORBIDDEN_TERMS = (
    "retrace",
    "dpa",
    "authorization court",
    "benchmark",
    "gold label",
    "hidden truth",
    "dataset",
)

# Discriminative, label-faithful definitions for the failure modes. v1.3 uses
# PRIMARY_FAILURE_MODES in gold labels; RESERVED_FAILURE_MODES remain documented
# for API compatibility and future releases.
FAILURE_MODE_DEFINITIONS = {
    "stale_memory_reuse": (
        "A newer valid record supersedes an older memory, but the older memory "
        "would still be used as the current basis."
    ),
    "under_update": (
        "A valid update should propagate to several related memories, but only "
        "part of the memory state would be updated (sibling memories left stale)."
    ),
    "over_update": (
        "A valid update is applied too broadly within the same scope or item "
        "family, changing memories it should not affect."
    ),
    "conflict_collapse": (
        "Two credible in-scope records genuinely conflict and the assistant "
        "collapses them into one unsupported answer instead of preserving the "
        "conflict as unresolved."
    ),
    "scope_leakage": (
        "An out-of-scope workspace/tenant/user record is used as if it applied "
        "in the current scope."
    ),
    "policy_violation": (
        "The proposed memory action would store or restate policy-sensitive "
        "content (e.g. credential-like values) that must not be retained."
    ),
    "wrong_source_attribution": (
        "The answer trusts a forwarded or non-authoritative record over the "
        "system of record, or attributes a fact to the wrong source/provenance."
    ),
    "memory_hallucination": (
        "The assistant invents or accepts a memory claim that has no supporting "
        "visible event."
    ),
    "unnecessary_memory_write": (
        "The assistant creates durable memory when the evidence only supports a "
        "one-shot action or duplicates an existing memory."
    ),
    "failure_to_forget": (
        "An explicit, valid deletion/forget request should remove or stop using "
        "a memory, but the assistant keeps it active."
    ),
    "failure_to_release_or_restore": (
        "A temporary block/hold should be lifted after valid release evidence "
        "(or kept when release evidence is invalid); the assistant mishandles "
        "this lifecycle."
    ),
}
