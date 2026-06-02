"""Neutral taxonomy for the general English ReTrace-Bench release."""

DOMAINS = (
    "software_engineering_agent",
    "enterprise_multi_tool_workflow",
    "customer_support_crm",
    "calendar_task_workflow",
    "research_knowledge_work",
    "personal_assistant_preference",
    "ecommerce_recommendation",
    "data_analysis_bi",
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

DIFFICULTIES = (
    "L1_single_hop_update",
    "L2_multi_hop_with_distractor",
    "L3_conditional_validity",
    "L4_cross_scope_adversarial_audit",
)

MEMORY_STATUSES = (
    "current",
    "outdated",
    "blocked",
    "unresolved",
    "out_of_scope",
    "deleted",
    "should_not_store",
    "restored",
)

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

# Discriminative, label-faithful definitions for the 11 failure modes. These are
# general (no per-scenario gold) and are intended to be shown to evaluation
# participants and baselines so the diagnostic enum is interpretable. The keys
# are exactly FAILURE_MODES; mechanisms are written to distinguish commonly
# confused pairs (under_update vs stale_memory_reuse, over_update vs
# scope_leakage, failure_to_release_or_restore vs conflict_collapse, etc.).
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

