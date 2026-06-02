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

