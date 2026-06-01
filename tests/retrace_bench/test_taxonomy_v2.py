from benchmark.retrace_bench.taxonomy_v2 import (
    DomainV2,
    ReliabilityFailureModeV2,
    TaskProtocolV2,
    ProbeTypeV2,
)


def test_taxonomy_domains():
    expected_domains = {
        "software_engineering_agent",
        "enterprise_multi_tool_workflow",
        "customer_support_crm",
        "calendar_task_workflow",
        "research_knowledge_work",
        "personal_assistant_preference",
        "ecommerce_recommendation",
        "data_analysis_bi",
    }
    actual_domains = {d.value for d in DomainV2}
    assert expected_domains == actual_domains


def test_taxonomy_failure_modes():
    expected_modes = {
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
    actual_modes = {m.value for m in ReliabilityFailureModeV2}
    assert expected_modes == actual_modes


def test_taxonomy_protocols():
    expected_protocols = {
        "black_box_task",
        "memory_state_task",
        "structured_revision_task",
        "oracle_diagnostic_task",
    }
    actual_protocols = {p.value for p in TaskProtocolV2}
    assert expected_protocols == actual_protocols
