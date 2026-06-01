from benchmark.retrace_bench.schemas_v2 import (
    EventV2,
    MemoryEntryV2,
    ScenarioV2,
    ScenarioMetadataV2,
    HiddenMemoryLifecycleV2,
    scenario_to_dict,
    scenario_from_dict,
)
from benchmark.retrace_bench.taxonomy_v2 import (
    DomainV2,
    EventSourceV2,
    EventTypeV2,
    TrustLevelV2,
)


def test_schema_instantiation():
    event = EventV2(
        event_id="e1",
        source=EventSourceV2.USER,
        event_type=EventTypeV2.USER_MESSAGE,
        actor="user",
        timestamp="2026-05-24T08:00:00Z",
        content={"text": "hello"},
        visibility_scope="public",
        trust_level=TrustLevelV2.TRUSTED,
    )
    assert event.event_id == "e1"
    assert event.source == EventSourceV2.USER


def test_scenario_serialization_roundtrip():
    meta = ScenarioMetadataV2(
        schema_version="2.0",
        contamination_policy={"policy": "evaluation_only"}
    )
    scenario = ScenarioV2(
        scenario_id="test-id",
        version="2.0",
        domain=DomainV2.CUSTOMER_SUPPORT_CRM,
        workflow_context="context",
        memory_snapshot=[],
        event_trace=[
            EventV2(
                event_id="e1",
                source=EventSourceV2.USER,
                event_type=EventTypeV2.USER_MESSAGE,
                actor="user",
                timestamp="2026-05-24",
                content={"text": "hi"},
                visibility_scope="public",
                trust_level=TrustLevelV2.TRUSTED,
            )
        ],
        hidden_memory_lifecycle=HiddenMemoryLifecycleV2(operations=[]),
        failure_modes=[],
        lifecycle_operations=[],
        tasks=[],
        metadata=meta,
    )

    s_dict = scenario_to_dict(scenario)
    assert s_dict["scenario_id"] == "test-id"
    assert s_dict["domain"] == "customer_support_crm"
    assert s_dict["event_trace"][0]["event_type"] == "user_message"

    reconstructed = scenario_from_dict(s_dict)
    assert reconstructed.scenario_id == "test-id"
    assert reconstructed.domain == DomainV2.CUSTOMER_SUPPORT_CRM
    assert reconstructed.event_trace[0].event_type == EventTypeV2.USER_MESSAGE
