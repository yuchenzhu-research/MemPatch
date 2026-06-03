import pytest
from benchmark.retrace_bench.schemas_v2 import (
    EventV2,
    MemoryEntryV2,
    ScenarioV2,
    ScenarioMetadataV2,
    HiddenMemoryLifecycleV2,
    TaskV2,
    GoldBehaviorV2,
)
from benchmark.retrace_bench.taxonomy_v2 import (
    DomainV2,
    EventSourceV2,
    EventTypeV2,
    TrustLevelV2,
    TaskProtocolV2,
    ProbeTypeV2,
    MemoryStatusV2,
)
from benchmark.retrace_bench.validation_v2 import validate_scenario_v2


def get_base_scenario():
    meta = ScenarioMetadataV2(
        schema_version="2.0",
        contamination_policy={"policy": "evaluation_only"}
    )
    return ScenarioV2(
        scenario_id="sc-01",
        version="2.0",
        domain=DomainV2.CUSTOMER_SUPPORT_CRM,
        workflow_context="ctx",
        memory_snapshot=[
            MemoryEntryV2(
                memory_id="m0",
                category="belief",
                content="test",
                source_event_ids=["e1"],
                status=MemoryStatusV2.AUTHORIZED,
            )
        ],
        event_trace=[
            EventV2(
                event_id="e1",
                source=EventSourceV2.USER,
                event_type=EventTypeV2.USER_MESSAGE,
                actor="user",
                timestamp="2026-05-24",
                content={"text": "hello"},
                visibility_scope="public",
                trust_level=TrustLevelV2.TRUSTED,
            )
        ],
        hidden_memory_lifecycle=HiddenMemoryLifecycleV2(operations=[]),
        failure_modes=[],
        lifecycle_operations=[],
        tasks=[
            TaskV2(
                task_id="t1",
                protocol=TaskProtocolV2.BLACK_BOX_TASK,
                probe_type=ProbeTypeV2.STATE_RESOLUTION,
                prompt="question",
                reference_event_ids=["e1"],
                gold_behavior=GoldBehaviorV2(answer="test"),
            )
        ],
        metadata=meta,
    )


def test_valid_scenario():
    scenario = get_base_scenario()
    validate_scenario_v2(scenario)  # Should not raise error


def test_duplicate_events():
    scenario = get_base_scenario()
    # Add a duplicate event_id
    scenario.event_trace.append(
        EventV2(
            event_id="e1",
            source=EventSourceV2.AGENT,
            event_type=EventTypeV2.AGENT_MESSAGE,
            actor="agent",
            timestamp="2026-05-24",
            content={"text": "hi again"},
        )
    )
    with pytest.raises(ValueError, match="Duplicate event_id 'e1'"):
        validate_scenario_v2(scenario)


def test_invalid_task_reference():
    scenario = get_base_scenario()
    scenario.tasks[0].reference_event_ids.append("e99")  # Non-existent event
    with pytest.raises(ValueError, match="references non-existent event_id 'e99'"):
        validate_scenario_v2(scenario)


def test_invalid_contamination_policy():
    scenario = get_base_scenario()
    scenario.metadata.contamination_policy = {"policy": "fine_tune"}  # Violates policy
    with pytest.raises(ValueError, match="Contamination policy metadata must specify 'evaluation_only'"):
        validate_scenario_v2(scenario)
