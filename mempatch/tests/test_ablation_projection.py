from __future__ import annotations

from benchmark.public_view import public_scenario_view
from mempatch.revision.runtime.ablation_projection import project_actions_without_dpa
from mempatch.revision.runtime.dpa_runtime import parse_actions
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view


def test_supersedes_without_replacement_remains_invalid() -> None:
    parsed = parse_actions(
        '[{"action_type":"SUPERSEDES","target_belief_id":"m_target",'
        '"target_condition_id":null,"replacement_belief_id":null,'
        '"evidence_ids":["e1"],"rationale":"replace"}]'
    )

    assert parsed.valid_json
    assert not parsed.schema_valid
    assert parsed.actions == ()
    assert parsed.error_message == "SUPERSEDES requires replacement_belief_id"


def test_no_dpa_projection_maps_block_action_directly() -> None:
    scenario = {
        "scenario_id": "case_ablation",
        "public_input": {
            "initial_memory": [
                {"memory_id": "m_target", "text": "Prior state."},
                {"memory_id": "m_condition", "text": "Condition rule: approval required."},
                {"memory_id": "m_distractor", "text": "Distractor info: beta only."},
            ],
            "event_trace": [
                {"event_id": "e1", "timestamp": "2027-01-01", "text": "Approval blocked."}
            ],
        },
        "black_box_task": {"prompt": "What is usable?"},
    }
    view = build_scenario_revision_view(scenario)
    condition_id = dict(view.candidate_conditions_by_belief)["m_target"][0].condition_id
    parsed = parse_actions(
        '[{"action_type":"BLOCKS","target_belief_id":null,'
        f'"target_condition_id":"{condition_id}","replacement_belief_id":null,'
        '"evidence_ids":["e1"],"rationale":"blocked"}]'
    )

    response = project_actions_without_dpa(
        view=view,
        parse_result=parsed,
        raw_response={"answer": "Escalate.", "failure_diagnosis": "conflict_collapse"},
        scenario_public_view=public_scenario_view(scenario),
    )

    assert response["decision"] == "escalate"
    assert response["memory_state"] == {
        "m_target": "blocked",
        "m_condition": "current",
        "m_distractor": "out_of_scope",
    }
    assert response["evidence_event_ids"] == ["e1"]


def test_no_dpa_projection_ignores_non_scalar_raw_fields() -> None:
    scenario = {
        "scenario_id": "case_malformed_raw_response",
        "public_input": {
            "initial_memory": [
                {"memory_id": "m_target", "text": "Prior state."},
                {"memory_id": "m_distractor", "text": "Distractor info: beta only."},
            ],
            "event_trace": [
                {"event_id": "e1", "timestamp": "2027-01-01", "text": "No change."}
            ],
        },
        "black_box_task": {"prompt": "What is usable?"},
    }
    view = build_scenario_revision_view(scenario)
    parsed = parse_actions(
        '[{"action_type":"NO_REVISION","target_belief_id":null,'
        '"target_condition_id":null,"replacement_belief_id":null,'
        '"evidence_ids":["e1"],"rationale":"no change"}]'
    )

    response = project_actions_without_dpa(
        view=view,
        parse_result=parsed,
        raw_response={
            "memory_state": {
                "m_target": ["should_not_store"],
                "m_distractor": {"status": "current"},
            },
            "failure_diagnosis": ["stale_memory_reuse"],
        },
        scenario_public_view=public_scenario_view(scenario),
    )

    assert response["memory_state"] == {
        "m_target": "current",
        "m_distractor": "out_of_scope",
    }
    assert response["failure_diagnosis"] == "memory_hallucination"


def test_action_parser_fails_closed_on_non_scalar_fields() -> None:
    malformed_payloads = [
        '[{"action_type":["NO_REVISION"],"evidence_ids":["e1"]}]',
        '[{"action_type":"UNCERTAIN","target_belief_id":["m_target"],'
        '"evidence_ids":["e1"]}]',
        '[{"action_type":"NO_REVISION","evidence_ids":[["e1"]]}]',
        '[{"action_type":"NO_REVISION","evidence_ids":["e1"],'
        '"rationale":["no change"]}]',
    ]

    for payload in malformed_payloads:
        parsed = parse_actions(payload)
        assert parsed.valid_json
        assert not parsed.schema_valid
        assert parsed.actions == ()
        assert parsed.error_code == "SCHEMA_CONSTRAINTS_VIOLATED"
