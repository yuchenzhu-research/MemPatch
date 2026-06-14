from __future__ import annotations

import json

from mempatch.revision.runtime.revision_module import run_revision_module_on_scenario
from mempatch.revision.runtime.learned_proposer import build_proposer_prompt
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view


def _replacement_scenario() -> dict:
    return {
        "scenario_id": "case_replacement_1",
        "workflow_context": "Track deployment status.",
        "public_input": {
            "initial_memory": [
                {
                    "memory_id": "m_target",
                    "text": "Prior state: service uses default config.",
                    "source_event_ids": ["e_init"],
                }
            ],
            "event_trace": [
                {"event_id": "e_init", "text": "Initial config recorded."},
                {
                    "event_id": "e_update",
                    "text": "README updated: service now supports batch deletes.",
                    "related_memory_ids": ["m_target"],
                },
            ],
        },
        "black_box_task": {"prompt": "What is the current service config?"},
    }


def _condition_scenario() -> dict:
    return {
        "scenario_id": "case_condition_1",
        "public_input": {
            "initial_memory": [
                {
                    "memory_id": "m_target",
                    "text": "Prior state: CASE-42 uses default route configuration.",
                    "source_event_ids": ["e_init"],
                },
                {
                    "memory_id": "m_condition",
                    "text": "Condition rule: Any update to CASE-42 requires verified release approval.",
                    "source_event_ids": ["e_init"],
                },
            ],
            "event_trace": [
                {"event_id": "e_init", "text": "Initial state recorded."},
                {"event_id": "e_latest", "text": "Latest audit note recorded."},
            ],
        },
        "diagnostic_task": {"prompt": "Is the target memory currently usable?"},
    }


def test_related_update_event_extracts_replacement_candidates() -> None:
    view = build_scenario_revision_view(_replacement_scenario())

    assert view.candidate_replacement_beliefs
    replacement = view.candidate_replacement_beliefs[0]
    assert replacement.belief_id == "m_target__replacement__e_update"
    assert "now supports batch deletes" in replacement.proposition
    assert replacement.source_evidence_ids == ("e_update",)


def test_condition_rule_memory_extracts_condition_nodes() -> None:
    view = build_scenario_revision_view(_condition_scenario())

    conditions = dict(view.candidate_conditions_by_belief)
    assert "m_target" in conditions
    assert conditions["m_target"][0].condition_id == "m_condition__cond__0"
    assert conditions["m_target"][0].text.startswith("Condition rule:")


def test_condition_memory_generates_requires_dependency() -> None:
    view = build_scenario_revision_view(_condition_scenario())

    deps = dict(view.dependency_edges_by_belief)
    assert "m_target" in deps
    edge = deps["m_target"][0]
    assert edge.edge_type == "REQUIRES"
    assert edge.belief_id == "m_target"
    assert edge.condition_id == "m_condition__cond__0"


def test_hidden_gold_does_not_change_revision_view() -> None:
    base = _condition_scenario()
    poisoned = {
        **base,
        "hidden_gold": {
            "expected_decision": "mark_unresolved",
            "expected_answer": "secret answer",
            "expected_memory_state": {"m_target": "blocked"},
            "expected_failure_diagnosis": "conflict_collapse",
            "expected_evidence_event_ids": ["e_latest"],
        },
    }

    view_without = build_scenario_revision_view(base)
    view_with = build_scenario_revision_view(poisoned)

    assert view_without.view_fingerprint == view_with.view_fingerprint
    assert view_with.query == view_without.query
    assert len(view_with.candidate_beliefs) == len(view_without.candidate_beliefs)


def test_noop_revision_runner_still_works_with_enriched_view() -> None:
    prediction = run_revision_module_on_scenario(_condition_scenario())
    response = prediction["response"]

    assert prediction["scenario_id"] == "case_condition_1"
    assert set(response.keys()) == {
        "answer",
        "decision",
        "memory_state",
        "evidence_event_ids",
        "failure_diagnosis",
    }
    assert response["memory_state"]["m_target"] == "current"
    assert response["evidence_event_ids"] == ["e_latest"]


def test_realistic_bench_scenario_has_conditions_and_optional_replacements() -> None:
    scenario = {
        "scenario_id": "case-000003",
        "public_input": {
            "initial_memory": [
                {
                    "memory_id": "m-case-000003-target",
                    "text": "Prior state: CASE-300002 uses default case route configuration on stable v1.",
                    "source_event_ids": ["e-init"],
                },
                {
                    "memory_id": "m-case-000003-condition",
                    "text": "Condition rule: Any update to CASE-300002 requires verified release approval.",
                    "source_event_ids": ["e-init"],
                },
            ],
            "event_trace": [
                {
                    "event_id": "e-case-000003-1",
                    "text": "README docs updated: CASE-300002 now supports batch deletes.",
                    "related_memory_ids": ["m-case-000003-target"],
                },
                {"event_id": "e-case-000003-2", "text": "Follow-up note."},
            ],
        },
        "black_box_task": {"prompt": "What is the current route configuration?"},
    }
    view = build_scenario_revision_view(scenario)

    assert dict(view.candidate_conditions_by_belief)
    assert dict(view.dependency_edges_by_belief)
    assert any(
        belief.belief_id == "m-case-000003-target__replacement__e-case-000003-1"
        for belief in view.candidate_replacement_beliefs
    )
    payload = json.dumps(
        {
            "conditions": [
                (bid, [c.condition_id for c in conds])
                for bid, conds in view.candidate_conditions_by_belief
            ],
            "replacements": [b.belief_id for b in view.candidate_replacement_beliefs],
        }
    )
    assert "hidden_gold" not in payload
    assert "expected_" not in payload


def test_proposer_prompt_contains_full_public_evidence_ledger() -> None:
    view = build_scenario_revision_view(_condition_scenario())
    prompt = build_proposer_prompt(view)

    assert '"evidence_id": "e_init"' in prompt
    assert '"evidence_id": "e_latest"' in prompt
    assert '"as_of_evidence_id": "e_latest"' in prompt
    assert "Use BLOCKS, UNCERTAIN, REAFFIRMS, or NO_REVISION" in prompt
    assert "Do not emit SUPERSEDES unless candidate_replacement_beliefs contains" in prompt
    assert "Do not emit RELEASES unless the view contains an explicit release target" in prompt
