from __future__ import annotations

import json

from benchmark.general_taxonomy import (
    PRIMARY_FAILURE_MODES,
    PRIMARY_MEMORY_STATUSES,
    RESERVED_FAILURE_MODES,
    normalize_difficulty,
)
from benchmark.model_runner import build_prompt
from benchmark.public_view import public_scenario_view
from scripts.data.validate_mempatch_bench_dataset import validate_one


def _v13_scenario(**overrides: object) -> dict:
    scenario = {
        "scenario_id": "case-taxonomy-001",
        "pattern": "authority_conflict",
        "benchmark_version": "v1.3",
        "public_split_name": "train",
        "domain": "enterprise_multi_tool_workflow",
        "primary_failure_mode": "conflict_collapse",
        "difficulty": "L3",
        "difficulty_level": "L3",
        "workflow_context": "Agent is checking state integrity for CASE-1.",
        "public_input": {
            "initial_memory": [
                {"memory_id": "m1", "text": "Current stable note"},
                {"memory_id": "m2", "text": "Blocked note"},
                {"memory_id": "m3", "text": "Out-of-scope note"},
            ],
            "event_trace": [
                {"event_id": "e1", "text": "Verified source A says value one."},
                {"event_id": "e2", "text": "Verified source B conflicts with source A."},
                {"event_id": "e3", "text": "User asks for final state."},
            ],
        },
        "hidden_gold": {
            "expected_decision": "mark_unresolved",
            "expected_answer": "The evidence is unresolved.",
            "expected_memory_state": {
                "m1": "unresolved",
                "m2": "blocked",
                "m3": "out_of_scope",
            },
            "expected_failure_diagnosis": "conflict_collapse",
            "expected_evidence_event_ids": ["e1", "e2"],
            "counterevidence_event_ids": [],
            "rubric": {"must_include": ["unresolved"]},
            "decision_aliases": {},
            "stale_or_wrong_answers": [],
        },
        "black_box_task": {"prompt": "What is the final state?"},
        "memory_state_task": {"prompt": "Classify memory statuses."},
        "evidence_retrieval_task": {"prompt": "Cite minimal evidence."},
        "diagnostic_task": {"prompt": "Diagnose the memory failure."},
    }
    scenario.update(overrides)
    return scenario


def test_model_prompt_exposes_v13_primary_failure_modes_only() -> None:
    prompt = json.loads(build_prompt(public_scenario_view(_v13_scenario())))
    schema = prompt["required_output_schema"]

    assert schema["failure_diagnosis"] == list(PRIMARY_FAILURE_MODES)
    assert not set(schema["failure_diagnosis"]) & set(RESERVED_FAILURE_MODES)
    assert set(schema["memory_state"]["m1"]) == set(PRIMARY_MEMORY_STATUSES)


def test_validate_one_rejects_reserved_v13_failure_mode_and_mismatched_difficulty() -> None:
    scenario = _v13_scenario(
        primary_failure_mode="over_update",
        difficulty="L3",
        difficulty_level="L4",
    )
    scenario["hidden_gold"]["expected_failure_diagnosis"] = "over_update"

    errors, _warnings = validate_one(scenario)

    assert any("invalid primary_failure_mode" in error for error in errors)
    assert any("not in PRIMARY_FAILURE_MODES" in error for error in errors)
    assert any("does not match difficulty_level" in error for error in errors)


def test_normalize_difficulty_accepts_short_and_legacy_long_labels() -> None:
    assert normalize_difficulty("L3") == "L3"
    assert normalize_difficulty("L4_cross_scope_adversarial_audit") == "L4"
