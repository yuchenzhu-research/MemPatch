from __future__ import annotations

from mempatch.benchmark.generate import build_scenario
from mempatch.benchmark.leakage import audit_public_rows
from mempatch.benchmark.release import label_row, public_row
from mempatch.benchmark.score import aggregate_scores, score_row


def _raw() -> dict:
    return {
        "scenario_id": "case-1",
        "domain": "software_release",
        "difficulty": "challenge",
        "primary_failure_mode": "scope_leakage",
        "pattern": "cross_scope_distractor",
        "metadata": {"resolver_trace": {"rule_id": "r1"}},
        "workflow_context": "A coding agent maintains persistent memories.",
        "public_input": {
            "initial_memories": [
                {
                    "memory_id": "mem-1",
                    "content": "Use beta config only in workspace B.",
                    "scope": "workspace-b",
                    "memory_type": "core",
                    "user_id": "user-a",
                    "session_id": "session-1",
                    "tags": ["config"],
                    "is_distractor": False,
                }
            ],
            "events": [
                {
                    "event_id": "ev-1",
                    "timestamp_order": 1,
                    "actor_role": "maintainer",
                    "trust_level": "verified",
                    "visibility_scope": "workspace-a",
                    "event_type": "comment",
                    "content": "Workspace A should not use the beta config.",
                    "related_memory_ids": ["mem-1"],
                    "user_id": "user-a",
                    "session_id": "session-2",
                }
            ],
        },
        "black_box_task": {"query": "What should the agent do in workspace A?"},
        "followup_task": {"prompt": "Later, what should the agent say about workspace A?"},
        "hidden_gold": {
            "expected_decision": "mark_unresolved",
            "expected_memory_operation": "RESTRICT_SCOPE",
            "expected_answer": "Do not use the beta config in workspace A.",
            "expected_followup_answer": "Workspace A remains unresolved and should not reuse the beta config.",
            "expected_followup_answer_key_facts": ["Workspace A", "unresolved"],
            "expected_memory_state": {"mem-1": "out_of_scope"},
            "expected_evidence_event_ids": ["ev-1"],
            "expected_failure_diagnosis": "scope_leakage",
            "stale_or_wrong_answers": ["Use beta config in workspace A."],
            "unsafe_reuse_patterns": ["Use beta config in workspace A."],
            "rubric": {"must_include": ["Do not use the beta config"]},
        },
    }


def test_public_row_has_no_gold_or_design_metadata() -> None:
    row = public_row(_raw())
    assert row["scenario_id"] == "case-1"
    assert "difficulty" not in row
    assert row["public_input"]["initial_memories"][0]["memory_type"] == "core"
    assert row["public_input"]["initial_memories"][0]["user_id"] == "user-a"
    assert row["public_input"]["events"][0]["session_id"] == "session-2"
    assert "followup_task" in row["tasks"]
    assert "memory_operation" in row["output_contract"]["required_fields"]
    assert "followup_answer" in row["output_contract"]["required_fields"]
    assert audit_public_rows([row]) == []
    assert "cross_scope_distractor" not in str(row)


def test_label_row_keeps_private_fields_as_list_schema() -> None:
    row = label_row(_raw(), "main_test_synthetic")
    assert row["difficulty"] == "challenge"
    assert row["failure_mode"] == "scope_leakage"
    assert row["expected_memory_operation"] == "RESTRICT_SCOPE"
    assert row["expected_followup_answer"]
    assert row["expected_memory_states"] == [{"memory_id": "mem-1", "status": "out_of_scope"}]


def test_gold_prediction_scores_perfect() -> None:
    label = label_row(_raw(), "main_test_synthetic")
    prediction = {
        "scenario_id": "case-1",
        "method": "gold",
        "model": "oracle",
        "parsed": {
            "answer": "Do not use the beta config in workspace A.",
            "decision": "mark_unresolved",
            "memory_operation": "RESTRICT_SCOPE",
            "memory_state": [{"memory_id": "mem-1", "status": "out_of_scope"}],
            "evidence_event_ids": ["ev-1"],
            "failure_diagnosis": "scope_leakage",
            "followup_answer": "Workspace A remains unresolved and should not reuse the beta config.",
        },
    }
    score = score_row(label, prediction)
    assert score["failure_mode"] == "scope_leakage"
    assert score["difficulty"] == "challenge"
    assert score["schema_valid"]
    assert score["strict_joint"]
    assert score["memory_state_accuracy"] == 1.0
    assert score["evidence_f1"] == 1.0
    assert score["memory_operation_correct"]
    assert score["followup_answer_correct"]
    assert not score["downstream_contamination"]


def test_malformed_prediction_fails_schema() -> None:
    label = label_row(_raw(), "main_test_synthetic")
    score = score_row(label, {"scenario_id": "case-1"})
    assert not score["schema_valid"]
    assert not score["strict_joint"]


def test_aggregate_scores_is_table_source() -> None:
    rows = [
        {
            "scenario_id": "a",
            "failure_mode": "scope_leakage",
            "method": "m",
            "model": "x",
            "schema_valid": True,
            "decision_correct": True,
            "exact_state_map": True,
            "memory_state_accuracy": 1.0,
            "evidence_f1": 1.0,
            "diagnosis_correct": True,
            "strict_joint": True,
            "unsafe_reuse": False,
            "downstream_contamination": False,
        },
        {
            "scenario_id": "b",
            "failure_mode": "scope_leakage",
            "method": "m",
            "model": "x",
            "schema_valid": True,
            "decision_correct": False,
            "exact_state_map": False,
            "memory_state_accuracy": 0.0,
            "evidence_f1": 0.0,
            "diagnosis_correct": False,
            "strict_joint": False,
            "unsafe_reuse": True,
            "downstream_contamination": True,
        },
    ]
    assert aggregate_scores(rows)[0]["strict_joint"] == 0.5
    assert aggregate_scores(rows)[0]["unsafe_reuse"] == 0.5
    assert aggregate_scores(rows, group_by=["failure_mode"])[0]["failure_mode"] == "scope_leakage"


def test_leakage_audit_catches_alias_keys_and_hidden_taxonomy_values() -> None:
    violations = audit_public_rows(
        [
            {
                "scenario_id": "bad",
                "resolverTrace": {"rule": "r"},
                "public_input": {"events": [{"content": "temporal_supersession"}]},
            }
        ]
    )
    assert violations
    assert "$.resolverTrace" in violations[0]["paths"]
    assert "$.public_input.events[0].content" in violations[0]["paths"]


def test_generated_public_row_strips_pattern_tag_values() -> None:
    row = public_row(build_scenario("main_test_synthetic", 0))
    serialized = str(row)
    assert "temporal_supersession" not in serialized
    assert "m_target" not in serialized
    assert "m_distractor" not in serialized
    assert audit_public_rows([row]) == []


def test_generated_synthetic_has_mixed_structure_and_operations() -> None:
    rows = [build_scenario("main_test_synthetic", idx) for idx in range(40)]
    difficulties = {row["difficulty"] for row in rows}
    event_counts = {len(row["public_input"]["events"]) for row in rows}
    memory_counts = {len(row["public_input"]["initial_memories"]) for row in rows}
    operations = {row["hidden_gold"]["expected_memory_operation"] for row in rows}
    assert {"medium", "hard", "challenge"} <= difficulties
    assert len(event_counts) > 2
    assert len(memory_counts) > 2
    assert "NO_WRITE" in operations
