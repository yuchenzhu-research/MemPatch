from __future__ import annotations

import pytest

from benchmark.general_taxonomy import DECISIONS, PRIMARY_FAILURE_MODES, PRIMARY_MEMORY_STATUSES
from scripts.data.prepare_mempatch_v13_smoke import (
    assert_label_coverage,
    gold_to_revision_actions,
    multitask_sft_examples,
)


def _rows_with_full_coverage() -> list[dict]:
    count = max(len(DECISIONS), len(PRIMARY_FAILURE_MODES), len(PRIMARY_MEMORY_STATUSES))
    return [
        {
            "hidden_gold": {
                "expected_decision": DECISIONS[index % len(DECISIONS)],
                "expected_failure_diagnosis": PRIMARY_FAILURE_MODES[index % len(PRIMARY_FAILURE_MODES)],
                "expected_memory_state": {
                    "m1": PRIMARY_MEMORY_STATUSES[index % len(PRIMARY_MEMORY_STATUSES)]
                },
            }
        }
        for index in range(count)
    ]


def test_assert_label_coverage_accepts_complete_partition() -> None:
    assert_label_coverage(_rows_with_full_coverage(), split_name="train")


def test_assert_label_coverage_rejects_missing_targets() -> None:
    with pytest.raises(ValueError, match="missing required labels"):
        assert_label_coverage(_rows_with_full_coverage()[:1], split_name="val")


def _scenario(*, decision: str, target_status: str) -> dict:
    return {
        "scenario_id": f"case_{decision}",
        "public_input": {
            "initial_memory": [
                {
                    "memory_id": "m_target",
                    "text": "Prior state: stable configuration.",
                    "source_event_ids": ["e_init"],
                },
                {
                    "memory_id": "m_condition",
                    "text": "Condition rule: verified approval is required.",
                    "source_event_ids": ["e_init"],
                },
                {
                    "memory_id": "m_distractor",
                    "text": "Distractor info: beta configuration.",
                    "source_event_ids": ["e_beta"],
                },
            ],
            "event_trace": [
                {
                    "event_id": "e1",
                    "timestamp": "2027-01-01T00:00:00Z",
                    "text": "Verified stable evidence.",
                },
                {
                    "event_id": "e2",
                    "timestamp": "2027-01-02T00:00:00Z",
                    "text": "Background event.",
                },
            ],
        },
        "hidden_gold": {
            "expected_decision": decision,
            "expected_answer": "Expected answer.",
            "expected_memory_state": {
                "m_target": target_status,
                "m_condition": "current",
                "m_distractor": "out_of_scope",
            },
            "expected_evidence_event_ids": ["e1"],
            "expected_failure_diagnosis": "conflict_collapse",
        },
        "black_box_task": {"prompt": "What is the authorized state?"},
    }


@pytest.mark.parametrize(
    ("decision", "target_status", "expected_action"),
    [
        ("use_current_memory", "current", "REAFFIRMS"),
        ("escalate", "blocked", "BLOCKS"),
        ("ask_clarification", "blocked", "BLOCKS"),
        ("mark_unresolved", "unresolved", "UNCERTAIN"),
        ("refuse_due_to_policy", "should_not_store", "NO_REVISION"),
    ],
)
def test_gold_to_revision_actions(
    decision: str,
    target_status: str,
    expected_action: str,
) -> None:
    actions = gold_to_revision_actions(
        _scenario(decision=decision, target_status=target_status)
    )

    assert [action.action_type for action in actions] == [expected_action]
    assert actions[0].evidence_ids == ("e1",)


def test_multitask_sft_keeps_two_separate_targets() -> None:
    rows = multitask_sft_examples(
        _scenario(decision="escalate", target_status="blocked")
    )

    assert [row["task_type"] for row in rows] == [
        "path_b_response",
        "path_a_typed_actions",
    ]
