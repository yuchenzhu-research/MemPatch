from __future__ import annotations

import pytest
from collections import Counter
from experiments.multiagent.run_stagec_adapter_eval import (
    extract_first_json_array,
    normalize_actions,
    get_target_ids,
    get_evidence_ids,
)


def test_extract_think_followed_by_json():
    text = """
    <think>
    Thinking about the revision policy...
    We need to block c1.
    </think>
    [
      {
        "action_type": "BLOCKS",
        "target_condition_id": "c1",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    parsed = extract_first_json_array(text)
    assert len(parsed) == 1
    assert parsed[0]["action_type"] == "BLOCKS"
    assert parsed[0]["target_condition_id"] == "c1"


def test_extract_prose_before_json():
    text = """
    Here is the requested JSON format action list:
    [
      {
        "action_type": "SUPERSEDES",
        "target_belief_id": "b1",
        "replacement_belief_id": "b2",
        "evidence_ids": ["ev_2"]
      }
    ]
    Hope this helps!
    """
    parsed = extract_first_json_array(text)
    assert len(parsed) == 1
    assert parsed[0]["action_type"] == "SUPERSEDES"
    assert parsed[0]["target_belief_id"] == "b1"


def test_extract_invalid_json():
    text_unclosed = "[ { 'action_type': 'BLOCKS' "
    with pytest.raises(ValueError, match="No JSON array end"):
        extract_first_json_array(text_unclosed)

    text_no_bracket = "hello world"
    with pytest.raises(ValueError, match="No JSON array start"):
        extract_first_json_array(text_no_bracket)


def test_missing_action_in_multiaction():
    # Gold has two actions: SUPERSEDES and BLOCKS
    gold_actions = [
        {"action_type": "SUPERSEDES", "target_belief_id": "b1", "replacement_belief_id": "b2", "evidence_ids": ["ev_1"]},
        {"action_type": "BLOCKS", "target_condition_id": "c1", "evidence_ids": ["ev_1"]},
    ]

    # Prediction has only SUPERSEDES
    pred_actions = [
        {"action_type": "SUPERSEDES", "target_belief_id": "b1", "replacement_belief_id": "b2", "evidence_ids": ["ev_1"]},
    ]

    gold_types = [a["action_type"] for a in gold_actions]
    pred_types = [a["action_type"] for a in pred_actions]

    # action_type_match should fail because BLOCKS is missing
    action_type_match = (Counter(gold_types) == Counter(pred_types))
    assert action_type_match is False

    # exact_match should fail
    exact_match = (normalize_actions(gold_actions) == normalize_actions(pred_actions))
    assert exact_match is False


def test_grounding_mismatch_correct_action_type():
    # Gold targets condition c1
    gold_actions = [
        {"action_type": "BLOCKS", "target_condition_id": "c1", "evidence_ids": ["ev_1"]},
    ]

    # Prediction targets condition c2 (correct action type BLOCKS but wrong grounding)
    pred_actions = [
        {"action_type": "BLOCKS", "target_condition_id": "c2", "evidence_ids": ["ev_1"]},
    ]

    # action_type_match should succeed (both BLOCKS)
    gold_types = [a["action_type"] for a in gold_actions]
    pred_types = [a["action_type"] for a in pred_actions]
    action_type_match = (Counter(gold_types) == Counter(pred_types))
    assert action_type_match is True

    # target_grounding should fail (c1 vs c2)
    gold_targets = get_target_ids(gold_actions)
    pred_targets = get_target_ids(pred_actions)
    target_grounding = (Counter(gold_targets) == Counter(pred_targets))
    assert target_grounding is False

    # exact_match should fail
    exact_match = (normalize_actions(gold_actions) == normalize_actions(pred_actions))
    assert exact_match is False
