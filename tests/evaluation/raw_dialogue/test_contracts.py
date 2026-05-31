import pytest

from retracemem.evaluation.raw_dialogue.contracts import (
    DialogueExtractionTarget,
    RawDialogue,
    RawDialogueUtterance,
    RawDialogueValidationError,
)


def _build_valid_target_dict():
    return {
        "example_id": "test_ex_1",
        "subagent_roles": ["planner", "critic"],
        "dialogue": {
            "utterances": [
                {
                    "speaker": "planner",
                    "text": "User is located at Office A.",
                    "timestamp": "2026-06-01T00:00:00Z",
                },
                {
                    "speaker": "critic",
                    "text": "Confirmed, but lease Seattle is required.",
                    "timestamp": "2026-06-01T00:01:00Z",
                },
            ]
        },
        "gold_graph": {
            "evidence_nodes": [
                {
                    "evidence_id": "ev_1",
                    "session_id": "sess_1",
                    "timestamp": "2026-06-01T00:00:00Z",
                    "text": "User is located at Seattle",
                    "source_dataset": "raw",
                    "source_pointer": "p1",
                }
            ],
            "belief_nodes": [
                {
                    "belief_id": "b_1",
                    "proposition": "User Seattle office location",
                    "source_evidence_ids": ["ev_1"],
                }
            ],
            "condition_nodes": [
                {
                    "condition_id": "c_1",
                    "scope_id": "lease",
                    "text": "Seattle lease is active",
                }
            ],
            "candidate_replacement_beliefs": [],
            "dependency_edges": [
                {
                    "edge_id": "dep_1",
                    "belief_id": "b_1",
                    "condition_id": "c_1",
                    "inducer": "inducer_1",
                    "edge_type": "REQUIRES",
                }
            ],
        },
    }


def test_valid_target_roundtrip():
    d = _build_valid_target_dict()
    target = DialogueExtractionTarget.from_dict(d)
    target.validate()

    serialized = target.to_dict()
    assert serialized["example_id"] == "test_ex_1"
    assert len(serialized["dialogue"]["utterances"]) == 2
    assert serialized["dialogue"]["utterances"][0]["speaker"] == "planner"
    assert serialized["gold_graph"]["belief_nodes"][0]["belief_id"] == "b_1"


def test_missing_required_fields():
    with pytest.raises(RawDialogueValidationError):
        DialogueExtractionTarget.from_dict({})


def test_invalid_utterance():
    with pytest.raises(RawDialogueValidationError):
        RawDialogueUtterance.from_dict({"speaker": "only"})


def test_missing_graph_keys():
    d = _build_valid_target_dict()
    del d["gold_graph"]["evidence_nodes"]
    target = DialogueExtractionTarget.from_dict(d)
    with pytest.raises(RawDialogueValidationError, match="missing required key 'evidence_nodes'"):
        target.validate()


def test_unknown_evidence_reference():
    d = _build_valid_target_dict()
    # Change source_evidence_ids to point to a non-existent evidence node
    d["gold_graph"]["belief_nodes"][0]["source_evidence_ids"] = ["ev_ghost"]
    target = DialogueExtractionTarget.from_dict(d)
    with pytest.raises(RawDialogueValidationError, match="references unknown evidence 'ev_ghost'"):
        target.validate()


def test_unknown_belief_reference_in_dependency():
    d = _build_valid_target_dict()
    d["gold_graph"]["dependency_edges"][0]["belief_id"] = "b_ghost"
    target = DialogueExtractionTarget.from_dict(d)
    with pytest.raises(RawDialogueValidationError, match="references unknown belief 'b_ghost'"):
        target.validate()


def test_invalid_dependency_edge_type():
    d = _build_valid_target_dict()
    d["gold_graph"]["dependency_edges"][0]["edge_type"] = "INVALID"
    target = DialogueExtractionTarget.from_dict(d)
    with pytest.raises(RawDialogueValidationError, match="only support edge_type REQUIRES"):
        target.validate()
