"""Schema + vocabulary-compatibility tests for ReTrace-Learn."""
from __future__ import annotations

import pytest

from retracemem.proposers.typed_revision_policy import CANONICAL_ACTIONS as RUNTIME_ACTIONS
from retracemem.schemas import AuthorizationStatus

from retrace_learn.schemas import (
    CANONICAL_ACTIONS,
    FINAL_STATUSES,
    RevisionAction,
    SchemaValidationError,
    TypedRevisionExample,
)
from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes


def test_action_vocabulary_matches_runtime():
    assert set(CANONICAL_ACTIONS) == set(RUNTIME_ACTIONS)


def test_final_status_vocabulary_matches_runtime():
    assert FINAL_STATUSES == frozenset(s.value for s in AuthorizationStatus)


def test_supersedes_requires_target_and_replacement():
    with pytest.raises(SchemaValidationError):
        RevisionAction(action_type="SUPERSEDES", target_belief_id="b1", evidence_ids=("e1",)).validate()
    with pytest.raises(SchemaValidationError):
        RevisionAction(
            action_type="SUPERSEDES", replacement_belief_id="b2", evidence_ids=("e1",)
        ).validate()
    RevisionAction(
        action_type="SUPERSEDES",
        target_belief_id="b1",
        replacement_belief_id="b2",
        evidence_ids=("e1",),
    ).validate()


def test_blocks_releases_target_condition_only():
    for at in ("BLOCKS", "RELEASES"):
        RevisionAction(action_type=at, target_condition_id="c1", evidence_ids=("e1",)).validate()
        with pytest.raises(SchemaValidationError):
            RevisionAction(action_type=at, target_belief_id="b1", evidence_ids=("e1",)).validate()


def test_uncertain_reaffirms_target_belief_only():
    for at in ("UNCERTAIN", "REAFFIRMS"):
        RevisionAction(action_type=at, target_belief_id="b1", evidence_ids=("e1",)).validate()
        with pytest.raises(SchemaValidationError):
            RevisionAction(action_type=at, target_condition_id="c1", evidence_ids=("e1",)).validate()


def test_no_revision_forbids_targets_and_needs_no_evidence():
    RevisionAction(action_type="NO_REVISION").validate()
    with pytest.raises(SchemaValidationError):
        RevisionAction(action_type="NO_REVISION", target_belief_id="b1").validate()


def test_non_no_revision_requires_evidence():
    with pytest.raises(SchemaValidationError):
        RevisionAction(action_type="REAFFIRMS", target_belief_id="b1").validate()


def test_unknown_action_rejected():
    with pytest.raises(SchemaValidationError):
        RevisionAction(action_type="DELETE", target_belief_id="b1", evidence_ids=("e1",)).validate()


def test_synthetic_examples_validate_and_roundtrip():
    for ep in build_synthetic_episodes():
        graph_ex = ep.to_graph_extraction_example()
        graph_ex.validate()
        rev_ex = ep.to_typed_revision_example()
        rev_ex.validate()
        # JSON round-trip
        again = TypedRevisionExample.from_dict(rev_ex.to_dict())
        again.validate()
        assert again.gold_final_statuses == rev_ex.gold_final_statuses
