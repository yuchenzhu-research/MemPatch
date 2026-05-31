"""Tests for the Stage C API-ICL exemplar contract (fail-closed, anti-leakage)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from retracemem.evaluation.multiagent.stage_c_icl import (
    ExemplarApprovalError,
    ExemplarLeakageError,
    ExemplarSchemaError,
    format_exemplars_for_prompt,
    load_approved_exemplars,
    select_icl_exemplars,
    validate_exemplar,
    validate_pack_approval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _good_exemplar(eid="ex1", category="missed_SUPERSEDES"):
    return {
        "exemplar_id": eid,
        "failure_category": category,
        "candidate_view_summary": "belief b1 requires c1; new evidence supersedes it",
        "submission_evidence": [{"evidence_id": "ev2", "content": "migration note"}],
        "proposed_actions": [{
            "action_type": "SUPERSEDES",
            "target_belief_id": "b1",
            "replacement_belief_id": "b2",
            "target_condition_id": None,
            "evidence_ids": ["ev2"],
            "rationale": "ev2 supersedes b1",
        }],
    }


def _approved_pack(exemplars=None):
    return {
        "pack_id": "p1",
        "split": "development_only",
        "source_manifest_sha256": "deadbeef" * 8,
        "approval": {"decision": "approved", "reviewer": "alice", "reviewed_at": "2026-05-31T00:00:00Z"},
        "exemplars": exemplars or [_good_exemplar()],
    }


# --------------------------------------------------------------------------- #
# Approval gating (fail-closed)
# --------------------------------------------------------------------------- #
def test_pending_pack_fails_closed():
    pack = _approved_pack()
    pack["approval"]["decision"] = "pending"
    with pytest.raises(ExemplarApprovalError):
        validate_pack_approval(pack)


def test_missing_manifest_hash_fails_closed():
    pack = _approved_pack()
    del pack["source_manifest_sha256"]
    with pytest.raises(ExemplarApprovalError):
        validate_pack_approval(pack)


def test_missing_reviewer_fails_closed():
    pack = _approved_pack()
    pack["approval"]["reviewer"] = None
    with pytest.raises(ExemplarApprovalError):
        validate_pack_approval(pack)


def test_approved_pack_passes():
    validate_pack_approval(_approved_pack())  # no raise


def test_shipped_example_template_is_pending_and_fails_closed():
    # The committed template must never be loadable as approved data.
    path = REPO_ROOT / "fixtures/stage_c_exemplars/EXAMPLE_pending_template.json"
    assert path.exists()
    with pytest.raises(ExemplarApprovalError):
        load_approved_exemplars(path)


def test_missing_pack_file_fails_closed(tmp_path):
    with pytest.raises(ExemplarApprovalError):
        load_approved_exemplars(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# Anti-leakage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leak_field", [
    "gold_final_status", "final_belief_statuses", "gold_typed_targets",
    "evaluator_status", "relevant_session_index", "M_old",
])
def test_leakage_fields_rejected(leak_field):
    ex = _good_exemplar()
    ex[leak_field] = "LEAK"
    with pytest.raises(ExemplarLeakageError):
        validate_exemplar(ex)


def test_nested_leakage_rejected():
    ex = _good_exemplar()
    ex["proposed_actions"][0]["gold_final_status"] = "AUTHORIZED"
    with pytest.raises(ExemplarLeakageError):
        validate_exemplar(ex)


# --------------------------------------------------------------------------- #
# Schema / grounding
# --------------------------------------------------------------------------- #
def test_action_must_cite_evidence():
    ex = _good_exemplar()
    ex["proposed_actions"][0]["evidence_ids"] = []
    with pytest.raises(ExemplarSchemaError):
        validate_exemplar(ex)


def test_no_revision_need_not_cite_evidence():
    ex = _good_exemplar()
    ex["proposed_actions"] = [{
        "action_type": "NO_REVISION", "target_belief_id": None,
        "replacement_belief_id": None, "target_condition_id": None,
        "evidence_ids": [], "rationale": "no conflict",
    }]
    validate_exemplar(ex)  # no raise


def test_non_canonical_action_rejected():
    ex = _good_exemplar()
    ex["proposed_actions"][0]["action_type"] = "AUTHORIZED"  # a status, not an action
    with pytest.raises(ExemplarSchemaError):
        validate_exemplar(ex)


def test_missing_required_field_rejected():
    ex = _good_exemplar()
    del ex["candidate_view_summary"]
    with pytest.raises(ExemplarSchemaError):
        validate_exemplar(ex)


# --------------------------------------------------------------------------- #
# Load + select + format
# --------------------------------------------------------------------------- #
def test_load_approved_roundtrip(tmp_path):
    pack = _approved_pack([_good_exemplar("e1", "missed_SUPERSEDES"),
                           _good_exemplar("e2", "over_update")])
    p = tmp_path / "pack.json"
    p.write_text(json.dumps(pack))
    exemplars = load_approved_exemplars(p)
    assert {e["exemplar_id"] for e in exemplars} == {"e1", "e2"}


def test_select_is_deterministic_and_diverse():
    exemplars = [
        _good_exemplar("e1", "missed_SUPERSEDES"),
        _good_exemplar("e2", "missed_SUPERSEDES"),
        _good_exemplar("e3", "over_update"),
    ]
    first = select_icl_exemplars(exemplars, 2)
    second = select_icl_exemplars(exemplars, 2)
    assert [e["exemplar_id"] for e in first] == [e["exemplar_id"] for e in second]
    # diversity: one per category before filling
    assert {e["failure_category"] for e in first} == {"missed_SUPERSEDES", "over_update"}


def test_format_for_prompt_has_no_gold_and_shows_actions():
    text = format_exemplars_for_prompt([_good_exemplar()])
    assert "SUPERSEDES" in text
    assert "AUTHORIZED" not in text  # no final-status vocabulary leaks in
