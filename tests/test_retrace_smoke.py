"""Consolidated ReTrace smoke tests (Phase 4).

Fast, offline, no-API checks over the canonical Paper 1 surfaces:
the typed-action parser, RevisionGate target rules, the deterministic DPA
precedence as observed through the public ``commit_submission_sequence``
wrapper, and Stage B (DirectJudge) canonicalization.

Each test maps to one of the 10 required smoke items; the item number is given
in the test name/docstring. These intentionally use only public/Stage surfaces
and hand-authored fixtures (no gold leakage, no live calls).
"""
from __future__ import annotations

import json

import pytest

from retracemem.authorization import EvidenceProposalBatch
from retracemem.evaluation.multiagent.directjudge import parse_direct_judge_response
from retracemem.multiagent.commit import commit_submission_sequence
from retracemem.multiagent.contracts import SubagentMemorySubmission
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)
from retracemem.verifier.typed_edge_response_parser import (
    EdgeTargetSpace,
    parse_typed_edge_response,
)


def _evidence(eid: str = "ev_new", ts: str = "2026-01-01T00:00:00Z") -> EvidenceNode:
    return EvidenceNode(
        evidence_id=eid, session_id="s1", timestamp=ts,
        text="new evidence", source_dataset="smoke", source_pointer="ptr",
    )


def _target_space() -> EdgeTargetSpace:
    repl = BeliefNode(belief_id="b_rep", proposition="replacement", source_evidence_ids=("ev_new",))
    return EdgeTargetSpace(
        valid_belief_ids=frozenset({"b1"}),
        valid_condition_ids=frozenset({"c1"}),
        replacement_map={"b_rep": repl},
    )


# --- Item 1: parser accepts valid typed action JSON ------------------------- #
def test_item1_parser_accepts_valid_typed_actions():
    edges = parse_typed_edge_response(
        json.dumps({"edges": [{"edge_type": "REAFFIRMS", "target_id": "b1", "rationale": "x"}]}),
        new_evidence=_evidence(), target_space=_target_space(),
        call_id="c", prompt_version="v1",
    )
    assert len(edges) == 1 and edges[0].edge_type == EvidenceEdgeType.REAFFIRMS


# --- Item 2: parser rejects a final DPA status used as an action ------------ #
@pytest.mark.parametrize("final_status", ["AUTHORIZED", "BLOCKED", "SUPERSEDED", "UNRESOLVED"])
def test_item2_parser_rejects_final_status_as_action(final_status):
    with pytest.raises(ValueError, match="[Uu]nknown edge_type"):
        parse_typed_edge_response(
            json.dumps({"edges": [{"edge_type": final_status, "target_id": "b1"}]}),
            new_evidence=_evidence(), target_space=_target_space(),
            call_id="c", prompt_version="v1",
        )


# --- Item 3: RevisionGate/parser rejects invalid target ids ----------------- #
def test_item3_rejects_invalid_target_ids():
    with pytest.raises(ValueError, match="unknown condition"):
        parse_typed_edge_response(
            json.dumps({"edges": [{"edge_type": "BLOCKS", "target_id": "c_bad"}]}),
            new_evidence=_evidence(), target_space=_target_space(),
            call_id="c", prompt_version="v1",
        )


# --- Item 4: missing new_evidence_id is rejected at commit boundary --------- #
def test_item4_missing_new_evidence_id_rejected():
    from retracemem.multiagent.commit import commit_subagent_submission

    sub = SubagentMemorySubmission(
        submission_id="s", producer_id="a", producer_role="r",
        parent_snapshot_id="snap_0", observed_at="2026-01-01T00:00:00Z",
        instance_id="i", query_id="q", query="Q",
        evidence_context=(_evidence("e1"),), new_evidence_id="e_missing",
        candidate_beliefs=(BeliefNode(belief_id="b1", proposition="p", source_evidence_ids=("e1",)),),
    )
    with pytest.raises(ValueError, match="must match exactly one"):
        commit_subagent_submission(sub)


# --- Item 5: NO_REVISION cannot be mixed with revision actions -------------- #
def test_item5_no_revision_cannot_mix_with_actions():
    from retracemem.evaluation.multiagent.contracts import FixedCandidateSubmission
    from retracemem.proposers.typed_revision_policy import PromptTypedRevisionPolicy

    policy = PromptTypedRevisionPolicy()
    sub = FixedCandidateSubmission(
        submission_id="s", producer_id="a", producer_role="r", task_id=None,
        parent_snapshot_id="snap_0", observed_at="2026-01-01T00:00:00Z",
        instance_id="i", query_id="q", query="Q",
        evidence_context=(_evidence("ev_new"),), new_evidence_id="ev_new",
        candidate_beliefs=(BeliefNode(belief_id="b1", proposition="p", source_evidence_ids=("ev_new",)),),
    )
    mixed = json.dumps([
        {"action_type": "REAFFIRMS", "target_belief_id": "b1", "evidence_ids": ["ev_new"]},
        {"action_type": "NO_REVISION", "target_belief_id": None, "target_condition_id": None,
         "replacement_belief_id": None, "evidence_ids": ["ev_new"]},
    ])
    # parse_response fails closed: it records the violation rather than admitting edges.
    out = policy.parse_response(mixed, example_id="ex1", submission=sub)
    assert out.parsing_valid is False
    assert any("NO_REVISION cannot be combined" in e for e in out.errors)
    assert out.proposal_batches == ()


# --- Item 6: BLOCKS / RELEASES only target conditions ----------------------- #
@pytest.mark.parametrize("etype", ["BLOCKS", "RELEASES"])
def test_item6_block_release_only_target_conditions(etype):
    # Targeting a belief id with a condition-only edge type is rejected.
    with pytest.raises(ValueError, match="unknown condition"):
        parse_typed_edge_response(
            json.dumps({"edges": [{"edge_type": etype, "target_id": "b1"}]}),
            new_evidence=_evidence(), target_space=_target_space(),
            call_id="c", prompt_version="v1",
        )


# --- Item 7: SUPERSEDES requires target belief + grounded replacement ------- #
def test_item7_supersedes_requires_replacement():
    with pytest.raises(ValueError, match="replacement_belief_id"):
        parse_typed_edge_response(
            json.dumps({"edges": [{"edge_type": "SUPERSEDES", "target_id": "b1"}]}),
            new_evidence=_evidence(), target_space=_target_space(),
            call_id="c", prompt_version="v1",
        )


def _supersede_sequence() -> tuple[SubagentMemorySubmission, ...]:
    ev1 = _evidence("e1", "2026-01-01T10:00:00Z")
    ev2 = _evidence("e2", "2026-01-01T11:00:00Z")
    b_old = BeliefNode(belief_id="b_old", proposition="old", source_evidence_ids=("e1",))
    b_new = BeliefNode(belief_id="b_new", proposition="new", source_evidence_ids=("e2",))
    edge = EvidenceEdge(
        edge_id="edge_sup", edge_type=EvidenceEdgeType.SUPERSEDES, evidence_id="e2",
        target_kind="belief", target_id="b_old", verifier="smoke", replacement_belief_id="b_new",
    )
    sub1 = SubagentMemorySubmission(
        submission_id="sub_1", producer_id="a", producer_role="r",
        parent_snapshot_id="snap_0", observed_at="2026-01-01T10:00:00Z",
        instance_id="i", query_id="q", query="Q",
        evidence_context=(ev1,), new_evidence_id="e1", candidate_beliefs=(b_old,),
    )
    sub2 = SubagentMemorySubmission(
        submission_id="sub_2", producer_id="a", producer_role="r",
        parent_snapshot_id="snap_1", observed_at="2026-01-01T11:00:00Z",
        instance_id="i", query_id="q", query="Q",
        evidence_context=(ev1, ev2), new_evidence_id="e2",
        candidate_beliefs=(), candidate_replacement_beliefs=(b_new,),
        proposal_batches=(EvidenceProposalBatch(edges=(edge,)),),
    )
    return (sub1, sub2)


# --- Item 8: commit_submission_sequence emits gate_decisions trace ---------- #
def test_item8_commit_sequence_emits_gate_decisions_trace():
    res = commit_submission_sequence(_supersede_sequence(), final_snapshot_evaluation=True)
    assert res.trace["number_of_submissions"] == 2
    gd = res.trace["gate_decisions"]
    assert any(d["edge_type"] == "SUPERSEDES" and d["admitted"] for d in gd)
    # Per-submission authorization traces are preserved for audit.
    assert all(r.commit_trace.get("auth_trace") is not None for r in res.submission_results)


# --- Item 9: DPA precedence (SUPERSEDED, BLOCKED over AUTHORIZED) ------------ #
def test_item9_dpa_status_precedence():
    # SUPERSEDES path: superseded belief excluded, replacement authorized.
    res = commit_submission_sequence(_supersede_sequence(), final_snapshot_evaluation=True)
    assert res.final_belief_statuses["b_old"] == "SUPERSEDED"
    assert res.final_belief_statuses["b_new"] == "AUTHORIZED"

    # PREREQUISITE_BLOCK path: a belief whose required condition is BLOCKED -> BLOCKED.
    ev1 = _evidence("e1", "2026-01-01T10:00:00Z")
    ev2 = _evidence("e2", "2026-01-01T11:00:00Z")
    b_dep = BeliefNode(belief_id="b_dep", proposition="dep", source_evidence_ids=("e1",))
    cond = ConditionNode(condition_id="c1", scope_id="scope", text="prereq")
    dep_edge = DependencyEdge(edge_id="dep1", belief_id="b_dep", condition_id="c1", inducer="smoke")
    block_edge = EvidenceEdge(
        edge_id="edge_block", edge_type=EvidenceEdgeType.BLOCKS, evidence_id="e2",
        target_kind="condition", target_id="c1", verifier="smoke",
    )
    s1 = SubagentMemorySubmission(
        submission_id="s1", producer_id="a", producer_role="r",
        parent_snapshot_id="snap_0", observed_at="2026-01-01T10:00:00Z",
        instance_id="i", query_id="q", query="Q",
        evidence_context=(ev1,), new_evidence_id="e1", candidate_beliefs=(b_dep,),
        candidate_conditions_by_belief=(("b_dep", (cond,)),),
        dependency_edges_by_belief=(("b_dep", (dep_edge,)),),
    )
    s2 = SubagentMemorySubmission(
        submission_id="s2", producer_id="a", producer_role="r",
        parent_snapshot_id="snap_1", observed_at="2026-01-01T11:00:00Z",
        instance_id="i", query_id="q", query="Q",
        evidence_context=(ev1, ev2), new_evidence_id="e2", candidate_beliefs=(),
        proposal_batches=(EvidenceProposalBatch(edges=(block_edge,)),),
    )
    res2 = commit_submission_sequence((s1, s2), final_snapshot_evaluation=True)
    assert res2.final_belief_statuses["b_dep"] == "BLOCKED"
    assert "b_dep" in res2.final_excluded_belief_ids


# --- Item 10: Stage B canonicalization is separate from strict matching ----- #
def test_item10_stage_b_canonicalization_tracked_separately():
    # "b_loc" prefix-matches the only valid id "b_location"; canonicalization is
    # recorded as applied/prefix WITHOUT silently treating it as an exact match.
    verdicts = parse_direct_judge_response(
        json.dumps({"verdicts": [{"belief_id": "b_loc", "status": "USABLE", "rationale": "x"}]}),
        valid_belief_ids={"b_location"},
    )
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v["raw_belief_id"] == "b_loc"
    assert v["canonical_belief_id"] == "b_location"
    assert v["canonicalization_applied"] is True
    assert v["canonicalization_type"] == "prefix"


def test_runner_imports_json():
    from retracemem.evaluation.multiagent import runner
    assert hasattr(runner, "json")
