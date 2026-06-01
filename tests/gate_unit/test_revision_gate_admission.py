"""Structural admission tests for `RevisionGate`.

These tests live in `tests/gate_unit/` because the gate is part of the
Wave-1A typed graph core; they assert the gate's structural rules
without invoking any verifier or DPA logic.

The gate is structural-only:

- DependencyEdge: edge_type must be REQUIRES; belief and condition must
  exist; an `inducer` provenance string is mandatory (amendment A8).
- EvidenceEdge: edge_type vs target_kind matrix is fixed; targets must
  exist in the store; SUPERSEDES carries a non-null
  `replacement_belief_id` that itself exists (amendment A1).

It does NOT enforce defeat anchoring (BLOCKS-needs-REQUIRES); that
constraint lives in `DefeatPathAuthorizationAlgorithm` and is exercised
by `test_blocks_without_requires_anchor_does_not_defeat_belief`.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(__file__)
_HELPERS_PATH = os.path.join(_HERE, "_helpers.py")
_helpers_spec = importlib.util.spec_from_file_location(
    "tests_gate_unit_helpers", _HELPERS_PATH
)
assert _helpers_spec is not None and _helpers_spec.loader is not None
_helpers = importlib.util.module_from_spec(_helpers_spec)
sys.modules.setdefault("tests_gate_unit_helpers", _helpers)
_helpers_spec.loader.exec_module(_helpers)

make_belief = _helpers.make_belief
make_condition = _helpers.make_condition
make_dependency = _helpers.make_dependency
make_evidence_edge = _helpers.make_evidence_edge

from retracemem.memory.belief_store import BeliefStore
from retracemem.schemas import (
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
)
from retracemem.tms.gate import RevisionGate


def _store_with(belief_ids: tuple[str, ...] = (), condition_ids: tuple[str, ...] = ()) -> BeliefStore:
    store = BeliefStore()
    for bid in belief_ids:
        store.add_belief(make_belief(bid, f"proposition for {bid}"))
    for cid in condition_ids:
        store.add_condition(make_condition(cid, f"text for {cid}"))
    return store


# ---------------------------------------------------------------------------
# DependencyEdge admission
# ---------------------------------------------------------------------------


def test_admit_well_formed_requires_edge() -> None:
    store = _store_with(belief_ids=("bel_x",), condition_ids=("cond_y",))
    edge = make_dependency(
        edge_id="dep_001",
        belief_id="bel_x",
        condition_id="cond_y",
    )
    decision = RevisionGate().admit_dependency_edge(edge, store)
    assert decision.admitted is True
    assert decision.reason == "ok"


def test_reject_dependency_edge_with_non_requires_type() -> None:
    store = _store_with(belief_ids=("bel_x",), condition_ids=("cond_y",))
    edge = DependencyEdge(
        edge_id="dep_002",
        belief_id="bel_x",
        condition_id="cond_y",
        inducer="manual_fixture",
        edge_type="IMPLIES",
    )
    decision = RevisionGate().admit_dependency_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "dependency_edge_type_not_requires"


def test_reject_dependency_edge_with_unknown_belief() -> None:
    store = _store_with(condition_ids=("cond_y",))
    edge = make_dependency(edge_id="dep_003", belief_id="bel_missing", condition_id="cond_y")
    decision = RevisionGate().admit_dependency_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "unknown_belief"


def test_reject_dependency_edge_without_inducer_provenance() -> None:
    """Amendment A8: dependency edges must carry first-class provenance."""

    store = _store_with(belief_ids=("bel_x",), condition_ids=("cond_y",))
    edge = DependencyEdge(
        edge_id="dep_004",
        belief_id="bel_x",
        condition_id="cond_y",
        inducer="",
    )
    decision = RevisionGate().admit_dependency_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "missing_inducer_provenance"


# ---------------------------------------------------------------------------
# EvidenceEdge admission
# ---------------------------------------------------------------------------


def test_admit_blocks_targeting_condition() -> None:
    store = _store_with(condition_ids=("cond_y",))
    edge = make_evidence_edge(
        edge_id="ev_edge_001",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_n_001",
        target_kind="condition",
        target_id="cond_y",
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is True
    assert decision.reason == "ok"


def test_reject_blocks_targeting_belief() -> None:
    """BLOCKS must target a condition, not a belief."""

    store = _store_with(belief_ids=("bel_x",), condition_ids=("cond_y",))
    edge = make_evidence_edge(
        edge_id="ev_edge_002",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_n_002",
        target_kind="belief",
        target_id="bel_x",
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "target_kind_mismatch_for_blocks"


def test_reject_supersedes_without_replacement_belief_id() -> None:
    """Amendment A1: SUPERSEDES must populate replacement_belief_id."""

    store = _store_with(belief_ids=("bel_old",))
    edge = make_evidence_edge(
        edge_id="ev_edge_003",
        edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_n_003",
        target_kind="belief",
        target_id="bel_old",
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "supersedes_missing_replacement_belief_id"


def test_reject_supersedes_with_self_replacement() -> None:
    store = _store_with(belief_ids=("bel_old",))
    edge = make_evidence_edge(
        edge_id="ev_edge_004",
        edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_n_004",
        target_kind="belief",
        target_id="bel_old",
        replacement_belief_id="bel_old",
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "supersedes_replacement_equals_target"


def test_reject_supersedes_with_unknown_replacement_belief() -> None:
    store = _store_with(belief_ids=("bel_old",))
    edge = make_evidence_edge(
        edge_id="ev_edge_005",
        edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_n_005",
        target_kind="belief",
        target_id="bel_old",
        replacement_belief_id="bel_missing",
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "supersedes_unknown_replacement_belief"


def test_admit_supersedes_with_valid_replacement() -> None:
    store = _store_with(belief_ids=("bel_old", "bel_new"))
    edge = make_evidence_edge(
        edge_id="ev_edge_006",
        edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_n_006",
        target_kind="belief",
        target_id="bel_old",
        replacement_belief_id="bel_new",
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is True
    assert decision.reason == "ok"


def test_reject_reaffirms_with_replacement_belief_id() -> None:
    """Only SUPERSEDES may carry replacement_belief_id."""

    store = _store_with(belief_ids=("bel_x", "bel_other"))
    edge = EvidenceEdge(
        edge_id="ev_edge_007",
        edge_type=EvidenceEdgeType.REAFFIRMS,
        evidence_id="ev_n_007",
        target_kind="belief",
        target_id="bel_x",
        verifier="manual_fixture",
        replacement_belief_id="bel_other",
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "replacement_belief_id_only_valid_for_supersedes"


def test_reject_evidence_edge_without_verifier_provenance() -> None:
    """Amendment A8: evidence edges must carry first-class provenance."""

    store = _store_with(condition_ids=("cond_y",))
    edge = EvidenceEdge(
        edge_id="ev_edge_008",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_n_008",
        target_kind="condition",
        target_id="cond_y",
        verifier="",
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "missing_verifier_provenance"


def test_admit_evidence_edge_with_scope() -> None:
    store = _store_with(condition_ids=("cond_y",))
    edge = EvidenceEdge(
        edge_id="ev_edge_009",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_n_009",
        target_kind="condition",
        target_id="cond_y",
        verifier="verifier_test",
        metadata={"scope": "partial"},
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is True
    assert decision.reason == "ok"


def test_reject_evidence_edge_with_invalid_scope() -> None:
    store = _store_with(condition_ids=("cond_y",))
    edge = EvidenceEdge(
        edge_id="ev_edge_010",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_n_010",
        target_kind="condition",
        target_id="cond_y",
        verifier="verifier_test",
        metadata={"scope": 123},
    )
    decision = RevisionGate().admit_evidence_edge(edge, store)
    assert decision.admitted is False
    assert decision.reason == "invalid_scope_type"
