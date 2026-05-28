from __future__ import annotations

from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)
from retracemem.tms.rollback import RollbackDiagnostics


def test_rollback_diagnostics_calculation() -> None:
    """Typed-graph rollback scenario: commute by bicycle blocked then released.

    Sequence:
    - ev_commute_bike: User commutes by bicycle (belief created)
    - ev_broke_leg: User broke their leg (BLOCKS cycling_ability)
    - ev_recovered: User recovered (RELEASES cycling_ability)

    Expected authorization over time:
    - at ev_commute_bike: b_commute_bike = True
    - at ev_broke_leg: b_commute_bike = False
    - at ev_recovered: b_commute_bike = True (recovered)
    """

    ledger = EpisodeLedger()
    store = BeliefStore()

    ev1 = EvidenceNode(
        evidence_id="ev_commute_bike",
        session_id="s1",
        timestamp="2026-05-27T01:00:00Z",
        text="User commutes by bicycle",
        source_dataset="s1",
        source_pointer="test",
    )
    ev2 = EvidenceNode(
        evidence_id="ev_broke_leg",
        session_id="s1",
        timestamp="2026-05-27T02:00:00Z",
        text="User broke their leg",
        source_dataset="s1",
        source_pointer="test",
    )
    ev3 = EvidenceNode(
        evidence_id="ev_recovered",
        session_id="s1",
        timestamp="2026-05-27T03:00:00Z",
        text="User recovered from broken leg",
        source_dataset="s1",
        source_pointer="test",
    )
    ledger.append(ev1)
    ledger.append(ev2)
    ledger.append(ev3)

    belief = BeliefNode(
        belief_id="b_commute_bike",
        proposition="User commutes by bicycle",
        source_evidence_ids=("ev_commute_bike",),
    )
    store.add_belief(belief)

    condition = ConditionNode(
        condition_id="cond_cycling_ability",
        scope_id="scope_default",
        text="cycling ability",
    )
    store.add_condition(condition)

    dep = DependencyEdge(
        edge_id="dep_bike_cycling",
        belief_id="b_commute_bike",
        condition_id="cond_cycling_ability",
        inducer="manual_fixture",
    )
    store.add_dependency_edge(dep)

    block_edge = EvidenceEdge(
        edge_id="ev_edge_block_cycling",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_broke_leg",
        target_kind="condition",
        target_id="cond_cycling_ability",
        verifier="manual_fixture",
    )
    store.add_evidence_edge(block_edge)

    release_edge = EvidenceEdge(
        edge_id="ev_edge_release_cycling",
        edge_type=EvidenceEdgeType.RELEASES,
        evidence_id="ev_recovered",
        target_kind="condition",
        target_id="cond_cycling_ability",
        verifier="manual_fixture",
    )
    store.add_evidence_edge(release_edge)

    diag = RollbackDiagnostics(store, ledger)

    # 1. HER test: 3 inputs retained
    her = diag.calculate_her(3)
    assert her == 1.0

    # 2. RAR and URR test
    ground_truth = {
        "ev_commute_bike": {"b_commute_bike": True},
        "ev_broke_leg": {"b_commute_bike": False},
        "ev_recovered": {"b_commute_bike": True},
    }

    metrics = diag.calculate_rar_and_urr(ground_truth)
    assert "RAR" in metrics
    assert "URR_react" in metrics
    assert metrics["RAR"] == 1.0
    assert metrics["URR_react"] == 0.0
