"""Archived (E4/E6): CupMem candidate -> subagent submission mapping.

Relocated from tests/experiments/shared/test_multiagent.py because it exercises
the archived external-validation bridge (experiments.archive.cupmem_adapter),
which is not part of the canonical Stage A/B/C evaluation path.
"""
from __future__ import annotations

from retracemem.schemas import BeliefNode, EvidenceNode
from experiments.archive.cupmem_adapter import (
    CupMemRevisionCandidate,
    map_cupmem_candidate_to_subagent_submission,
)


def test_cupmem_candidate_mapping():
    ev_1 = EvidenceNode(
        evidence_id="e_1", session_id="s_1", timestamp="2026-05-29T10:00:00Z",
        text="A", source_dataset="t", source_pointer="p"
    )
    b_1 = BeliefNode(belief_id="b_1", proposition="A", source_evidence_ids=("e_1",))

    candidate = CupMemRevisionCandidate(
        submission_id="sub_c_1",
        producer_id="agent_c",
        producer_role="role_c",
        parent_snapshot_id="snap_0",
        observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_1",
        query_id="q_1",
        query="Q",
        evidence_context=(ev_1,),
        new_evidence_id="e_1",
        candidate_beliefs=(b_1,),
        upstream_trace={"task_id": "task_cup"}
    )

    sub = map_cupmem_candidate_to_subagent_submission(candidate)
    assert sub.submission_id == "sub_c_1"
    assert sub.task_id == "task_cup"
    assert sub.metadata == {"task_id": "task_cup"}
