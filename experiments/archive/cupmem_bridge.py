from __future__ import annotations

from typing import Any, Dict, List, Tuple
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
)
from retracemem.authorization import EvidenceProposalBatch
from retracemem.multiagent.contracts import SubagentMemorySubmission
from experiments.archive.stale_adapter import assert_no_evaluation_leakage


def map_delta_to_submission(
    delta: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    active_items: List[Dict[str, Any]],
    parent_snapshot_id: str,
) -> SubagentMemorySubmission:
    """Truthfully maps a CUPMem SessionDelta update to a SubagentMemorySubmission."""
    # Ensure no leakage of STALE gold fields
    assert_no_evaluation_leakage(delta)
    assert_no_evaluation_leakage(chunks)
    assert_no_evaluation_leakage(active_items)

    session_id = str(delta["session_id"])
    session_time = str(delta["session_time"])
    delta_id = str(delta["delta_id"])
    bucket = str(delta["bucket"])
    track = str(delta["local_track"])

    # 1. Build EvidenceNodes from chunks referenced by the delta
    evidence_map = {}
    for c in chunks:
        ev_id = str(c["chunk_id"])
        evidence_map[ev_id] = EvidenceNode(
            evidence_id=ev_id,
            session_id=str(c.get("session_id", session_id)),
            timestamp=session_time,
            text=str(c["text"]),
            source_dataset="stale",
            source_pointer=f"stale://{session_id}/{ev_id}",
        )

    evidence_context = tuple(evidence_map.values())
    if not evidence_context:
        # Fallback evidence node if none referenced
        fallback_ev = EvidenceNode(
            evidence_id=f"e_{delta_id}",
            session_id=session_id,
            timestamp=session_time,
            text=f"Update for {bucket}/{track}",
            source_dataset="stale",
            source_pointer=f"stale://{session_id}/fallback_{delta_id}",
        )
        evidence_context = (fallback_ev,)
        new_evidence_id = fallback_ev.evidence_id
    else:
        # Use the first referenced evidence chunk id or the latest one
        new_evidence_id = str(delta["evidence_chunk_ids"][0]) if delta.get("evidence_chunk_ids") else evidence_context[0].evidence_id

    # 2. Build Candidate Beliefs from active items matching the bucket and local_track
    candidate_beliefs_list = []
    super_edges = []
    
    # Replacement belief for the new delta
    new_belief = BeliefNode(
        belief_id=f"b_{delta_id}",
        proposition=str(delta["proposed_value"]),
        source_evidence_ids=tuple(delta.get("evidence_chunk_ids", [new_evidence_id])),
    )

    for item in active_items:
        if str(item.get("bucket")) == bucket and str(item.get("local_track")) == track:
            old_belief = BeliefNode(
                belief_id=f"b_{item['item_id']}",
                proposition=str(item["value"]),
                source_evidence_ids=tuple(item.get("evidence_chunk_ids", [])),
            )
            candidate_beliefs_list.append(old_belief)
            
            # Create a SUPERSEDES proposal edge targeting the old active item
            super_edges.append(
                EvidenceEdge(
                    edge_id=f"edge_super_{delta_id}_{item['item_id']}",
                    edge_type=EvidenceEdgeType.SUPERSEDES,
                    evidence_id=new_evidence_id,
                    target_kind="belief",
                    target_id=old_belief.belief_id,
                    verifier="cupmem_bridge",
                    replacement_belief_id=new_belief.belief_id,
                )
            )

    # 3. Compile proposal batch
    proposal_batches = ()
    if super_edges:
        proposal_batches = (
            EvidenceProposalBatch(
                edges=tuple(super_edges),
                model_call_trace_id=f"trace_{delta_id}",
            ),
        )

    # Ensure no overlap between candidate_beliefs and candidate_replacement_beliefs
    return SubagentMemorySubmission(
        submission_id=delta_id,
        producer_id="subagent_cupmem",
        producer_role="profile_writer",
        parent_snapshot_id=parent_snapshot_id,
        observed_at=session_time,
        instance_id=f"inst_{session_id}",
        query_id="q_write",
        query=f"Write update for {bucket}/{track}",
        evidence_context=evidence_context,
        new_evidence_id=new_evidence_id,
        candidate_beliefs=tuple(candidate_beliefs_list),
        candidate_replacement_beliefs=(new_belief,),
        proposal_batches=proposal_batches,
        task_id=f"task_{session_id}",
        metadata={"bucket": bucket, "local_track": track},
    )


def map_invalidation_to_submission(
    proposal: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    active_items: List[Dict[str, Any]],
    parent_snapshot_id: str,
    session_time: str,
) -> SubagentMemorySubmission:
    """Truthfully maps a CUPMem InvalidationProposal to a SubagentMemorySubmission."""
    assert_no_evaluation_leakage(proposal)
    assert_no_evaluation_leakage(chunks)
    assert_no_evaluation_leakage(active_items)

    session_id = str(proposal["session_id"])
    proposal_id = str(proposal["proposal_id"])
    bucket = str(proposal["target_bucket"])
    track = str(proposal["target_local_track"])

    # 1. Build EvidenceNodes
    evidence_map = {}
    for c in chunks:
        ev_id = str(c["chunk_id"])
        evidence_map[ev_id] = EvidenceNode(
            evidence_id=ev_id,
            session_id=str(c.get("session_id", session_id)),
            timestamp=session_time,
            text=str(c["text"]),
            source_dataset="stale",
            source_pointer=f"stale://{session_id}/{ev_id}",
        )

    evidence_context = tuple(evidence_map.values())
    if not evidence_context:
        fallback_ev = EvidenceNode(
            evidence_id=f"e_{proposal_id}",
            session_id=session_id,
            timestamp=session_time,
            text=f"Invalidation proposal for {bucket}/{track}",
            source_dataset="stale",
            source_pointer=f"stale://{session_id}/fallback_{proposal_id}",
        )
        evidence_context = (fallback_ev,)
        new_evidence_id = fallback_ev.evidence_id
    else:
        new_evidence_id = str(proposal["evidence_chunk_ids"][0]) if proposal.get("evidence_chunk_ids") else evidence_context[0].evidence_id

    # 2. Build candidate conditions and dependency edges
    candidate_beliefs_list = []
    candidate_conditions = []
    dep_edges = []
    block_edges = []

    # Map target active profile items
    for item in active_items:
        if str(item.get("bucket")) == bucket and str(item.get("local_track")) == track:
            belief_id = f"b_{item['item_id']}"
            old_belief = BeliefNode(
                belief_id=belief_id,
                proposition=str(item["value"]),
                source_evidence_ids=tuple(item.get("evidence_chunk_ids", [])),
            )
            candidate_beliefs_list.append(old_belief)

            # Establish condition node representing profile track validity
            cond = ConditionNode(
                condition_id=f"c_{item['item_id']}",
                scope_id=f"scope_{bucket}_{track}",
                text=f"Profile track '{bucket}/{track}' remains active and valid",
            )
            candidate_conditions.append((belief_id, (cond,)))

            # Dependency: belief REQUIRES condition
            dep = DependencyEdge(
                edge_id=f"dep_{item['item_id']}",
                belief_id=belief_id,
                condition_id=cond.condition_id,
                inducer="cupmem_bridge",
            )
            dep_edges.append((belief_id, (dep,)))

            # BLOCKS edge from invalidation trigger to the condition
            block_edges.append(
                EvidenceEdge(
                    edge_id=f"edge_block_{proposal_id}_{item['item_id']}",
                    edge_type=EvidenceEdgeType.BLOCKS,
                    evidence_id=new_evidence_id,
                    target_kind="condition",
                    target_id=cond.condition_id,
                    verifier="cupmem_bridge",
                )
            )

    proposal_batches = ()
    if block_edges:
        proposal_batches = (
            EvidenceProposalBatch(
                edges=tuple(block_edges),
                model_call_trace_id=f"trace_{proposal_id}",
            ),
        )

    return SubagentMemorySubmission(
        submission_id=proposal_id,
        producer_id="subagent_cupmem",
        producer_role="profile_invalidator",
        parent_snapshot_id=parent_snapshot_id,
        observed_at=session_time,
        instance_id=f"inst_{session_id}",
        query_id="q_invalidate",
        query=f"Propose invalidation for {bucket}/{track}",
        evidence_context=evidence_context,
        new_evidence_id=new_evidence_id,
        candidate_beliefs=tuple(candidate_beliefs_list),
        candidate_replacement_beliefs=(),
        candidate_conditions_by_belief=tuple(candidate_conditions),
        dependency_edges_by_belief=tuple(dep_edges),
        proposal_batches=proposal_batches,
        task_id=f"task_{session_id}",
        metadata={"bucket": bucket, "local_track": track},
    )
