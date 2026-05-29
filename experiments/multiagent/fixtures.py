from __future__ import annotations

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


def get_diagnostic_fixtures() -> dict[str, tuple[SubagentMemorySubmission, ...]]:
    # Common base nodes
    ev_init = EvidenceNode(
        evidence_id="e_init", session_id="sess_0", timestamp="2026-05-29T10:00:00Z",
        text="User lives in Seattle.", source_dataset="stale_dev", source_pointer="dev://1"
    )
    b_location = BeliefNode(
        belief_id="b_location", proposition="User lives in Seattle.", source_evidence_ids=("e_init",)
    )

    # 1. Conflicting submissions from two subagents (Sequential updates)
    ev_sub_a = EvidenceNode(
        evidence_id="e_sub_a", session_id="sess_1", timestamp="2026-05-29T11:00:00Z",
        text="User moved to Portland.", source_dataset="stale_dev", source_pointer="dev://2"
    )
    b_portland = BeliefNode(
        belief_id="b_portland", proposition="User lives in Portland.", source_evidence_ids=("e_sub_a",)
    )
    super_edge_a = EvidenceEdge(
        edge_id="edge_super_a", edge_type=EvidenceEdgeType.SUPERSEDES, evidence_id="e_sub_a",
        target_kind="belief", target_id="b_location", verifier="agent_a",
        replacement_belief_id="b_portland"
    )

    sub_conflict_1 = SubagentMemorySubmission(
        submission_id="sub_conflict_01",
        producer_id="agent_a",
        producer_role="location_tracker",
        parent_snapshot_id="snap_root",
        observed_at="2026-05-29T11:00:00Z",
        instance_id="diag_conflict",
        query_id="q_1",
        query="Where does the user live?",
        evidence_context=(ev_init, ev_sub_a),
        new_evidence_id="e_sub_a",
        candidate_beliefs=(b_location,),
        candidate_replacement_beliefs=(b_portland,),
        proposal_batches=(
            EvidenceProposalBatch(edges=(super_edge_a,), model_call_trace_id="trace_a"),
        )
    )

    ev_sub_b = EvidenceNode(
        evidence_id="e_sub_b", session_id="sess_2", timestamp="2026-05-29T12:00:00Z",
        text="User moved to Vancouver.", source_dataset="stale_dev", source_pointer="dev://3"
    )
    b_vancouver = BeliefNode(
        belief_id="b_vancouver", proposition="User lives in Vancouver.", source_evidence_ids=("e_sub_b",)
    )
    super_edge_b = EvidenceEdge(
        edge_id="edge_super_b", edge_type=EvidenceEdgeType.SUPERSEDES, evidence_id="e_sub_b",
        target_kind="belief", target_id="b_portland", verifier="agent_b",
        replacement_belief_id="b_vancouver"
    )

    sub_conflict_2 = SubagentMemorySubmission(
        submission_id="sub_conflict_02",
        producer_id="agent_b",
        producer_role="location_tracker",
        parent_snapshot_id="snap_conflict_1",
        observed_at="2026-05-29T12:00:00Z",
        instance_id="diag_conflict",
        query_id="q_1",
        query="Where does the user live?",
        evidence_context=(ev_init, ev_sub_a, ev_sub_b),
        new_evidence_id="e_sub_b",
        candidate_beliefs=(b_portland,),
        candidate_replacement_beliefs=(b_vancouver,),
        proposal_batches=(
            EvidenceProposalBatch(edges=(super_edge_b,), model_call_trace_id="trace_b"),
        )
    )

    sub_conflict_query = SubagentMemorySubmission(
        submission_id="sub_conflict_query",
        producer_id="agent_reader",
        producer_role="query_reader",
        parent_snapshot_id="snap_conflict_2",
        observed_at="2026-05-29T12:05:00Z",
        instance_id="diag_conflict",
        query_id="q_1",
        query="Where does the user live?",
        evidence_context=(ev_init, ev_sub_a, ev_sub_b),
        new_evidence_id="e_sub_b",
        candidate_beliefs=(b_location, b_portland, b_vancouver),
        candidate_replacement_beliefs=(),
        proposal_batches=(
            EvidenceProposalBatch(edges=(super_edge_a, super_edge_b), model_call_trace_id="trace_q"),
        )
    )

    # 2. Stale propagation update
    ev_stale = EvidenceNode(
        evidence_id="e_stale", session_id="sess_3", timestamp="2026-05-29T09:00:00Z",
        text="User spotted in Seattle.", source_dataset="stale_dev", source_pointer="dev://4"
    )
    reaffirm_edge = EvidenceEdge(
        edge_id="edge_reaffirm_stale", edge_type=EvidenceEdgeType.REAFFIRMS, evidence_id="e_stale",
        target_kind="belief", target_id="b_location", verifier="agent_c"
    )
    sub_stale = SubagentMemorySubmission(
        submission_id="sub_stale_01",
        producer_id="agent_c",
        producer_role="stale_reporter",
        parent_snapshot_id="snap_conflict_1",
        observed_at="2026-05-29T13:00:00Z",
        instance_id="diag_stale",
        query_id="q_1",
        query="Where does the user live?",
        evidence_context=(ev_init, ev_sub_a, ev_stale),
        new_evidence_id="e_stale",
        candidate_beliefs=(b_location, b_portland),
        proposal_batches=(
            EvidenceProposalBatch(edges=(reaffirm_edge, super_edge_a), model_call_trace_id="trace_c"),
        )
    )

    sub_stale_query = SubagentMemorySubmission(
        submission_id="sub_stale_query",
        producer_id="agent_reader",
        producer_role="query_reader",
        parent_snapshot_id="snap_stale_1",
        observed_at="2026-05-29T13:05:00Z",
        instance_id="diag_stale",
        query_id="q_1",
        query="Where does the user live?",
        evidence_context=(ev_init, ev_sub_a, ev_stale),
        new_evidence_id="e_sub_a",  # Query as of the newest chronological evidence
        candidate_beliefs=(b_location, b_portland),
        candidate_replacement_beliefs=(),
        proposal_batches=(
            EvidenceProposalBatch(edges=(reaffirm_edge, super_edge_a), model_call_trace_id="trace_stale_q"),
        )
    )

    # 3. Scope-expansion trap (protected_belief)
    b_hobby = BeliefNode(
        belief_id="b_hobby", proposition="User likes hiking.", source_evidence_ids=("e_init",)
    )
    sub_scope = SubagentMemorySubmission(
        submission_id="sub_scope_01",
        producer_id="agent_d",
        producer_role="profile_builder",
        parent_snapshot_id="snap_root",
        observed_at="2026-05-29T11:00:00Z",
        instance_id="diag_scope",
        query_id="q_2",
        query="What are the user's hobbies?",
        evidence_context=(ev_init, ev_sub_a),
        new_evidence_id="e_sub_a",
        candidate_beliefs=(b_hobby,),
        proposal_batches=()  # Empty proposals -> hobby remains AUTHORIZED
    )

    # 4. Temporary blocker followed by recovery/release
    c_phys = ConditionNode(
        condition_id="c_phys", scope_id="user_scope", text="User is physically able to cycle."
    )
    b_commute = BeliefNode(
        belief_id="b_commute", proposition="User commutes by bicycle.", source_evidence_ids=("e_init",)
    )
    dep_commute = DependencyEdge(
        edge_id="dep_commute", belief_id="b_commute", condition_id="c_phys", inducer="writer"
    )

    ev_broken_leg = EvidenceNode(
        evidence_id="e_broken_leg", session_id="sess_4", timestamp="2026-05-29T12:00:00Z",
        text="User broke their leg.", source_dataset="stale_dev", source_pointer="dev://5"
    )
    block_edge = EvidenceEdge(
        edge_id="edge_block", edge_type=EvidenceEdgeType.BLOCKS, evidence_id="e_broken_leg",
        target_kind="condition", target_id="c_phys", verifier="agent_e"
    )
    sub_block = SubagentMemorySubmission(
        submission_id="sub_block_01",
        producer_id="agent_e",
        producer_role="health_monitor",
        parent_snapshot_id="snap_root",
        observed_at="2026-05-29T12:00:00Z",
        instance_id="diag_block",
        query_id="q_3",
        query="How does the user commute?",
        evidence_context=(ev_init, ev_broken_leg),
        new_evidence_id="e_broken_leg",
        candidate_beliefs=(b_commute,),
        candidate_conditions_by_belief=(("b_commute", (c_phys,)),),
        dependency_edges_by_belief=(("b_commute", (dep_commute,)),),
        proposal_batches=(
            EvidenceProposalBatch(edges=(block_edge,), model_call_trace_id="trace_e"),
        )
    )

    ev_recovered = EvidenceNode(
        evidence_id="e_recovered", session_id="sess_5", timestamp="2026-05-29T15:00:00Z",
        text="User's leg has fully healed.", source_dataset="stale_dev", source_pointer="dev://6"
    )
    release_edge = EvidenceEdge(
        edge_id="edge_release", edge_type=EvidenceEdgeType.RELEASES, evidence_id="e_recovered",
        target_kind="condition", target_id="c_phys", verifier="agent_f"
    )
    sub_release = SubagentMemorySubmission(
        submission_id="sub_release_01",
        producer_id="agent_f",
        producer_role="health_monitor",
        parent_snapshot_id="snap_blocked",
        observed_at="2026-05-29T15:00:00Z",
        instance_id="diag_block",
        query_id="q_3",
        query="How does the user commute?",
        evidence_context=(ev_init, ev_broken_leg, ev_recovered),
        new_evidence_id="e_recovered",
        candidate_beliefs=(b_commute,),
        candidate_conditions_by_belief=(("b_commute", (c_phys,)),),
        dependency_edges_by_belief=(("b_commute", (dep_commute,)),),
        proposal_batches=(
            EvidenceProposalBatch(edges=(release_edge, block_edge), model_call_trace_id="trace_f"),
        )
    )

    # 5. Duplicate evidence from two agents
    ev_dog = EvidenceNode(
        evidence_id="e_dog", session_id="sess_6", timestamp="2026-05-29T14:00:00Z",
        text="User owns a Labrador retriever.", source_dataset="stale_dev", source_pointer="dev://7"
    )
    b_dog = BeliefNode(
        belief_id="b_dog", proposition="User has a dog.", source_evidence_ids=("e_dog",)
    )
    sub_dup_1 = SubagentMemorySubmission(
        submission_id="sub_dup_01",
        producer_id="agent_x",
        producer_role="pet_detector",
        parent_snapshot_id="snap_root",
        observed_at="2026-05-29T14:00:00Z",
        instance_id="diag_dup",
        query_id="q_4",
        query="Does the user have pets?",
        evidence_context=(ev_dog,),
        new_evidence_id="e_dog",
        candidate_beliefs=(b_dog,),
        proposal_batches=()
    )
    sub_dup_2 = SubagentMemorySubmission(
        submission_id="sub_dup_02",
        producer_id="agent_y",
        producer_role="lifestyle_detector",
        parent_snapshot_id="snap_dup_1",
        observed_at="2026-05-29T14:05:00Z",
        instance_id="diag_dup",
        query_id="q_4",
        query="Does the user have pets?",
        evidence_context=(ev_dog,),
        new_evidence_id="e_dog",
        candidate_beliefs=(b_dog,),
        proposal_batches=()
    )

    # 6. Uncertain update that remains unresolved
    ev_cat = EvidenceNode(
        evidence_id="e_cat", session_id="sess_7", timestamp="2026-05-29T16:00:00Z",
        text="A cat food bowl was seen in the house.", source_dataset="stale_dev", source_pointer="dev://8"
    )
    b_cat = BeliefNode(
        belief_id="b_cat", proposition="User owns a cat.", source_evidence_ids=("e_cat",)
    )
    uncertain_edge = EvidenceEdge(
        edge_id="edge_uncertain", edge_type=EvidenceEdgeType.UNCERTAIN, evidence_id="e_cat",
        target_kind="belief", target_id="b_cat", verifier="agent_z"
    )
    sub_uncertain = SubagentMemorySubmission(
        submission_id="sub_uncertain_01",
        producer_id="agent_z",
        producer_role="pet_detector",
        parent_snapshot_id="snap_root",
        observed_at="2026-05-29T16:00:00Z",
        instance_id="diag_uncertain",
        query_id="q_5",
        query="Does the user have a cat?",
        evidence_context=(ev_cat,),
        new_evidence_id="e_cat",
        candidate_beliefs=(b_cat,),
        proposal_batches=(
            EvidenceProposalBatch(edges=(uncertain_edge,), model_call_trace_id="trace_z"),
        )
    )

    return {
        "conflict": (sub_conflict_1, sub_conflict_2, sub_conflict_query),
        "stale_propagation": (sub_conflict_1, sub_stale, sub_stale_query),
        "protected_belief": (sub_scope,),
        "temporary_blocker": (sub_block, sub_release),
        "duplicate_evidence": (sub_dup_1, sub_dup_2),
        "uncertain_update": (sub_uncertain,),
    }
