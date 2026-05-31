from __future__ import annotations

from typing import List
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
from experiments.multiagent.fixtures import get_diagnostic_fixtures
from retracemem.evaluation.multiagent.contracts import (
    MultiAgentMemoryEpisode,
    DownstreamTask,
    GoldSnapshotExpectation,
)


def get_dev_episodes() -> List[MultiAgentMemoryEpisode]:
    fixtures = get_diagnostic_fixtures()
    episodes = []

    ev_init = EvidenceNode(
        evidence_id="e_init", session_id="sess_0", timestamp="2026-05-29T10:00:00Z",
        text="User lives in Seattle.", source_dataset="stale_dev", source_pointer="dev://1"
    )

    # 1. Conflict (Migrated -> software_engineering, cross_agent_conflict)
    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_conflict",
            domain="software_engineering",
            failure_type="cross_agent_conflict",
            subagent_roles=("location_tracker", "location_tracker", "query_reader"),
            submissions=fixtures["conflict"],
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_conflict",
                    query="Where does the user live?",
                    expected_answer_or_action="Vancouver",
                    relevant_belief_ids=("b_vancouver",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_vancouver": "AUTHORIZED",
                    "b_portland": "SUPERSEDED",
                    "b_location": "SUPERSEDED",
                }
            ),
            stress_factors={"conflict_density": 0.8, "delay_depth": 1},
        )
    )

    # 2. Stale propagation (Migrated -> software_engineering, stale_propagation)
    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_stale_propagation",
            domain="software_engineering",
            failure_type="stale_propagation",
            subagent_roles=("location_tracker", "stale_reporter", "query_reader"),
            submissions=fixtures["stale_propagation"],
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_stale",
                    query="Where does the user live?",
                    expected_answer_or_action="Portland",
                    relevant_belief_ids=("b_portland",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_portland": "AUTHORIZED",
                    "b_location": "SUPERSEDED",
                }
            ),
            stress_factors={"conflict_density": 0.5, "delay_depth": 2},
        )
    )

    # 3. Scope expansion (Migrated -> software_engineering, scope_expansion)
    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_scope_expansion",
            domain="software_engineering",
            failure_type="scope_expansion",
            subagent_roles=("profile_builder",),
            submissions=fixtures["protected_belief"],
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_scope",
                    query="What are the user's hobbies?",
                    expected_answer_or_action="likes hiking",
                    relevant_belief_ids=("b_hobby",),
                    protected_belief_ids=("b_hobby",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_hobby": "AUTHORIZED",
                }
            ),
            stress_factors={"conflict_density": 0.0, "delay_depth": 0},
        )
    )

    # 4. Temporary blocker (Migrated -> software_engineering, temporary_blocker_recovery)
    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_temporary_blocker",
            domain="software_engineering",
            failure_type="temporary_blocker_recovery",
            subagent_roles=("health_monitor", "health_monitor"),
            submissions=fixtures["temporary_blocker"],
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_blocker",
                    query="How does the user commute?",
                    expected_answer_or_action="bicycle",
                    relevant_belief_ids=("b_commute",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_commute": "AUTHORIZED",
                }
            ),
            stress_factors={"conflict_density": 0.2, "delay_depth": 1},
        )
    )

    # 5. Duplicate evidence (Migrated -> software_engineering, duplicate_evidence)
    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_duplicate_evidence",
            domain="software_engineering",
            failure_type="duplicate_evidence",
            subagent_roles=("pet_detector", "lifestyle_detector"),
            submissions=fixtures["duplicate_evidence"],
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_dup",
                    query="Does the user have pets?",
                    expected_answer_or_action="dog",
                    relevant_belief_ids=("b_dog",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_dog": "AUTHORIZED",
                }
            ),
            stress_factors={"conflict_density": 0.1, "delay_depth": 0},
        )
    )

    # 6. Uncertain update (Migrated -> software_engineering, ambiguous_update)
    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_uncertain_update",
            domain="software_engineering",
            failure_type="ambiguous_update",
            subagent_roles=("pet_detector",),
            submissions=fixtures["uncertain_update"],
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_uncertain",
                    query="Does the user have a cat?",
                    expected_answer_or_action="unresolved",
                    relevant_belief_ids=("b_cat",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_cat": "UNRESOLVED",
                }
            ),
            stress_factors={"conflict_density": 0.3, "delay_depth": 0},
        )
    )

    # 7. Research workflow - direct_supersession
    ev_paper_v1 = EvidenceNode(
        evidence_id="e_paper_v1", session_id="s_res_0", timestamp="2026-05-29T10:00:00Z",
        text="Version 1 claims 85% accuracy.", source_dataset="stale_dev", source_pointer="dev://10"
    )
    ev_paper_v2 = EvidenceNode(
        evidence_id="e_paper_v2", session_id="s_res_1", timestamp="2026-05-29T11:00:00Z",
        text="Camera ready claims 89% accuracy.", source_dataset="stale_dev", source_pointer="dev://11"
    )
    b_accuracy = BeliefNode(
        belief_id="b_accuracy", proposition="Accuracy is 85%.", source_evidence_ids=("e_paper_v1",)
    )
    b_accuracy_v2 = BeliefNode(
        belief_id="b_accuracy_v2", proposition="Accuracy is 89%.", source_evidence_ids=("e_paper_v2",)
    )
    super_edge_res = EvidenceEdge(
        edge_id="edge_super_res", edge_type=EvidenceEdgeType.SUPERSEDES, evidence_id="e_paper_v2",
        target_kind="belief", target_id="b_accuracy", verifier="agent_res",
        replacement_belief_id="b_accuracy_v2"
    )

    sub_res_1 = SubagentMemorySubmission(
        submission_id="sub_res_01", producer_id="agent_res", producer_role="researcher",
        parent_snapshot_id="snap_root", observed_at="2026-05-29T10:00:00Z",
        instance_id="diag_res_1", query_id="q_res_1", query="What is accuracy?",
        evidence_context=(ev_paper_v1,), new_evidence_id="e_paper_v1", candidate_beliefs=(b_accuracy,)
    )
    sub_res_2 = SubagentMemorySubmission(
        submission_id="sub_res_02", producer_id="agent_res", producer_role="researcher",
        parent_snapshot_id="snap_res_1", observed_at="2026-05-29T11:00:00Z",
        instance_id="diag_res_1", query_id="q_res_1", query="What is accuracy?",
        evidence_context=(ev_paper_v1, ev_paper_v2), new_evidence_id="e_paper_v2",
        candidate_beliefs=(b_accuracy,), candidate_replacement_beliefs=(b_accuracy_v2,),
        proposal_batches=(EvidenceProposalBatch(edges=(super_edge_res,), model_call_trace_id="trace_res_2"),)
    )
    sub_res_query = SubagentMemorySubmission(
        submission_id="sub_res_query", producer_id="agent_res", producer_role="researcher",
        parent_snapshot_id="snap_res_2", observed_at="2026-05-29T11:05:00Z",
        instance_id="diag_res_1", query_id="q_res_1", query="What is accuracy?",
        evidence_context=(ev_paper_v1, ev_paper_v2), new_evidence_id="e_paper_v2",
        candidate_beliefs=(b_accuracy, b_accuracy_v2), proposal_batches=(
            EvidenceProposalBatch(edges=(super_edge_res,), model_call_trace_id="trace_res_q"),
        )
    )

    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_research_direct_supersession",
            domain="research_workflow",
            failure_type="direct_supersession",
            subagent_roles=("researcher", "researcher"),
            submissions=(sub_res_1, sub_res_2, sub_res_query),
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_res_accuracy",
                    query="What is the final accuracy?",
                    expected_answer_or_action="89%",
                    relevant_belief_ids=("b_accuracy_v2",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_accuracy": "SUPERSEDED",
                    "b_accuracy_v2": "AUTHORIZED",
                }
            ),
            stress_factors={"conflict_density": 0.4, "delay_depth": 0},
        )
    )

    # 8. Research workflow - cross_agent_conflict
    ev_arxiv = EvidenceNode(
        evidence_id="e_arxiv", session_id="s_res_2", timestamp="2026-05-29T10:00:00Z",
        text="NEURIPS 2026.", source_dataset="stale_dev", source_pointer="dev://12"
    )
    ev_openreview = EvidenceNode(
        evidence_id="e_openreview", session_id="s_res_3", timestamp="2026-05-29T11:00:00Z",
        text="ICML 2026.", source_dataset="stale_dev", source_pointer="dev://13"
    )
    b_venue_neurips = BeliefNode(
        belief_id="b_venue_neurips", proposition="Paper X is NEURIPS.", source_evidence_ids=("e_arxiv",)
    )
    b_venue_icml = BeliefNode(
        belief_id="b_venue_icml", proposition="Paper X is ICML.", source_evidence_ids=("e_openreview",)
    )
    super_edge_conflict = EvidenceEdge(
        edge_id="edge_super_conflict", edge_type=EvidenceEdgeType.SUPERSEDES, evidence_id="e_openreview",
        target_kind="belief", target_id="b_venue_neurips", verifier="agent_res_b",
        replacement_belief_id="b_venue_icml"
    )

    sub_conflict_res_1 = SubagentMemorySubmission(
        submission_id="sub_c_res_01", producer_id="agent_res_a", producer_role="crawler",
        parent_snapshot_id="snap_root", observed_at="2026-05-29T10:00:00Z",
        instance_id="diag_res_2", query_id="q_res_2", query="What is venue?",
        evidence_context=(ev_arxiv,), new_evidence_id="e_arxiv", candidate_beliefs=(b_venue_neurips,)
    )
    sub_conflict_res_2 = SubagentMemorySubmission(
        submission_id="sub_c_res_02", producer_id="agent_res_b", producer_role="crawler",
        parent_snapshot_id="snap_res_1", observed_at="2026-05-29T11:00:00Z",
        instance_id="diag_res_2", query_id="q_res_2", query="What is venue?",
        evidence_context=(ev_arxiv, ev_openreview), new_evidence_id="e_openreview",
        candidate_beliefs=(b_venue_neurips,), candidate_replacement_beliefs=(b_venue_icml,),
        proposal_batches=(EvidenceProposalBatch(edges=(super_edge_conflict,), model_call_trace_id="trace_res_c2"),)
    )
    sub_conflict_res_query = SubagentMemorySubmission(
        submission_id="sub_c_res_query", producer_id="agent_res_reader", producer_role="reviewer",
        parent_snapshot_id="snap_res_2", observed_at="2026-05-29T11:05:00Z",
        instance_id="diag_res_2", query_id="q_res_2", query="What is venue?",
        evidence_context=(ev_arxiv, ev_openreview), new_evidence_id="e_openreview",
        candidate_beliefs=(b_venue_neurips, b_venue_icml), proposal_batches=(
            EvidenceProposalBatch(edges=(super_edge_conflict,), model_call_trace_id="trace_res_cq"),
        )
    )

    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_research_cross_agent_conflict",
            domain="research_workflow",
            failure_type="cross_agent_conflict",
            subagent_roles=("crawler", "crawler", "reviewer"),
            submissions=(sub_conflict_res_1, sub_conflict_res_2, sub_conflict_res_query),
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_res_venue",
                    query="What is the final venue?",
                    expected_answer_or_action="ICML",
                    relevant_belief_ids=("b_venue_icml",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_venue_neurips": "SUPERSEDED",
                    "b_venue_icml": "AUTHORIZED",
                }
            ),
            stress_factors={"conflict_density": 0.9, "delay_depth": 1},
        )
    )

    # 9. Research workflow - temporary_blocker_recovery
    c_access = ConditionNode(
        condition_id="c_access", scope_id="res_scope", text="PDF is accessible."
    )
    b_citation = BeliefNode(
        belief_id="b_citation", proposition="Paper Y is cited.", source_evidence_ids=("e_init",)
    )
    dep_citation = DependencyEdge(
        edge_id="dep_citation", belief_id="b_citation", condition_id="c_access", inducer="writer"
    )
    ev_paywall = EvidenceNode(
        evidence_id="e_paywall", session_id="s_res_4", timestamp="2026-05-29T12:00:00Z",
        text="Access paywalled.", source_dataset="stale_dev", source_pointer="dev://14"
    )
    ev_preprint = EvidenceNode(
        evidence_id="e_preprint", session_id="s_res_5", timestamp="2026-05-29T15:00:00Z",
        text="Open access preprint found.", source_dataset="stale_dev", source_pointer="dev://15"
    )
    block_edge_res = EvidenceEdge(
        edge_id="edge_block_res", edge_type=EvidenceEdgeType.BLOCKS, evidence_id="e_paywall",
        target_kind="condition", target_id="c_access", verifier="crawler_a"
    )
    release_edge_res = EvidenceEdge(
        edge_id="edge_release_res", edge_type=EvidenceEdgeType.RELEASES, evidence_id="e_preprint",
        target_kind="condition", target_id="c_access", verifier="crawler_b"
    )

    sub_block_res = SubagentMemorySubmission(
        submission_id="sub_b_res_01", producer_id="agent_crawler_a", producer_role="crawler",
        parent_snapshot_id="snap_root", observed_at="2026-05-29T12:00:00Z",
        instance_id="diag_res_3", query_id="q_res_3", query="Cite Y?",
        evidence_context=(ev_init, ev_paywall), new_evidence_id="e_paywall",
        candidate_beliefs=(b_citation,), candidate_conditions_by_belief=(("b_citation", (c_access,)),),
        dependency_edges_by_belief=(("b_citation", (dep_citation,)),),
        proposal_batches=(EvidenceProposalBatch(edges=(block_edge_res,), model_call_trace_id="trace_res_b1"),)
    )
    sub_release_res = SubagentMemorySubmission(
        submission_id="sub_b_res_02", producer_id="agent_crawler_b", producer_role="crawler",
        parent_snapshot_id="snap_res_3", observed_at="2026-05-29T15:00:00Z",
        instance_id="diag_res_3", query_id="q_res_3", query="Cite Y?",
        evidence_context=(ev_init, ev_paywall, ev_preprint), new_evidence_id="e_preprint",
        candidate_beliefs=(b_citation,), candidate_conditions_by_belief=(("b_citation", (c_access,)),),
        dependency_edges_by_belief=(("b_citation", (dep_citation,)),),
        proposal_batches=(EvidenceProposalBatch(edges=(release_edge_res, block_edge_res), model_call_trace_id="trace_res_b2"),)
    )

    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_research_temporary_blocker",
            domain="research_workflow",
            failure_type="temporary_blocker_recovery",
            subagent_roles=("crawler", "crawler"),
            submissions=(sub_block_res, sub_release_res),
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_res_cite",
                    query="Is citation valid?",
                    expected_answer_or_action="yes",
                    relevant_belief_ids=("b_citation",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_citation": "AUTHORIZED",
                }
            ),
            stress_factors={"conflict_density": 0.2, "delay_depth": 2},
        )
    )

    # 10. Research workflow - stale_propagation
    ev_proof_fixed = EvidenceNode(
        evidence_id="e_proof_fixed", session_id="s_res_6", timestamp="2026-05-29T11:00:00Z",
        text="Proof has log N factor.", source_dataset="stale_dev", source_pointer="dev://16"
    )
    ev_old_ref = EvidenceNode(
        evidence_id="e_old_ref", session_id="s_res_7", timestamp="2026-05-29T09:00:00Z",
        text="Old book says N complexity.", source_dataset="stale_dev", source_pointer="dev://17"
    )
    b_result = BeliefNode(
        belief_id="b_result", proposition="Algorithm is O(N).", source_evidence_ids=("e_init",)
    )
    b_complexity_new = BeliefNode(
        belief_id="b_complexity_new", proposition="Algorithm is O(N log N).", source_evidence_ids=("e_proof_fixed",)
    )
    super_edge_stale = EvidenceEdge(
        edge_id="edge_super_stale", edge_type=EvidenceEdgeType.SUPERSEDES, evidence_id="e_proof_fixed",
        target_kind="belief", target_id="b_result", verifier="proof_analyst",
        replacement_belief_id="b_complexity_new"
    )
    reaffirm_edge_stale = EvidenceEdge(
        edge_id="edge_reaffirm_stale", edge_type=EvidenceEdgeType.REAFFIRMS, evidence_id="e_old_ref",
        target_kind="belief", target_id="b_result", verifier="textbook_reader"
    )

    sub_stale_res_1 = SubagentMemorySubmission(
        submission_id="sub_s_res_01", producer_id="agent_proof", producer_role="researcher",
        parent_snapshot_id="snap_root", observed_at="2026-05-29T11:00:00Z",
        instance_id="diag_res_4", query_id="q_res_4", query="What complexity?",
        evidence_context=(ev_init, ev_proof_fixed), new_evidence_id="e_proof_fixed",
        candidate_beliefs=(b_result,), candidate_replacement_beliefs=(b_complexity_new,),
        proposal_batches=(EvidenceProposalBatch(edges=(super_edge_stale,), model_call_trace_id="trace_res_s1"),)
    )
    sub_stale_res_2 = SubagentMemorySubmission(
        submission_id="sub_s_res_02", producer_id="agent_textbook", producer_role="researcher",
        parent_snapshot_id="snap_res_1", observed_at="2026-05-29T13:00:00Z",
        instance_id="diag_res_4", query_id="q_res_4", query="What complexity?",
        evidence_context=(ev_init, ev_proof_fixed, ev_old_ref), new_evidence_id="e_old_ref",
        candidate_beliefs=(b_result, b_complexity_new), proposal_batches=(
            EvidenceProposalBatch(edges=(reaffirm_edge_stale, super_edge_stale), model_call_trace_id="trace_res_s2"),
        )
    )
    sub_stale_res_query = SubagentMemorySubmission(
        submission_id="sub_s_res_query", producer_id="agent_reader", producer_role="reviewer",
        parent_snapshot_id="snap_res_2", observed_at="2026-05-29T13:05:00Z",
        instance_id="diag_res_4", query_id="q_res_4", query="What complexity?",
        evidence_context=(ev_init, ev_proof_fixed, ev_old_ref), new_evidence_id="e_proof_fixed",
        candidate_beliefs=(b_result, b_complexity_new), proposal_batches=(
            EvidenceProposalBatch(edges=(reaffirm_edge_stale, super_edge_stale), model_call_trace_id="trace_res_sq"),
        )
    )

    episodes.append(
        MultiAgentMemoryEpisode(
            episode_id="dev_research_stale_propagation",
            domain="research_workflow",
            failure_type="stale_propagation",
            subagent_roles=("researcher", "researcher", "reviewer"),
            submissions=(sub_stale_res_1, sub_stale_res_2, sub_stale_res_query),
            downstream_tasks=(
                DownstreamTask(
                    task_id="t_res_complexity",
                    query="Complexity class?",
                    expected_answer_or_action="O(N log N)",
                    relevant_belief_ids=("b_complexity_new",),
                ),
            ),
            gold_snapshot=GoldSnapshotExpectation(
                belief_statuses={
                    "b_result": "SUPERSEDED",
                    "b_complexity_new": "AUTHORIZED",
                }
            ),
            stress_factors={"conflict_density": 0.6, "delay_depth": 2},
        )
    )

    return episodes
