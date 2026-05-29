from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    EvidenceEdge,
    EvidenceEdgeType,
)
from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    FixedCandidateInputEpisode,
    FixedCandidateGoldRecord,
    MethodDecisionArtifact,
    MethodDecisionRecord,
    DownstreamTask,
    GoldSnapshotExpectation,
    TypedRevisionTarget,
    _proposal_batch_to_dict,
)
from retracemem import EvidenceProposalBatch

@dataclass
class _FCSubmissionBuilder:
    submission_id: str
    producer_id: str
    producer_role: str
    timestamp: str
    evidence_context: Tuple[EvidenceNode, ...]
    candidate_beliefs: Tuple[BeliefNode, ...]
    candidate_edges: Tuple[EvidenceEdge, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class _FCEpisodeBuilder:
    episode_id: str
    domain: str
    failure_type: str
    subagent_roles: Tuple[str, ...]
    submissions: Tuple[_FCSubmissionBuilder, ...]
    downstream_tasks: Tuple[DownstreamTask, ...]
    gold_snapshot: GoldSnapshotExpectation
    replay_decisions: Tuple[MethodDecisionRecord, ...] = ()
    stress_factors: Dict[str, Any] = field(default_factory=dict)
    protocol_mode: str = "fixed_candidate_revision"
    proposal_source: str = "hand_authored_development"
    split: str = "development_only"
    metadata: Dict[str, Any] = field(default_factory=dict)

def _convert_builder_to_fair_contracts(
    builder: _FCEpisodeBuilder
) -> Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord, MethodDecisionArtifact]:
    fair_submissions = []
    typed_proposal_batches_by_submission = []
    gold_targets = []
    
    for sub in builder.submissions:
        proposal_batches = ()
        if sub.candidate_edges:
            proposal_batches = (
                EvidenceProposalBatch(
                    edges=sub.candidate_edges,
                    model_call_trace_id=sub.metadata.get("model_call_trace_id"),
                    source_belief_id=sub.metadata.get("source_belief_id"),
                ),
            )
            typed_proposal_batches_by_submission.append((sub.submission_id, proposal_batches))
            
            for edge in sub.candidate_edges:
                gold_targets.append(
                    TypedRevisionTarget(
                        submission_id=sub.submission_id,
                        action_type=edge.edge_type.value if hasattr(edge.edge_type, "value") else str(edge.edge_type),
                        target_belief_id=edge.target_id if edge.target_kind == "belief" else None,
                        target_condition_id=edge.target_id if edge.target_kind == "condition" else None,
                        replacement_belief_id=edge.replacement_belief_id,
                        rationale=edge.rationale or "",
                        evidence_ids=(edge.evidence_id,),
                    )
                )

        replacement_beliefs = []
        replacement_ids = set()
        for edge in sub.candidate_edges:
            if edge.edge_type == EvidenceEdgeType.SUPERSEDES and edge.replacement_belief_id:
                for b in sub.candidate_beliefs:
                    if b.belief_id == edge.replacement_belief_id:
                        replacement_beliefs.append(b)
                        replacement_ids.add(b.belief_id)
                        break

        normal_candidate_beliefs = [b for b in sub.candidate_beliefs if b.belief_id not in replacement_ids]
        
        fair_sub = FixedCandidateSubmission(
            submission_id=sub.submission_id,
            producer_id=sub.producer_id,
            producer_role=sub.producer_role,
            task_id=sub.metadata.get("task_id"),
            parent_snapshot_id=sub.metadata.get("parent_snapshot_id", "snapshot_init"),
            observed_at=sub.timestamp,
            instance_id=sub.metadata.get("instance_id", "instance_0"),
            query_id=sub.metadata.get("query_id", "query_0"),
            query=sub.metadata.get("query", "Default query"),
            evidence_context=sub.evidence_context,
            new_evidence_id=sub.evidence_context[-1].evidence_id if sub.evidence_context else "ev_unknown",
            candidate_beliefs=tuple(normal_candidate_beliefs),
            candidate_replacement_beliefs=tuple(replacement_beliefs),
            candidate_conditions_by_belief=(),
            dependency_edges_by_belief=(),
            metadata=sub.metadata,
        )
        fair_submissions.append(fair_sub)

    direct_verdicts_by_submission = ()
    if builder.submissions and builder.replay_decisions:
        last_sub_id = builder.submissions[-1].submission_id
        direct_verdicts_by_submission = ((last_sub_id, builder.replay_decisions),)

    artifact = MethodDecisionArtifact(
        episode_id=builder.episode_id,
        method_name="DirectJudge_Replay",
        protocol_mode="oracle_edge_replay",
        proposal_source="oracle_replay",
        backbone_model=None,
        typed_proposal_batches_by_submission=tuple(typed_proposal_batches_by_submission),
        direct_verdicts_by_submission=direct_verdicts_by_submission,
        scientific_status="pipeline_validation_only",
    )

    input_episode = FixedCandidateInputEpisode(
        episode_id=builder.episode_id,
        domain=builder.domain,
        failure_type_public_or_controlled=builder.failure_type,
        subagent_roles=builder.subagent_roles,
        submissions=tuple(fair_submissions),
        downstream_tasks=builder.downstream_tasks,
        stress_factors=builder.stress_factors,
        split=builder.split,
        protocol_mode=builder.protocol_mode,
        proposal_source=builder.proposal_source,
        metadata=builder.metadata,
    )

    gold_record = FixedCandidateGoldRecord(
        episode_id=builder.episode_id,
        gold_snapshot=builder.gold_snapshot,
        gold_typed_targets=tuple(gold_targets),
        failure_type=builder.failure_type,
        metadata=builder.metadata,
    )

    return input_episode, gold_record, artifact

def get_fc_dev_episodes() -> List[Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord, MethodDecisionArtifact]]:
    """Return 14 fixed-candidate development episodes as (input_episode, gold_record, decision_artifact) tuples.

    Coverage matrix:
        7 failure_types × 2 domains = 14 episodes

    Failure types:
        cross_agent_conflict, direct_supersession, stale_propagation,
        scope_expansion, temporary_blocker_recovery, duplicate_evidence,
        ambiguous_update

    Domains:
        software_engineering, research_workflow
    """
    episodes: List[_FCEpisodeBuilder] = []

    # ========================================================================
    # Domain: software_engineering
    # ========================================================================

    # SE-1: cross_agent_conflict
    # Two agents report conflicting deployment targets. Agent A says "us-east-1",
    # Agent B (later) says "eu-west-1" and supersedes A.
    ev_se1_a = EvidenceNode(
        evidence_id="ev_se1_a", session_id="se1_s0", timestamp="2026-01-10T10:00:00Z",
        text="Deploy target is us-east-1.", source_dataset="fc_dev", source_pointer="fc://se1a",
    )
    ev_se1_b = EvidenceNode(
        evidence_id="ev_se1_b", session_id="se1_s1", timestamp="2026-01-10T11:00:00Z",
        text="Deploy target changed to eu-west-1.", source_dataset="fc_dev", source_pointer="fc://se1b",
    )
    b_se1_east = BeliefNode(
        belief_id="b_se1_east", proposition="Deploy target is us-east-1.",
        source_evidence_ids=("ev_se1_a",),
    )
    b_se1_west = BeliefNode(
        belief_id="b_se1_west", proposition="Deploy target is eu-west-1.",
        source_evidence_ids=("ev_se1_b",),
    )
    edge_se1_super = EvidenceEdge(
        edge_id="edge_se1_super", edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_se1_b", target_kind="belief", target_id="b_se1_east",
        verifier="deploy_agent_b", replacement_belief_id="b_se1_west",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_se_cross_agent_conflict",
        domain="software_engineering",
        failure_type="cross_agent_conflict",
        subagent_roles=("deploy_agent_a", "deploy_agent_b"),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_se1_sub1", producer_id="deploy_agent_a",
                producer_role="deploy_agent", timestamp="2026-01-10T10:00:00Z",
                evidence_context=(ev_se1_a,),
                candidate_beliefs=(b_se1_east,), candidate_edges=(),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_se1_sub2", producer_id="deploy_agent_b",
                producer_role="deploy_agent", timestamp="2026-01-10T11:00:00Z",
                evidence_context=(ev_se1_a, ev_se1_b),
                candidate_beliefs=(b_se1_east, b_se1_west),
                candidate_edges=(edge_se1_super,),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_se1", query="What is the deploy target?",
                expected_answer_or_action="eu-west-1",
                relevant_belief_ids=("b_se1_west",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_se1_east": "SUPERSEDED", "b_se1_west": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_se1_east", decision="SUPERSEDE",
                                 replacement_belief_id="b_se1_west"),
            MethodDecisionRecord(belief_id="b_se1_west", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.9, "delay_depth": 1},
    ))

    # SE-2: direct_supersession
    # API version upgraded from v2 to v3.
    ev_se2_v2 = EvidenceNode(
        evidence_id="ev_se2_v2", session_id="se2_s0", timestamp="2026-01-10T09:00:00Z",
        text="API runs on v2.", source_dataset="fc_dev", source_pointer="fc://se2a",
    )
    ev_se2_v3 = EvidenceNode(
        evidence_id="ev_se2_v3", session_id="se2_s1", timestamp="2026-01-10T12:00:00Z",
        text="API upgraded to v3.", source_dataset="fc_dev", source_pointer="fc://se2b",
    )
    b_se2_v2 = BeliefNode(
        belief_id="b_se2_v2", proposition="API version is v2.",
        source_evidence_ids=("ev_se2_v2",),
    )
    b_se2_v3 = BeliefNode(
        belief_id="b_se2_v3", proposition="API version is v3.",
        source_evidence_ids=("ev_se2_v3",),
    )
    edge_se2_super = EvidenceEdge(
        edge_id="edge_se2_super", edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_se2_v3", target_kind="belief", target_id="b_se2_v2",
        verifier="api_monitor", replacement_belief_id="b_se2_v3",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_se_direct_supersession",
        domain="software_engineering",
        failure_type="direct_supersession",
        subagent_roles=("api_monitor",),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_se2_sub1", producer_id="api_monitor",
                producer_role="api_monitor", timestamp="2026-01-10T09:00:00Z",
                evidence_context=(ev_se2_v2,),
                candidate_beliefs=(b_se2_v2,), candidate_edges=(),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_se2_sub2", producer_id="api_monitor",
                producer_role="api_monitor", timestamp="2026-01-10T12:00:00Z",
                evidence_context=(ev_se2_v2, ev_se2_v3),
                candidate_beliefs=(b_se2_v2, b_se2_v3),
                candidate_edges=(edge_se2_super,),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_se2", query="What API version is running?",
                expected_answer_or_action="v3",
                relevant_belief_ids=("b_se2_v3",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_se2_v2": "SUPERSEDED", "b_se2_v3": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_se2_v2", decision="SUPERSEDE",
                                 replacement_belief_id="b_se2_v3"),
            MethodDecisionRecord(belief_id="b_se2_v3", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.4, "delay_depth": 0},
    ))

    # SE-3: stale_propagation
    # A stale CI report reaffirms an old build status that was already superseded.
    ev_se3_pass = EvidenceNode(
        evidence_id="ev_se3_pass", session_id="se3_s0", timestamp="2026-01-10T08:00:00Z",
        text="CI build passed at commit abc123.", source_dataset="fc_dev", source_pointer="fc://se3a",
    )
    ev_se3_fail = EvidenceNode(
        evidence_id="ev_se3_fail", session_id="se3_s1", timestamp="2026-01-10T10:00:00Z",
        text="CI build failed at commit def456.", source_dataset="fc_dev", source_pointer="fc://se3b",
    )
    ev_se3_stale = EvidenceNode(
        evidence_id="ev_se3_stale", session_id="se3_s2", timestamp="2026-01-10T07:00:00Z",
        text="CI build passed at commit abc123 (cached).", source_dataset="fc_dev", source_pointer="fc://se3c",
    )
    b_se3_pass = BeliefNode(
        belief_id="b_se3_pass", proposition="CI build is passing.",
        source_evidence_ids=("ev_se3_pass",),
    )
    b_se3_fail = BeliefNode(
        belief_id="b_se3_fail", proposition="CI build is failing.",
        source_evidence_ids=("ev_se3_fail",),
    )
    edge_se3_super = EvidenceEdge(
        edge_id="edge_se3_super", edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_se3_fail", target_kind="belief", target_id="b_se3_pass",
        verifier="ci_monitor", replacement_belief_id="b_se3_fail",
    )
    edge_se3_reaffirm = EvidenceEdge(
        edge_id="edge_se3_reaffirm", edge_type=EvidenceEdgeType.REAFFIRMS,
        evidence_id="ev_se3_stale", target_kind="belief", target_id="b_se3_pass",
        verifier="ci_cache_agent",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_se_stale_propagation",
        domain="software_engineering",
        failure_type="stale_propagation",
        subagent_roles=("ci_monitor", "ci_cache_agent"),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_se3_sub1", producer_id="ci_monitor",
                producer_role="ci_monitor", timestamp="2026-01-10T10:00:00Z",
                evidence_context=(ev_se3_pass, ev_se3_fail),
                candidate_beliefs=(b_se3_pass, b_se3_fail),
                candidate_edges=(edge_se3_super,),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_se3_sub2", producer_id="ci_cache_agent",
                producer_role="ci_cache_agent", timestamp="2026-01-10T10:30:00Z",
                evidence_context=(ev_se3_pass, ev_se3_fail, ev_se3_stale),
                candidate_beliefs=(b_se3_pass, b_se3_fail),
                candidate_edges=(edge_se3_super, edge_se3_reaffirm),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_se3", query="Is the CI build passing?",
                expected_answer_or_action="No, build is failing.",
                relevant_belief_ids=("b_se3_fail",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_se3_pass": "SUPERSEDED", "b_se3_fail": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_se3_pass", decision="SUPERSEDE",
                                 replacement_belief_id="b_se3_fail"),
            MethodDecisionRecord(belief_id="b_se3_fail", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.5, "delay_depth": 2},
    ))

    # SE-4: scope_expansion
    # A coding style preference is added without evidence of change.
    ev_se4_style = EvidenceNode(
        evidence_id="ev_se4_style", session_id="se4_s0", timestamp="2026-01-10T09:00:00Z",
        text="Team uses black formatter.", source_dataset="fc_dev", source_pointer="fc://se4a",
    )
    b_se4_style = BeliefNode(
        belief_id="b_se4_style", proposition="Team uses black formatter.",
        source_evidence_ids=("ev_se4_style",),
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_se_scope_expansion",
        domain="software_engineering",
        failure_type="scope_expansion",
        subagent_roles=("style_checker",),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_se4_sub1", producer_id="style_checker",
                producer_role="style_checker", timestamp="2026-01-10T09:00:00Z",
                evidence_context=(ev_se4_style,),
                candidate_beliefs=(b_se4_style,), candidate_edges=(),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_se4", query="What formatter does the team use?",
                expected_answer_or_action="black",
                relevant_belief_ids=("b_se4_style",),
                protected_belief_ids=("b_se4_style",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_se4_style": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_se4_style", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.0, "delay_depth": 0},
    ))

    # SE-5: temporary_blocker_recovery
    # Database is temporarily down, then recovers.
    ev_se5_down = EvidenceNode(
        evidence_id="ev_se5_down", session_id="se5_s0", timestamp="2026-01-10T14:00:00Z",
        text="Database is unreachable.", source_dataset="fc_dev", source_pointer="fc://se5a",
    )
    ev_se5_up = EvidenceNode(
        evidence_id="ev_se5_up", session_id="se5_s1", timestamp="2026-01-10T15:00:00Z",
        text="Database connectivity restored.", source_dataset="fc_dev", source_pointer="fc://se5b",
    )
    b_se5_avail = BeliefNode(
        belief_id="b_se5_avail", proposition="Database is available.",
        source_evidence_ids=("ev_se5_up",),
    )
    edge_se5_block = EvidenceEdge(
        edge_id="edge_se5_block", edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_se5_down", target_kind="condition", target_id="c_se5_db",
        verifier="infra_monitor",
    )
    edge_se5_release = EvidenceEdge(
        edge_id="edge_se5_release", edge_type=EvidenceEdgeType.RELEASES,
        evidence_id="ev_se5_up", target_kind="condition", target_id="c_se5_db",
        verifier="infra_monitor",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_se_temporary_blocker_recovery",
        domain="software_engineering",
        failure_type="temporary_blocker_recovery",
        subagent_roles=("infra_monitor", "infra_monitor"),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_se5_sub1", producer_id="infra_monitor",
                producer_role="infra_monitor", timestamp="2026-01-10T14:00:00Z",
                evidence_context=(ev_se5_down,),
                candidate_beliefs=(),
                candidate_edges=(edge_se5_block,),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_se5_sub2", producer_id="infra_monitor",
                producer_role="infra_monitor", timestamp="2026-01-10T15:00:00Z",
                evidence_context=(ev_se5_down, ev_se5_up),
                candidate_beliefs=(b_se5_avail,),
                candidate_edges=(edge_se5_release, edge_se5_block),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_se5", query="Is the database available?",
                expected_answer_or_action="yes",
                relevant_belief_ids=("b_se5_avail",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_se5_avail": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_se5_avail", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.2, "delay_depth": 1},
    ))

    # SE-6: duplicate_evidence
    # Two agents observe the same log entry about memory usage.
    ev_se6_mem = EvidenceNode(
        evidence_id="ev_se6_mem", session_id="se6_s0", timestamp="2026-01-10T11:00:00Z",
        text="Memory usage is 85%.", source_dataset="fc_dev", source_pointer="fc://se6a",
    )
    b_se6_mem = BeliefNode(
        belief_id="b_se6_mem", proposition="Memory usage is at 85%.",
        source_evidence_ids=("ev_se6_mem",),
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_se_duplicate_evidence",
        domain="software_engineering",
        failure_type="duplicate_evidence",
        subagent_roles=("perf_monitor_a", "perf_monitor_b"),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_se6_sub1", producer_id="perf_monitor_a",
                producer_role="perf_monitor", timestamp="2026-01-10T11:00:00Z",
                evidence_context=(ev_se6_mem,),
                candidate_beliefs=(b_se6_mem,), candidate_edges=(),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_se6_sub2", producer_id="perf_monitor_b",
                producer_role="perf_monitor", timestamp="2026-01-10T11:05:00Z",
                evidence_context=(ev_se6_mem,),
                candidate_beliefs=(b_se6_mem,), candidate_edges=(),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_se6", query="What is current memory usage?",
                expected_answer_or_action="85%",
                relevant_belief_ids=("b_se6_mem",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_se6_mem": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_se6_mem", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.1, "delay_depth": 0},
    ))

    # SE-7: ambiguous_update
    # Unclear whether a config change was intentional.
    ev_se7_config = EvidenceNode(
        evidence_id="ev_se7_config", session_id="se7_s0", timestamp="2026-01-10T13:00:00Z",
        text="Timeout changed from 30s to 5s in config.", source_dataset="fc_dev", source_pointer="fc://se7a",
    )
    b_se7_timeout = BeliefNode(
        belief_id="b_se7_timeout", proposition="Service timeout is 5 seconds.",
        source_evidence_ids=("ev_se7_config",),
    )
    edge_se7_uncertain = EvidenceEdge(
        edge_id="edge_se7_uncertain", edge_type=EvidenceEdgeType.UNCERTAIN,
        evidence_id="ev_se7_config", target_kind="belief", target_id="b_se7_timeout",
        verifier="config_auditor",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_se_ambiguous_update",
        domain="software_engineering",
        failure_type="ambiguous_update",
        subagent_roles=("config_auditor",),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_se7_sub1", producer_id="config_auditor",
                producer_role="config_auditor", timestamp="2026-01-10T13:00:00Z",
                evidence_context=(ev_se7_config,),
                candidate_beliefs=(b_se7_timeout,),
                candidate_edges=(edge_se7_uncertain,),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_se7", query="Is the timeout change intentional?",
                expected_answer_or_action="unresolved",
                relevant_belief_ids=("b_se7_timeout",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_se7_timeout": "UNRESOLVED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_se7_timeout", decision="DEFER"),
        ),
        stress_factors={"conflict_density": 0.3, "delay_depth": 0},
    ))

    # ========================================================================
    # Domain: research_workflow
    # ========================================================================

    # RW-1: cross_agent_conflict
    # Two literature crawlers report conflicting publication years.
    ev_rw1_2024 = EvidenceNode(
        evidence_id="ev_rw1_2024", session_id="rw1_s0", timestamp="2026-02-01T10:00:00Z",
        text="Paper Z published in 2024.", source_dataset="fc_dev", source_pointer="fc://rw1a",
    )
    ev_rw1_2025 = EvidenceNode(
        evidence_id="ev_rw1_2025", session_id="rw1_s1", timestamp="2026-02-01T11:00:00Z",
        text="Paper Z camera-ready published 2025.", source_dataset="fc_dev", source_pointer="fc://rw1b",
    )
    b_rw1_2024 = BeliefNode(
        belief_id="b_rw1_2024", proposition="Paper Z was published in 2024.",
        source_evidence_ids=("ev_rw1_2024",),
    )
    b_rw1_2025 = BeliefNode(
        belief_id="b_rw1_2025", proposition="Paper Z was published in 2025.",
        source_evidence_ids=("ev_rw1_2025",),
    )
    edge_rw1_super = EvidenceEdge(
        edge_id="edge_rw1_super", edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_rw1_2025", target_kind="belief", target_id="b_rw1_2024",
        verifier="crawler_b", replacement_belief_id="b_rw1_2025",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_rw_cross_agent_conflict",
        domain="research_workflow",
        failure_type="cross_agent_conflict",
        subagent_roles=("crawler_a", "crawler_b"),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_rw1_sub1", producer_id="crawler_a",
                producer_role="crawler", timestamp="2026-02-01T10:00:00Z",
                evidence_context=(ev_rw1_2024,),
                candidate_beliefs=(b_rw1_2024,), candidate_edges=(),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_rw1_sub2", producer_id="crawler_b",
                producer_role="crawler", timestamp="2026-02-01T11:00:00Z",
                evidence_context=(ev_rw1_2024, ev_rw1_2025),
                candidate_beliefs=(b_rw1_2024, b_rw1_2025),
                candidate_edges=(edge_rw1_super,),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_rw1", query="When was Paper Z published?",
                expected_answer_or_action="2025",
                relevant_belief_ids=("b_rw1_2025",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_rw1_2024": "SUPERSEDED", "b_rw1_2025": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_rw1_2024", decision="SUPERSEDE",
                                 replacement_belief_id="b_rw1_2025"),
            MethodDecisionRecord(belief_id="b_rw1_2025", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.9, "delay_depth": 1},
    ))

    # RW-2: direct_supersession
    # Dataset size updated from 10K to 50K samples.
    ev_rw2_10k = EvidenceNode(
        evidence_id="ev_rw2_10k", session_id="rw2_s0", timestamp="2026-02-01T09:00:00Z",
        text="Dataset has 10K samples.", source_dataset="fc_dev", source_pointer="fc://rw2a",
    )
    ev_rw2_50k = EvidenceNode(
        evidence_id="ev_rw2_50k", session_id="rw2_s1", timestamp="2026-02-01T14:00:00Z",
        text="Dataset expanded to 50K samples.", source_dataset="fc_dev", source_pointer="fc://rw2b",
    )
    b_rw2_10k = BeliefNode(
        belief_id="b_rw2_10k", proposition="Dataset has 10K samples.",
        source_evidence_ids=("ev_rw2_10k",),
    )
    b_rw2_50k = BeliefNode(
        belief_id="b_rw2_50k", proposition="Dataset has 50K samples.",
        source_evidence_ids=("ev_rw2_50k",),
    )
    edge_rw2_super = EvidenceEdge(
        edge_id="edge_rw2_super", edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_rw2_50k", target_kind="belief", target_id="b_rw2_10k",
        verifier="data_curator", replacement_belief_id="b_rw2_50k",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_rw_direct_supersession",
        domain="research_workflow",
        failure_type="direct_supersession",
        subagent_roles=("data_curator",),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_rw2_sub1", producer_id="data_curator",
                producer_role="data_curator", timestamp="2026-02-01T09:00:00Z",
                evidence_context=(ev_rw2_10k,),
                candidate_beliefs=(b_rw2_10k,), candidate_edges=(),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_rw2_sub2", producer_id="data_curator",
                producer_role="data_curator", timestamp="2026-02-01T14:00:00Z",
                evidence_context=(ev_rw2_10k, ev_rw2_50k),
                candidate_beliefs=(b_rw2_10k, b_rw2_50k),
                candidate_edges=(edge_rw2_super,),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_rw2", query="How many samples in the dataset?",
                expected_answer_or_action="50K",
                relevant_belief_ids=("b_rw2_50k",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_rw2_10k": "SUPERSEDED", "b_rw2_50k": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_rw2_10k", decision="SUPERSEDE",
                                 replacement_belief_id="b_rw2_50k"),
            MethodDecisionRecord(belief_id="b_rw2_50k", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.4, "delay_depth": 0},
    ))

    # RW-3: stale_propagation
    # Stale arxiv abstract reaffirms old SOTA that was already superseded.
    ev_rw3_old = EvidenceNode(
        evidence_id="ev_rw3_old", session_id="rw3_s0", timestamp="2026-02-01T08:00:00Z",
        text="SOTA accuracy is 91.2%.", source_dataset="fc_dev", source_pointer="fc://rw3a",
    )
    ev_rw3_new = EvidenceNode(
        evidence_id="ev_rw3_new", session_id="rw3_s1", timestamp="2026-02-01T12:00:00Z",
        text="New method achieves 94.5% accuracy.", source_dataset="fc_dev", source_pointer="fc://rw3b",
    )
    ev_rw3_stale = EvidenceNode(
        evidence_id="ev_rw3_stale", session_id="rw3_s2", timestamp="2026-02-01T07:00:00Z",
        text="Abstract from 2024 claims 91.2% SOTA.", source_dataset="fc_dev", source_pointer="fc://rw3c",
    )
    b_rw3_old = BeliefNode(
        belief_id="b_rw3_old", proposition="SOTA accuracy is 91.2%.",
        source_evidence_ids=("ev_rw3_old",),
    )
    b_rw3_new = BeliefNode(
        belief_id="b_rw3_new", proposition="SOTA accuracy is 94.5%.",
        source_evidence_ids=("ev_rw3_new",),
    )
    edge_rw3_super = EvidenceEdge(
        edge_id="edge_rw3_super", edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_rw3_new", target_kind="belief", target_id="b_rw3_old",
        verifier="benchmark_reader", replacement_belief_id="b_rw3_new",
    )
    edge_rw3_reaffirm = EvidenceEdge(
        edge_id="edge_rw3_reaffirm", edge_type=EvidenceEdgeType.REAFFIRMS,
        evidence_id="ev_rw3_stale", target_kind="belief", target_id="b_rw3_old",
        verifier="abstract_crawler",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_rw_stale_propagation",
        domain="research_workflow",
        failure_type="stale_propagation",
        subagent_roles=("benchmark_reader", "abstract_crawler"),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_rw3_sub1", producer_id="benchmark_reader",
                producer_role="benchmark_reader", timestamp="2026-02-01T12:00:00Z",
                evidence_context=(ev_rw3_old, ev_rw3_new),
                candidate_beliefs=(b_rw3_old, b_rw3_new),
                candidate_edges=(edge_rw3_super,),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_rw3_sub2", producer_id="abstract_crawler",
                producer_role="abstract_crawler", timestamp="2026-02-01T12:30:00Z",
                evidence_context=(ev_rw3_old, ev_rw3_new, ev_rw3_stale),
                candidate_beliefs=(b_rw3_old, b_rw3_new),
                candidate_edges=(edge_rw3_super, edge_rw3_reaffirm),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_rw3", query="What is current SOTA accuracy?",
                expected_answer_or_action="94.5%",
                relevant_belief_ids=("b_rw3_new",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_rw3_old": "SUPERSEDED", "b_rw3_new": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_rw3_old", decision="SUPERSEDE",
                                 replacement_belief_id="b_rw3_new"),
            MethodDecisionRecord(belief_id="b_rw3_new", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.6, "delay_depth": 2},
    ))

    # RW-4: scope_expansion
    # A known research methodology is preserved without change.
    ev_rw4_method = EvidenceNode(
        evidence_id="ev_rw4_method", session_id="rw4_s0", timestamp="2026-02-01T09:00:00Z",
        text="Study uses stratified 5-fold cross-validation.", source_dataset="fc_dev", source_pointer="fc://rw4a",
    )
    b_rw4_method = BeliefNode(
        belief_id="b_rw4_method", proposition="Evaluation uses stratified 5-fold CV.",
        source_evidence_ids=("ev_rw4_method",),
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_rw_scope_expansion",
        domain="research_workflow",
        failure_type="scope_expansion",
        subagent_roles=("methodology_reader",),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_rw4_sub1", producer_id="methodology_reader",
                producer_role="methodology_reader", timestamp="2026-02-01T09:00:00Z",
                evidence_context=(ev_rw4_method,),
                candidate_beliefs=(b_rw4_method,), candidate_edges=(),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_rw4", query="What evaluation methodology is used?",
                expected_answer_or_action="stratified 5-fold CV",
                relevant_belief_ids=("b_rw4_method",),
                protected_belief_ids=("b_rw4_method",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_rw4_method": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_rw4_method", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.0, "delay_depth": 0},
    ))

    # RW-5: temporary_blocker_recovery
    # Paper PDF behind paywall, then open-access preprint found.
    ev_rw5_paywall = EvidenceNode(
        evidence_id="ev_rw5_paywall", session_id="rw5_s0", timestamp="2026-02-01T13:00:00Z",
        text="Paper behind publisher paywall.", source_dataset="fc_dev", source_pointer="fc://rw5a",
    )
    ev_rw5_preprint = EvidenceNode(
        evidence_id="ev_rw5_preprint", session_id="rw5_s1", timestamp="2026-02-01T16:00:00Z",
        text="Open-access preprint found on arxiv.", source_dataset="fc_dev", source_pointer="fc://rw5b",
    )
    b_rw5_cited = BeliefNode(
        belief_id="b_rw5_cited", proposition="Paper Q is citable.",
        source_evidence_ids=("ev_rw5_preprint",),
    )
    edge_rw5_block = EvidenceEdge(
        edge_id="edge_rw5_block", edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_rw5_paywall", target_kind="condition", target_id="c_rw5_access",
        verifier="crawl_agent",
    )
    edge_rw5_release = EvidenceEdge(
        edge_id="edge_rw5_release", edge_type=EvidenceEdgeType.RELEASES,
        evidence_id="ev_rw5_preprint", target_kind="condition", target_id="c_rw5_access",
        verifier="crawl_agent",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_rw_temporary_blocker_recovery",
        domain="research_workflow",
        failure_type="temporary_blocker_recovery",
        subagent_roles=("crawl_agent", "crawl_agent"),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_rw5_sub1", producer_id="crawl_agent",
                producer_role="crawl_agent", timestamp="2026-02-01T13:00:00Z",
                evidence_context=(ev_rw5_paywall,),
                candidate_beliefs=(),
                candidate_edges=(edge_rw5_block,),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_rw5_sub2", producer_id="crawl_agent",
                producer_role="crawl_agent", timestamp="2026-02-01T16:00:00Z",
                evidence_context=(ev_rw5_paywall, ev_rw5_preprint),
                candidate_beliefs=(b_rw5_cited,),
                candidate_edges=(edge_rw5_release, edge_rw5_block),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_rw5", query="Can we cite Paper Q?",
                expected_answer_or_action="yes",
                relevant_belief_ids=("b_rw5_cited",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_rw5_cited": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_rw5_cited", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.2, "delay_depth": 2},
    ))

    # RW-6: duplicate_evidence
    # Two crawlers find the same citation count.
    ev_rw6_cite = EvidenceNode(
        evidence_id="ev_rw6_cite", session_id="rw6_s0", timestamp="2026-02-01T11:00:00Z",
        text="Paper has 142 citations.", source_dataset="fc_dev", source_pointer="fc://rw6a",
    )
    b_rw6_cite = BeliefNode(
        belief_id="b_rw6_cite", proposition="Paper has 142 citations.",
        source_evidence_ids=("ev_rw6_cite",),
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_rw_duplicate_evidence",
        domain="research_workflow",
        failure_type="duplicate_evidence",
        subagent_roles=("citation_crawler_a", "citation_crawler_b"),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_rw6_sub1", producer_id="citation_crawler_a",
                producer_role="citation_crawler", timestamp="2026-02-01T11:00:00Z",
                evidence_context=(ev_rw6_cite,),
                candidate_beliefs=(b_rw6_cite,), candidate_edges=(),
            ),
            _FCSubmissionBuilder(
                submission_id="fc_rw6_sub2", producer_id="citation_crawler_b",
                producer_role="citation_crawler", timestamp="2026-02-01T11:05:00Z",
                evidence_context=(ev_rw6_cite,),
                candidate_beliefs=(b_rw6_cite,), candidate_edges=(),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_rw6", query="How many citations does the paper have?",
                expected_answer_or_action="142",
                relevant_belief_ids=("b_rw6_cite",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_rw6_cite": "AUTHORIZED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_rw6_cite", decision="AUTHORIZE"),
        ),
        stress_factors={"conflict_density": 0.1, "delay_depth": 0},
    ))

    # RW-7: ambiguous_update
    # Unclear if a reproducibility claim is valid.
    ev_rw7_repro = EvidenceNode(
        evidence_id="ev_rw7_repro", session_id="rw7_s0", timestamp="2026-02-01T15:00:00Z",
        text="Partial reproduction: 3 of 5 experiments match.", source_dataset="fc_dev", source_pointer="fc://rw7a",
    )
    b_rw7_reproducible = BeliefNode(
        belief_id="b_rw7_reproducible", proposition="Results are reproducible.",
        source_evidence_ids=("ev_rw7_repro",),
    )
    edge_rw7_uncertain = EvidenceEdge(
        edge_id="edge_rw7_uncertain", edge_type=EvidenceEdgeType.UNCERTAIN,
        evidence_id="ev_rw7_repro", target_kind="belief", target_id="b_rw7_reproducible",
        verifier="repro_checker",
    )

    episodes.append(_FCEpisodeBuilder(
        episode_id="fc_rw_ambiguous_update",
        domain="research_workflow",
        failure_type="ambiguous_update",
        subagent_roles=("repro_checker",),
        submissions=(
            _FCSubmissionBuilder(
                submission_id="fc_rw7_sub1", producer_id="repro_checker",
                producer_role="repro_checker", timestamp="2026-02-01T15:00:00Z",
                evidence_context=(ev_rw7_repro,),
                candidate_beliefs=(b_rw7_reproducible,),
                candidate_edges=(edge_rw7_uncertain,),
            ),
        ),
        downstream_tasks=(
            DownstreamTask(
                task_id="t_rw7", query="Are results reproducible?",
                expected_answer_or_action="unresolved",
                relevant_belief_ids=("b_rw7_reproducible",),
            ),
        ),
        gold_snapshot=GoldSnapshotExpectation(
            belief_statuses={"b_rw7_reproducible": "UNRESOLVED"},
        ),
        replay_decisions=(
            MethodDecisionRecord(belief_id="b_rw7_reproducible", decision="DEFER"),
        ),
        stress_factors={"conflict_density": 0.3, "delay_depth": 0},
    ))

    return [_convert_builder_to_fair_contracts(ep) for ep in episodes]
