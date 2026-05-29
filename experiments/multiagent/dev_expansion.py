from __future__ import annotations

from typing import List, Tuple, Dict, Any
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
)
from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    FixedCandidateInputEpisode,
    FixedCandidateGoldRecord,
    DownstreamTask,
    GoldSnapshotExpectation,
    TypedRevisionTarget,
)

DOMAINS = ["software_engineering", "research_workflow"]
FAILURE_TYPES = [
    "direct_supersession",
    "stale_propagation",
    "scope_expansion",
    "cross_agent_conflict",
    "temporary_blocker_recovery",
    "duplicate_evidence",
    "ambiguous_update",
]

def generate_expanded_episodes() -> List[Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]]:
    episodes = []
    
    for domain in DOMAINS:
        for f_type in FAILURE_TYPES:
            for v in range(1, 6): # 5 variants
                episode_id = f"ep_expansion_{domain}_{f_type}_v{v}"
                
                # Domain-specific text helpers
                if domain == "software_engineering":
                    query = f"Verify component security check for patch {v}"
                    b_init_text = f"Patch {v} uses basic SHA-1 verification."
                    b_rev_text = f"Patch {v} uses SHA-256 for secure verification."
                    ev_init_text = f"Patch {v} documentation specifies SHA-1."
                    ev_rev_text = f"Commit logs for patch {v} indicate SHA-256 migration."
                    c_text = f"Patch {v} signature verifier is loaded."
                    c_blocked_text = f"Signature verification module for patch {v} is disabled."
                else: # research_workflow
                    query = f"Check medical study results for trial {v}"
                    b_init_text = f"Clinical Trial {v} records show safety efficacy."
                    b_rev_text = f"Clinical Trial {v} has severe side-effects detected."
                    ev_init_text = f"Trial {v} clinical trial log 1 reports no incidents."
                    ev_rev_text = f"Trial {v} adverse event report logs severe interaction."
                    c_text = f"Trial {v} study protocol approved by IRB."
                    c_blocked_text = f"IRB ethical approval for trial {v} is temporarily suspended."

                # IDs
                sub1_id = f"sub_{episode_id}_1"
                sub2_id = f"sub_{episode_id}_2"
                ev1_id = f"ev_{episode_id}_1"
                ev2_id = f"ev_{episode_id}_2"
                b1_id = f"b_{episode_id}_1"
                b2_id = f"b_{episode_id}_2"
                c1_id = f"c_{episode_id}_1"
                
                # Nodes
                ev1 = EvidenceNode(
                    evidence_id=ev1_id,
                    session_id=f"sess_{episode_id}_1",
                    timestamp="2026-05-30T00:00:00Z",
                    text=ev_init_text,
                    source_dataset="dev_expansion",
                    source_pointer=f"file:///src/{domain}/v{v}/v1.txt",
                )
                
                ev2 = EvidenceNode(
                    evidence_id=ev2_id,
                    session_id=f"sess_{episode_id}_2",
                    timestamp="2026-05-30T01:00:00Z",
                    text=ev_rev_text,
                    source_dataset="dev_expansion",
                    source_pointer=f"file:///src/{domain}/v{v}/v2.txt",
                )
                
                b1 = BeliefNode(
                    belief_id=b1_id,
                    proposition=b_init_text,
                    source_evidence_ids=(ev1_id,),
                )
                
                b2 = BeliefNode(
                    belief_id=b2_id,
                    proposition=b_rev_text,
                    source_evidence_ids=(ev2_id,),
                )
                
                cond1 = ConditionNode(
                    condition_id=c1_id,
                    scope_id=f"scope_{episode_id}_1",
                    text=c_text if f_type != "temporary_blocker_recovery" else c_blocked_text,
                )

                # Set up structures and targets based on failure_type
                sub1_candidate_beliefs = (b1,)
                sub1_candidate_replacement_beliefs = ()
                sub1_candidate_conditions_by_belief = ()
                sub1_dependency_edges_by_belief = ()
                
                # Submission 2 is the main Stage C decision point
                sub2_candidate_beliefs = (b1,)
                sub2_candidate_replacement_beliefs = ()
                sub2_candidate_conditions_by_belief = ()
                sub2_dependency_edges_by_belief = ()
                
                targets = []
                gold_statuses = {b1_id: "AUTHORIZED"}

                if f_type == "direct_supersession":
                    sub2_candidate_replacement_beliefs = (b2,)
                    targets = [
                        TypedRevisionTarget(
                            submission_id=sub2_id,
                            action_type="SUPERSEDES",
                            target_belief_id=b1_id,
                            replacement_belief_id=b2_id,
                            rationale="Superseded by newer evidence.",
                            evidence_ids=(ev2_id,),
                        )
                    ]
                    gold_statuses = {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED"}
                    
                elif f_type == "stale_propagation":
                    sub2_candidate_replacement_beliefs = (b2,)
                    # Add child belief in sub2 dependent on b1
                    b_child_id = f"b_{episode_id}_child"
                    b_child = BeliefNode(
                        belief_id=b_child_id,
                        proposition=f"Propagated child assertion for v{v}.",
                        source_evidence_ids=(ev1_id,),
                    )
                    sub2_candidate_beliefs = (b1, b_child)
                    sub2_candidate_conditions_by_belief = (
                        (b_child_id, (cond1,)),
                    )
                    sub2_dependency_edges_by_belief = (
                        (b_child_id, (DependencyEdge(
                            edge_id=f"dep_{episode_id}_child",
                            belief_id=b_child_id,
                            condition_id=c1_id,
                            inducer="system",
                        ),)),
                    )
                    targets = [
                        TypedRevisionTarget(
                            submission_id=sub2_id,
                            action_type="SUPERSEDES",
                            target_belief_id=b1_id,
                            replacement_belief_id=b2_id,
                            rationale="Stale parent superseded.",
                            evidence_ids=(ev2_id,),
                        )
                    ]
                    # b_child will be blocked or unresolved due to dependency, parent is superseded
                    gold_statuses = {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED", b_child_id: "UNRESOLVED"}

                elif f_type == "scope_expansion":
                    # b1 requires cond1 in submission 2
                    sub2_candidate_conditions_by_belief = (
                        (b1_id, (cond1,)),
                    )
                    sub2_dependency_edges_by_belief = (
                        (b1_id, (DependencyEdge(
                            edge_id=f"dep_{episode_id}_1",
                            belief_id=b1_id,
                            condition_id=c1_id,
                            inducer="system",
                        ),)),
                    )
                    targets = [
                        TypedRevisionTarget(
                            submission_id=sub2_id,
                            action_type="BLOCKS",
                            target_condition_id=c1_id,
                            rationale="New evidence blocks the prerequisite condition.",
                            evidence_ids=(ev2_id,),
                        )
                    ]
                    gold_statuses = {b1_id: "BLOCKED"}

                elif f_type == "cross_agent_conflict":
                    # Propose conflicting belief
                    sub2_candidate_beliefs = (b1, b2)
                    targets = [
                        TypedRevisionTarget(
                            submission_id=sub2_id,
                            action_type="UNCERTAIN",
                            target_belief_id=b1_id,
                            rationale="Conflict between agent assertions creates uncertainty.",
                            evidence_ids=(ev2_id,),
                        )
                    ]
                    gold_statuses = {b1_id: "UNRESOLVED"}

                elif f_type == "temporary_blocker_recovery":
                    # Pre-existing blocked condition
                    sub2_candidate_conditions_by_belief = (
                        (b1_id, (cond1,)),
                    )
                    sub2_dependency_edges_by_belief = (
                        (b1_id, (DependencyEdge(
                            edge_id=f"dep_{episode_id}_1",
                            belief_id=b1_id,
                            condition_id=c1_id,
                            inducer="system",
                        ),)),
                    )
                    targets = [
                        TypedRevisionTarget(
                            submission_id=sub2_id,
                            action_type="RELEASES",
                            target_condition_id=c1_id,
                            rationale="Temporary blocker is now resolved.",
                            evidence_ids=(ev2_id,),
                        )
                    ]
                    gold_statuses = {b1_id: "AUTHORIZED"}

                elif f_type == "duplicate_evidence":
                    # ev2 identical text, NO_REVISION or REAFFIRMS
                    targets = [
                        TypedRevisionTarget(
                            submission_id=sub2_id,
                            action_type="NO_REVISION",
                            rationale="Duplicate evidence warrants no revision.",
                            evidence_ids=(ev2_id,),
                        )
                    ]
                    gold_statuses = {b1_id: "AUTHORIZED"}

                elif f_type == "ambiguous_update":
                    targets = [
                        TypedRevisionTarget(
                            submission_id=sub2_id,
                            action_type="UNCERTAIN",
                            target_belief_id=b1_id,
                            rationale="Evidence is ambiguous.",
                            evidence_ids=(ev2_id,),
                        )
                    ]
                    gold_statuses = {b1_id: "UNRESOLVED"}

                # Build submissions
                sub1 = FixedCandidateSubmission(
                    submission_id=sub1_id,
                    producer_id="subagent_1",
                    producer_role="writer",
                    task_id=f"task_{episode_id}",
                    parent_snapshot_id="snapshot_init",
                    observed_at="2026-05-30T00:00:00Z",
                    instance_id=f"inst_{episode_id}_1",
                    query_id=f"query_{episode_id}_1",
                    query=query,
                    evidence_context=(ev1,),
                    new_evidence_id=ev1_id,
                    candidate_beliefs=sub1_candidate_beliefs,
                    candidate_replacement_beliefs=sub1_candidate_replacement_beliefs,
                    candidate_conditions_by_belief=sub1_candidate_conditions_by_belief,
                    dependency_edges_by_belief=sub1_dependency_edges_by_belief,
                    metadata={"step": 1},
                )
                
                sub2 = FixedCandidateSubmission(
                    submission_id=sub2_id,
                    producer_id="subagent_2",
                    producer_role="reviewer",
                    task_id=f"task_{episode_id}",
                    parent_snapshot_id=f"snapshot_{episode_id}_1",
                    observed_at="2026-05-30T01:00:00Z",
                    instance_id=f"inst_{episode_id}_2",
                    query_id=f"query_{episode_id}_2",
                    query=query,
                    evidence_context=(ev1, ev2),
                    new_evidence_id=ev2_id,
                    candidate_beliefs=sub2_candidate_beliefs,
                    candidate_replacement_beliefs=sub2_candidate_replacement_beliefs,
                    candidate_conditions_by_belief=sub2_candidate_conditions_by_belief,
                    dependency_edges_by_belief=sub2_dependency_edges_by_belief,
                    metadata={"step": 2},
                )

                # Episode & Gold record metadata
                metadata = {
                    "review_status": "pending_human_review",
                    "training_eligible": False,
                    "scientific_status": "not_evaluated",
                    "label_source": "template_authored_pending_review",
                }

                # Downstream tasks
                tasks = (
                    DownstreamTask(
                        task_id=f"task_{episode_id}",
                        query=query,
                        expected_answer_or_action="AUTHORIZED" if gold_statuses.get(b2_id) == "AUTHORIZED" else "UNRESOLVED",
                    ),
                )

                # Gold expectations
                gold_snapshot = GoldSnapshotExpectation(
                    belief_statuses=gold_statuses,
                    rationale=f"Deterministic gold expectations for {f_type}.",
                )

                episode = FixedCandidateInputEpisode(
                    episode_id=episode_id,
                    domain=domain,
                    failure_type_public_or_controlled=f_type,
                    subagent_roles=("writer", "reviewer"),
                    submissions=(sub1, sub2),
                    downstream_tasks=tasks,
                    split="development_candidate",
                    protocol_mode="fixed_candidate_revision",
                    proposal_source="template_authored",
                    metadata=metadata,
                )

                gold_record = FixedCandidateGoldRecord(
                    episode_id=episode_id,
                    gold_snapshot=gold_snapshot,
                    gold_typed_targets=tuple(targets),
                    failure_type=f_type,
                    metadata=metadata,
                )
                
                episodes.append((episode, gold_record))
                
    return episodes
