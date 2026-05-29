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

def check_targets_visible(sub: FixedCandidateSubmission, targets: Tuple[TypedRevisionTarget, ...]) -> bool:
    visible_beliefs = {b.belief_id for b in sub.candidate_beliefs} | {b.belief_id for b in sub.candidate_replacement_beliefs}
    visible_conditions = set()
    for _, conds in sub.candidate_conditions_by_belief:
        for c in conds:
            visible_conditions.add(c.condition_id)
            
    for t in targets:
        if t.target_belief_id and t.target_belief_id not in visible_beliefs:
            return False
        if t.target_condition_id and t.target_condition_id not in visible_conditions:
            return False
        if t.replacement_belief_id and t.replacement_belief_id not in visible_beliefs:
            return False
    return True

def generate_expanded_episodes() -> List[Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]]:
    episodes = []
    
    for domain in DOMAINS:
        for f_type in FAILURE_TYPES:
            for v in range(1, 6): # 5 variants
                episode_id = f"ep_expansion_{domain}_{f_type}_v{v}"
                
                # Setup base metadata placeholders
                meta = {
                    "review_status": "pending_human_review",
                    "training_eligible": False,
                    "scientific_status": "not_evaluated",
                    "label_source": "template_authored_pending_review",
                }

                # 3. FAILURE-TYPE-SPECIFIC SEMANTIC TEMPLATES
                if f_type == "direct_supersession":
                    if domain == "software_engineering":
                        query = f"Check active logging framework for module {v}"
                        b_init_text = f"Module {v} is configured to use legacy console logger."
                        b_rev_text = f"Module {v} is successfully migrated to structured JSON logger."
                        ev_init_text = f"Legacy configurations for module {v} specify log=console."
                        ev_rev_text = f"Pull request #202{v} completed, setting log=structured_json."
                    else: # research_workflow
                        query = f"Check citation dataset version for project {v}"
                        b_init_text = f"Project {v} citation parsing relies on dataset release v1."
                        b_rev_text = f"Project {v} citation parsing migrated to dataset release v2."
                        ev_init_text = f"Project {v} setup manifest lists dataset v1."
                        ev_rev_text = f"Research report confirmed project {v} migrated to citation v2."
                    
                    sub1_id, sub2_id = f"sub_{episode_id}_1", f"sub_{episode_id}_2"
                    ev1_id, ev2_id = f"ev_{episode_id}_1", f"ev_{episode_id}_2"
                    b1_id, b2_id = f"b_{episode_id}_1", f"b_{episode_id}_2"
                    
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "dev_expansion", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_rev_text, "dev_expansion", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))
                    b2 = BeliefNode(b2_id, b_rev_text, (ev2_id,))
                    
                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    sub2 = FixedCandidateSubmission(sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id, (b1,), (b2,))
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b2_id, rationale="Explicit migration supersedes old config.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED"}

                elif f_type == "stale_propagation":
                    if domain == "software_engineering":
                        query = f"Check service gateway dependencies for stack {v}"
                        b_init_text = f"Stack {v} api-gateway depends on database pool config v1."
                        b_rev_text = f"Stack {v} api-gateway migrated to database pool config v2."
                        ev_init_text = f"Initial deploy configuration lists pool_config=v1."
                        ev_rev_text = f"Database administrator upgraded Stack {v} database pool to config v2."
                        c_text = f"Database pool config v1 is active."
                    else: # research_workflow
                        query = f"Verify pipeline configuration for experiment {v}"
                        b_init_text = f"Experiment {v} parser depends on base parser library v1.0."
                        b_rev_text = f"Experiment {v} parser upgraded to base parser library v2.0."
                        ev_init_text = f"Experiment setup lists base_parser_version=1.0."
                        ev_rev_text = f"System log confirms main parser version upgraded to 2.0."
                        c_text = f"Base parser library v1.0 is active."
                        
                    sub1_id, sub2_id = f"sub_{episode_id}_1", f"sub_{episode_id}_2"
                    ev1_id, ev2_id = f"ev_{episode_id}_1", f"ev_{episode_id}_2"
                    b1_id, b2_id = f"b_{episode_id}_1", f"b_{episode_id}_2"
                    b_child_id = f"b_{episode_id}_child"
                    c1_id = f"c_{episode_id}_1"
                    
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "dev_expansion", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_rev_text, "dev_expansion", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))
                    b2 = BeliefNode(b2_id, b_rev_text, (ev2_id,))
                    b_child = BeliefNode(b_child_id, f"Dependent component {v} remains active.", (ev1_id,))
                    cond1 = ConditionNode(c1_id, f"scope_{episode_id}_1", c_text)
                    dep = DependencyEdge(f"dep_{episode_id}_child", b_child_id, c1_id, "system")
                    
                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    sub2 = FixedCandidateSubmission(
                        sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id,
                        candidate_beliefs=(b1, b_child),
                        candidate_replacement_beliefs=(b2,),
                        candidate_conditions_by_belief=((b_child_id, (cond1,)),),
                        dependency_edges_by_belief=((b_child_id, (dep,)),),
                    )
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b2_id, rationale="Root database pool superseded.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED", b_child_id: "UNRESOLVED"}

                elif f_type == "scope_expansion":
                    # Local versus Protected beliefs
                    if domain == "software_engineering":
                        query = f"Validate server credentials status for build {v}"
                        b_local_text = f"Staging server for build {v} can accept deployments."
                        b_protected_text = f"Production server for build {v} can accept deployments."
                        ev_init_text = f"Staging and production deployment routes are verified."
                        ev_rev_text = f"Staging API authentication key expired for build {v}."
                        c_staging_text = f"Staging authentication is valid for build {v}."
                        c_prod_text = f"Production authentication is valid for build {v}."
                    else: # research_workflow
                        query = f"Verify pipeline access status for study {v}"
                        b_local_text = f"Staging pipeline {v} can execute data ingestion."
                        b_protected_text = f"Production database {v} is authorized to accept query traffic."
                        ev_init_text = f"Staging and production database links initialized."
                        ev_rev_text = f"Staging database connection timed out for study {v}."
                        c_staging_text = f"Staging link status active for study {v}."
                        c_prod_text = f"Production database link status active for study {v}."
                        
                    sub1_id, sub2_id = f"sub_{episode_id}_1", f"sub_{episode_id}_2"
                    ev1_id, ev2_id = f"ev_{episode_id}_1", f"ev_{episode_id}_2"
                    b_local_id, b_protected_id = f"b_{episode_id}_local", f"b_{episode_id}_protected"
                    c_staging_id, c_prod_id = f"c_{episode_id}_staging", f"c_{episode_id}_prod"
                    
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "dev_expansion", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_rev_text, "dev_expansion", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b_local = BeliefNode(b_local_id, b_local_text, (ev1_id,))
                    b_protected = BeliefNode(b_protected_id, b_protected_text, (ev1_id,))
                    
                    cond_staging = ConditionNode(c_staging_id, f"scope_{episode_id}_staging", c_staging_text)
                    cond_prod = ConditionNode(c_prod_id, f"scope_{episode_id}_prod", c_prod_text)
                    
                    dep_local = DependencyEdge(f"dep_{episode_id}_local", b_local_id, c_staging_id, "system")
                    dep_prod = DependencyEdge(f"dep_{episode_id}_prod", b_protected_id, c_prod_id, "system")
                    
                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b_local, b_protected), ())
                    sub2 = FixedCandidateSubmission(
                        sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id,
                        candidate_beliefs=(b_local, b_protected),
                        candidate_replacement_beliefs=(),
                        candidate_conditions_by_belief=((b_local_id, (cond_staging,)), (b_protected_id, (cond_prod,))),
                        dependency_edges_by_belief=((b_local_id, (dep_local,)), (b_protected_id, (dep_prod,))),
                    )
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "BLOCKS", target_condition_id=c_staging_id, rationale="Staging key expiration blocks staging route.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b_local_id: "BLOCKED", b_protected_id: "AUTHORIZED"}

                elif f_type == "cross_agent_conflict":
                    if domain == "software_engineering":
                        query = f"Resolve memory leak report for module {v}"
                        b_init_text = f"Linter indicates module {v} memory leaks are resolved."
                        b_conflict_text = f"Runtime monitor reports module {v} memory usage is still growing."
                        ev_init_text = f"Linter output: code pattern issues resolved."
                        ev_rev_text = f"Runtime statistics: heap usage increased by 150MB in module {v}."
                    else: # research_workflow
                        query = f"Check experimental replicability for study {v}"
                        b_init_text = f"Reviewer 1 reports study {v} replication succeeded."
                        b_conflict_text = f"Reviewer 2 reports study {v} replication failed due to config drift."
                        ev_init_text = f"Replication log file 1 marks success."
                        ev_rev_text = f"Replication log file 2 logs validation error."
                        
                    sub1_id, sub2_id = f"sub_{episode_id}_1", f"sub_{episode_id}_2"
                    ev1_id, ev2_id = f"ev_{episode_id}_1", f"ev_{episode_id}_2"
                    b1_id, b2_id = f"b_{episode_id}_1", f"b_{episode_id}_2"
                    
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "dev_expansion", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_rev_text, "dev_expansion", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))
                    b2 = BeliefNode(b2_id, b_conflict_text, (ev2_id,))
                    
                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    sub2 = FixedCandidateSubmission(sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id, (b1, b2), ())
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "UNCERTAIN", target_belief_id=b1_id, rationale="Conflicting agents report conflicting statuses.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "UNRESOLVED", b2_id: "UNRESOLVED"}

                elif f_type == "temporary_blocker_recovery":
                    # REQUIRES 3 temporally ordered submissions
                    if domain == "software_engineering":
                        query = f"Validate compiler routes for deployment {v}"
                        b_init_text = f"Compiler route for deployment {v} is validated."
                        ev_init_text = f"Initial configuration file verifier logs success."
                        ev_block_text = f"Compiler server disk space is full for deployment {v}."
                        ev_recover_text = f"Disk clean script executed successfully for deployment {v}."
                        c_text = f"Compiler server workspace for deployment {v} has free space."
                    else: # research_workflow
                        query = f"Verify IRB authorization for clinical study {v}"
                        b_init_text = f"Clinical study {v} patient enrolment is active."
                        ev_init_text = f"Ethics board approval document signed."
                        ev_block_text = f"Annual report deadline missed, study {v} suspended."
                        ev_recover_text = f"Annual report submitted, study {v} suspension lifted."
                        c_text = f"IRB ethical approval status is active for study {v}."
                        
                    sub1_id, sub2_id, sub3_id = f"sub_{episode_id}_1", f"sub_{episode_id}_2", f"sub_{episode_id}_3"
                    ev1_id, ev2_id, ev3_id = f"ev_{episode_id}_1", f"ev_{episode_id}_2", f"ev_{episode_id}_3"
                    b1_id = f"b_{episode_id}_1"
                    c1_id = f"c_{episode_id}_1"
                    
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "dev_expansion", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_block_text, "dev_expansion", f"file:///src/{domain}/v{v}/2.txt")
                    ev3 = EvidenceNode(ev3_id, f"sess_{episode_id}_3", "2026-05-30T02:00:00Z", ev_recover_text, "dev_expansion", f"file:///src/{domain}/v{v}/3.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))
                    cond1 = ConditionNode(c1_id, f"scope_{episode_id}_1", c_text)
                    dep = DependencyEdge(f"dep_{episode_id}_1", b1_id, c1_id, "system")
                    
                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    sub2 = FixedCandidateSubmission(
                        sub2_id, "blocker_notifier", "blocker_notifier", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id,
                        candidate_beliefs=(b1,),
                        candidate_conditions_by_belief=((b1_id, (cond1,)),),
                        dependency_edges_by_belief=((b1_id, (dep,)),),
                    )
                    sub3 = FixedCandidateSubmission(
                        sub3_id, "recovery_notifier", "recovery_notifier", f"task_{episode_id}", f"snapshot_{episode_id}_2", "2026-05-30T02:00:00Z", f"inst_{episode_id}_3", f"q_{episode_id}_3", query, (ev1, ev2, ev3), ev3_id,
                        candidate_beliefs=(b1,),
                        candidate_conditions_by_belief=((b1_id, (cond1,)),),
                        dependency_edges_by_belief=((b1_id, (dep,)),),
                    )
                    submissions = (sub1, sub2, sub3)
                    
                    # We have targets across sub2 and sub3
                    # For E1 fixed candidate gold targets, we collect them across submissions
                    targets = [
                        TypedRevisionTarget(sub2_id, "BLOCKS", target_condition_id=c1_id, rationale="Full disk or ethics board suspension blocks prerequisite.", evidence_ids=(ev2_id,)),
                        TypedRevisionTarget(sub3_id, "RELEASES", target_condition_id=c1_id, rationale="Disk cleanup or suspension lifted releases prerequisite.", evidence_ids=(ev3_id,)),
                    ]
                    gold_statuses = {b1_id: "AUTHORIZED"}

                elif f_type == "duplicate_evidence":
                    if domain == "software_engineering":
                        query = f"Verify deploy status for unit {v}"
                        b_init_text = f"Unit {v} is deployed on k8s cluster."
                        ev_init_text = f"Logger node 1 reports k8s deploy success."
                        ev_rev_text = f"Logger node 2 duplicate message: k8s deploy success."
                    else: # research_workflow
                        query = f"Check parser database sync status {v}"
                        b_init_text = f"Parser DB sync {v} completed."
                        ev_init_text = f"Sync log file reports successful completion."
                        ev_rev_text = f"Sync validator confirms successful completion."
                        
                    sub1_id, sub2_id = f"sub_{episode_id}_1", f"sub_{episode_id}_2"
                    ev1_id, ev2_id = f"ev_{episode_id}_1", f"ev_{episode_id}_2"
                    b1_id = f"b_{episode_id}_1"
                    
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "dev_expansion", f"file:///src/{domain}/v{v}/1.txt")
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_rev_text, "dev_expansion", f"file:///src/{domain}/v{v}/2.txt")
                    
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))
                    
                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    # Metadata documents duplicate mapping
                    sub2 = FixedCandidateSubmission(sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id, (b1,), (), metadata={"duplicates_evidence_id": ev1_id})
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "NO_REVISION", rationale="Duplicate log reports no state changes.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "AUTHORIZED"}

                elif f_type == "ambiguous_update":
                    if domain == "software_engineering":
                        query = f"Check linter validation warning for parser {v}"
                        b_init_text = f"Parser {v} uses compliant syntax standard."
                        ev_init_text = f"Initial standard verification logs compliance."
                        ev_rev_text = f"Linter warning: parser {v} might contain deprecated features; compliance pending validation."
                    else: # research_workflow
                        query = f"Check replication signal for treatment {v}"
                        b_init_text = f"Treatment {v} data output is consistent."
                        ev_init_text = f"Efficacy test logs stable consistent outputs."
                        ev_rev_text = f"Ad-hoc sensor anomaly warning: treatment {v} test 2 shows a potential minor drift signal."
                        
                    sub1_id, sub2_id = f"sub_{episode_id}_1", f"sub_{episode_id}_2"
                    ev1_id, ev2_id = f"ev_{episode_id}_1", f"ev_{episode_id}_2"
                    b1_id = f"b_{episode_id}_1"
                    
                    ev1 = EvidenceNode(ev1_id, f"sess_{episode_id}_1", "2026-05-30T00:00:00Z", ev_init_text, "dev_expansion", f"file:///src/{domain}/v{v}/1.txt")
                    # Metadata documents uncertainty cue
                    ev2 = EvidenceNode(ev2_id, f"sess_{episode_id}_2", "2026-05-30T01:00:00Z", ev_rev_text, "dev_expansion", f"file:///src/{domain}/v{v}/2.txt", metadata={"uncertainty_cue": "warning/potential"})
                    
                    b1 = BeliefNode(b1_id, b_init_text, (ev1_id,))
                    
                    sub1 = FixedCandidateSubmission(sub1_id, "writer", "writer", f"task_{episode_id}", "snapshot_init", "2026-05-30T00:00:00Z", f"inst_{episode_id}_1", f"q_{episode_id}_1", query, (ev1,), ev1_id, (b1,), ())
                    sub2 = FixedCandidateSubmission(sub2_id, "reviewer", "reviewer", f"task_{episode_id}", f"snapshot_{episode_id}_1", "2026-05-30T01:00:00Z", f"inst_{episode_id}_2", f"q_{episode_id}_2", query, (ev1, ev2), ev2_id, (b1,), ())
                    
                    submissions = (sub1, sub2)
                    targets = [
                        TypedRevisionTarget(sub2_id, "UNCERTAIN", target_belief_id=b1_id, rationale="Hedged log warning creates uncertainty on compliance.", evidence_ids=(ev2_id,)),
                    ]
                    gold_statuses = {b1_id: "UNRESOLVED"}

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

                # Run checklists
                last_sub = submissions[-1]
                has_visible_new_ev = last_sub.new_evidence_id in [e.evidence_id for e in last_sub.evidence_context]
                targets_visible = check_targets_visible(last_sub, tuple(targets))
                
                # protected belief
                protected_belief_ok = True
                if f_type == "scope_expansion":
                    has_prot = any(b.belief_id.endswith("protected") for b in last_sub.candidate_beliefs)
                    protected_belief_ok = has_prot
                
                # prior blocker present
                blocker_present = True
                if f_type == "temporary_blocker_recovery":
                    has_block = any(t.action_type == "BLOCKS" for t in targets)
                    has_release = any(t.action_type == "RELEASES" for t in targets)
                    blocker_present = has_block and has_release
                    
                # duplicate metadata
                dup_declared = True
                if f_type == "duplicate_evidence":
                    dup_declared = "duplicates_evidence_id" in last_sub.metadata
                    
                # ambiguity cue
                ambiguity_ok = True
                if f_type == "ambiguous_update":
                    ambiguity_ok = any("uncertainty_cue" in e.metadata for e in last_sub.evidence_context)
                    
                task_ok = len(tasks) > 0
                
                checklist = {
                    "has_visible_new_evidence": has_visible_new_ev,
                    "typed_target_ids_visible": targets_visible,
                    "protected_belief_present_if_required": protected_belief_ok,
                    "prior_blocker_present_if_release": blocker_present,
                    "duplicate_relation_declared_if_no_revision": dup_declared,
                    "ambiguity_marker_present_if_uncertain": ambiguity_ok,
                    "downstream_task_defined": task_ok,
                }
                
                passes_structural = all(checklist.values())
                
                validation_status = {
                    "passes_structural_checks": passes_structural,
                    "requires_human_semantic_review": True,
                }
                
                # Merge checks into metadata
                episode_meta = meta.copy()
                episode_meta.update({
                    "semantic_checklist": checklist,
                    "semantic_validation_status": validation_status,
                })

                episode = FixedCandidateInputEpisode(
                    episode_id=episode_id,
                    domain=domain,
                    failure_type_public_or_controlled=f_type,
                    subagent_roles=tuple(set(s.producer_role for s in submissions)),
                    submissions=submissions,
                    downstream_tasks=tasks,
                    split="development_candidate",
                    protocol_mode="fixed_candidate_revision",
                    proposal_source="template_authored",
                    metadata=episode_meta,
                )

                gold_record = FixedCandidateGoldRecord(
                    episode_id=episode_id,
                    gold_snapshot=gold_snapshot,
                    gold_typed_targets=tuple(targets),
                    failure_type=f_type,
                    metadata=episode_meta,
                )
                
                episodes.append((episode, gold_record))
                
    return episodes
