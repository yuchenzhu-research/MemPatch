"""Deterministic ``paper1_balanced`` internal validation suite.

This is an **internal balanced synthetic validation set** for ReTrace-Learn
(Multi-Agent / Subagent Shared-Memory Revision Authorization). It is NOT a
Stage C training set, and it is NOT an external benchmark. It must not be
described as official STALE / Memora / CUPMem; those remain separate external
validation pathways (see ``AGENTS.md``).

Structure: ``len(FAILURE_TYPES) x len(DOMAINS) x VARIANTS`` cases.
With 14 failure types, 2 domains, and 15 variants this yields 420 cases.

Every case is fully executable: its ``gold_typed_targets`` route through the
deterministic ``authorize(...)`` / DPA kernel to produce exactly the authored
``gold_snapshot.belief_statuses``. The generator is purely programmatic — no
large generated JSONL is committed.

The templates for the seven shared failure types intentionally mirror the
proven ``dev_expansion`` templates (with a distinct ``ep_paper1_`` namespace);
the remaining seven add multi-action and single-action coverage that
``dev_expansion`` does not exercise.
"""
from __future__ import annotations

from typing import List, Tuple

from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
)
from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateSubmission,
    FixedCandidateInputEpisode,
    FixedCandidateGoldRecord,
    DownstreamTask,
    GoldSnapshotExpectation,
    TypedRevisionTarget,
)

GENERATOR_VERSION = "paper1_balanced_v1"
DATASET_NAME = "paper1_balanced"
VARIANTS = 15

DOMAINS = ["software_engineering", "research_workflow"]

FAILURE_TYPES = [
    "direct_supersession",
    "stale_propagation",
    "scope_expansion",
    "cross_agent_conflict",
    "temporary_blocker_recovery",
    "duplicate_evidence",
    "ambiguous_update",
    "multi_action_supersedes_blocks",
    "multi_action_supersedes_releases",
    "blocks_uncertain",
    "reaffirms_only",
    "evidence_conflict",
    "target_ambiguity",
    "no_revision",
]

_T0 = "2026-05-30T00:00:00Z"
_T1 = "2026-05-30T01:00:00Z"
_T2 = "2026-05-30T02:00:00Z"


def check_targets_visible(
    sub: FixedCandidateSubmission, targets: Tuple[TypedRevisionTarget, ...]
) -> bool:
    visible_beliefs = {b.belief_id for b in sub.candidate_beliefs} | {
        b.belief_id for b in sub.candidate_replacement_beliefs
    }
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


def _ev(eid: str, ep: str, idx: int, text: str, domain: str, v: int, **kw) -> EvidenceNode:
    ts = {1: _T0, 2: _T1, 3: _T2}[idx]
    return EvidenceNode(
        eid, f"sess_{ep}_{idx}", ts, text, "paper1_balanced",
        f"file:///paper1/{domain}/v{v}/{idx}.txt", **kw,
    )


def _build_case(domain: str, f_type: str, v: int):
    """Return (submissions, targets, gold_statuses, downstream_belief, expected_status, query)."""
    ep = f"ep_paper1_{domain}_{f_type}_v{v}"
    se = domain == "software_engineering"
    s1, s2, s3 = f"sub_{ep}_1", f"sub_{ep}_2", f"sub_{ep}_3"
    e1, e2, e3 = f"ev_{ep}_1", f"ev_{ep}_2", f"ev_{ep}_3"
    task = f"task_{ep}"

    if f_type == "direct_supersession":
        query = f"Check active logging framework for module {v}" if se else f"Check citation dataset version for project {v}"
        b1_id, b2_id = f"b_{ep}_1", f"b_{ep}_2"
        b1t = f"Module {v} is configured to use legacy console logger." if se else f"Project {v} citation parsing relies on dataset release v1."
        b2t = f"Module {v} is migrated to structured JSON logger." if se else f"Project {v} citation parsing migrated to dataset release v2."
        ev1 = _ev(e1, ep, 1, f"Legacy configuration for module {v} specifies log=console." if se else f"Setup manifest lists dataset v1 for project {v}.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Pull request #202{v} completed, setting log=structured_json." if se else f"Report confirms project {v} migrated to citation v2.", domain, v)
        b1, b2 = BeliefNode(b1_id, b1t, (e1,)), BeliefNode(b2_id, b2t, (e2,))
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2, (b1,), (b2,))
        targets = [TypedRevisionTarget(s2, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b2_id, rationale="Explicit migration supersedes old config.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED"}, b2_id, "AUTHORIZED", query

    if f_type == "stale_propagation":
        query = f"Check service gateway dependencies for stack {v}" if se else f"Verify pipeline configuration for experiment {v}"
        b1_id, b2_id, bc_id, c1_id = f"b_{ep}_1", f"b_{ep}_2", f"b_{ep}_child", f"c_{ep}_1"
        b1t = f"Stack {v} api-gateway depends on database pool config v1." if se else f"Experiment {v} parser depends on base parser library v1.0."
        b2t = f"Stack {v} api-gateway migrated to database pool config v2." if se else f"Experiment {v} parser upgraded to base parser library v2.0."
        ct = f"Database pool config v1 is active." if se else f"Base parser library v1.0 is active."
        ev1 = _ev(e1, ep, 1, f"Initial deploy lists pool_config=v1." if se else f"Setup lists base_parser_version=1.0.", domain, v)
        ev2 = _ev(e2, ep, 2, f"DBA upgraded stack {v} pool to config v2." if se else f"Log confirms parser upgraded to 2.0.", domain, v)
        b1, b2 = BeliefNode(b1_id, b1t, (e1,)), BeliefNode(b2_id, b2t, (e2,))
        bc = BeliefNode(bc_id, f"Dependent component {v} remains active.", (e1,))
        c1 = ConditionNode(c1_id, f"scope_{ep}_1", ct)
        dep = DependencyEdge(f"dep_{ep}_child", bc_id, c1_id, "system")
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2,
                                        candidate_beliefs=(b1, bc), candidate_replacement_beliefs=(b2,),
                                        candidate_conditions_by_belief=((bc_id, (c1,)),), dependency_edges_by_belief=((bc_id, (dep,)),))
        targets = [
            TypedRevisionTarget(s2, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b2_id, rationale="Root dependency superseded.", evidence_ids=(e2,)),
            TypedRevisionTarget(s2, "BLOCKS", target_condition_id=c1_id, rationale="Old config no longer active.", evidence_ids=(e2,)),
        ]
        return (sub1, sub2), targets, {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED", bc_id: "BLOCKED"}, bc_id, "BLOCKED", query

    if f_type == "scope_expansion":
        query = f"Validate server credentials status for build {v}" if se else f"Verify pipeline access status for study {v}"
        bl_id, bp_id, cs_id, cp_id = f"b_{ep}_local", f"b_{ep}_protected", f"c_{ep}_staging", f"c_{ep}_prod"
        blt = f"Staging server for build {v} can accept deployments." if se else f"Staging pipeline {v} can execute data ingestion."
        bpt = f"Production server for build {v} can accept deployments." if se else f"Production database {v} is authorized to accept query traffic."
        cst = f"Staging authentication is valid for build {v}." if se else f"Staging link status active for study {v}."
        cpt = f"Production authentication is valid for build {v}." if se else f"Production database link status active for study {v}."
        ev1 = _ev(e1, ep, 1, "Staging and production routes verified." if se else "Staging and production links initialized.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Staging key expired for build {v}." if se else f"Staging connection timed out for study {v}.", domain, v)
        bl, bp = BeliefNode(bl_id, blt, (e1,)), BeliefNode(bp_id, bpt, (e1,))
        cs, cp = ConditionNode(cs_id, f"scope_{ep}_staging", cst), ConditionNode(cp_id, f"scope_{ep}_prod", cpt)
        dl = DependencyEdge(f"dep_{ep}_local", bl_id, cs_id, "system")
        dp = DependencyEdge(f"dep_{ep}_prod", bp_id, cp_id, "system")
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (bl, bp), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2,
                                        candidate_beliefs=(bl, bp), candidate_replacement_beliefs=(),
                                        candidate_conditions_by_belief=((bl_id, (cs,)), (bp_id, (cp,))),
                                        dependency_edges_by_belief=((bl_id, (dl,)), (bp_id, (dp,))))
        targets = [TypedRevisionTarget(s2, "BLOCKS", target_condition_id=cs_id, rationale="Staging-only expiration blocks staging route.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {bl_id: "BLOCKED", bp_id: "AUTHORIZED"}, bl_id, "BLOCKED", query

    if f_type == "cross_agent_conflict":
        query = f"Resolve memory leak report for module {v}" if se else f"Check experimental replicability for study {v}"
        b1_id, b2_id = f"b_{ep}_1", f"b_{ep}_2"
        b1t = f"Linter indicates module {v} memory leaks are resolved." if se else f"Reviewer 1 reports study {v} replication succeeded."
        b2t = f"Runtime monitor reports module {v} memory usage still growing." if se else f"Reviewer 2 reports study {v} replication failed due to config drift."
        ev1 = _ev(e1, ep, 1, "Linter output: issues resolved." if se else "Replication log 1 marks success.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Runtime stats: heap increased 150MB in module {v}." if se else "Replication log 2 logs validation error.", domain, v)
        b1, b2 = BeliefNode(b1_id, b1t, (e1,)), BeliefNode(b2_id, b2t, (e2,))
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2, (b1, b2), ())
        targets = [TypedRevisionTarget(s2, "UNCERTAIN", target_belief_id=b1_id, rationale="Conflicting agents report conflicting statuses.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {b1_id: "UNRESOLVED", b2_id: "AUTHORIZED"}, b1_id, "UNRESOLVED", query

    if f_type == "temporary_blocker_recovery":
        query = f"Validate compiler routes for deployment {v}" if se else f"Verify IRB authorization for clinical study {v}"
        b1_id, c1_id = f"b_{ep}_1", f"c_{ep}_1"
        b1t = f"Compiler route for deployment {v} is validated." if se else f"Clinical study {v} patient enrolment is active."
        ct = f"Compiler server workspace for deployment {v} has free space." if se else f"IRB ethical approval status is active for study {v}."
        ev1 = _ev(e1, ep, 1, "Initial verifier logs success." if se else "Ethics board approval signed.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Compiler disk full for deployment {v}." if se else f"Annual report missed, study {v} suspended.", domain, v)
        ev3 = _ev(e3, ep, 3, f"Disk clean executed for deployment {v}." if se else f"Annual report submitted, study {v} suspension lifted.", domain, v)
        b1 = BeliefNode(b1_id, b1t, (e1,))
        c1 = ConditionNode(c1_id, f"scope_{ep}_1", ct)
        dep = DependencyEdge(f"dep_{ep}_1", b1_id, c1_id, "system")
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "blocker_notifier", "blocker_notifier", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2,
                                        candidate_beliefs=(b1,), candidate_conditions_by_belief=((b1_id, (c1,)),), dependency_edges_by_belief=((b1_id, (dep,)),))
        sub3 = FixedCandidateSubmission(s3, "recovery_notifier", "recovery_notifier", task, f"snapshot_{ep}_2", _T2, f"inst_{ep}_3", f"q_{ep}_3", query, (ev1, ev2, ev3), e3,
                                        candidate_beliefs=(b1,), candidate_conditions_by_belief=((b1_id, (c1,)),), dependency_edges_by_belief=((b1_id, (dep,)),))
        targets = [
            TypedRevisionTarget(s2, "BLOCKS", target_condition_id=c1_id, rationale="Prerequisite temporarily blocked.", evidence_ids=(e2,)),
            TypedRevisionTarget(s3, "RELEASES", target_condition_id=c1_id, rationale="Prerequisite restored.", evidence_ids=(e3,)),
        ]
        return (sub1, sub2, sub3), targets, {b1_id: "AUTHORIZED"}, b1_id, "AUTHORIZED", query

    if f_type == "duplicate_evidence":
        query = f"Verify deploy status for unit {v}" if se else f"Check parser database sync status {v}"
        b1_id = f"b_{ep}_1"
        b1t = f"Unit {v} is deployed on k8s cluster." if se else f"Parser DB sync {v} completed."
        ev1 = _ev(e1, ep, 1, "Logger node 1 reports deploy success." if se else "Sync log reports success.", domain, v)
        ev2 = _ev(e2, ep, 2, "Logger node 2 duplicate: deploy success." if se else "Sync validator confirms success.", domain, v)
        b1 = BeliefNode(b1_id, b1t, (e1,))
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2, (b1,), (), metadata={"duplicates_evidence_id": e1})
        targets = [TypedRevisionTarget(s2, "NO_REVISION", rationale="Duplicate report; no state change.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {b1_id: "AUTHORIZED"}, b1_id, "AUTHORIZED", query

    if f_type == "ambiguous_update":
        query = f"Check linter validation warning for parser {v}" if se else f"Check replication signal for treatment {v}"
        b1_id = f"b_{ep}_1"
        b1t = f"Parser {v} uses compliant syntax standard." if se else f"Treatment {v} data output is consistent."
        ev1 = _ev(e1, ep, 1, "Initial verification logs compliance." if se else "Efficacy test logs consistent output.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Warning: parser {v} might contain deprecated features; pending validation." if se else f"Anomaly warning: treatment {v} shows a potential minor drift.", domain, v, metadata={"uncertainty_cue": "warning/potential"})
        b1 = BeliefNode(b1_id, b1t, (e1,))
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2, (b1,), ())
        targets = [TypedRevisionTarget(s2, "UNCERTAIN", target_belief_id=b1_id, rationale="Hedged warning creates uncertainty.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {b1_id: "UNRESOLVED"}, b1_id, "UNRESOLVED", query

    if f_type == "multi_action_supersedes_blocks":
        query = f"Audit rollout pipeline for service {v}" if se else f"Audit analysis pipeline for cohort {v}"
        b1_id, b2_id, bc_id, c1_id = f"b_{ep}_1", f"b_{ep}_2", f"b_{ep}_child", f"c_{ep}_1"
        b1t = f"Service {v} routes traffic through ingress controller v1." if se else f"Cohort {v} analysis uses statistical model v1."
        b2t = f"Service {v} routes traffic through ingress controller v2." if se else f"Cohort {v} analysis migrated to statistical model v2."
        ct = f"Ingress controller v1 endpoint is reachable." if se else f"Statistical model v1 license is active."
        ev1 = _ev(e1, ep, 1, "Initial topology lists ingress v1." if se else "Setup lists model v1.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Migration to ingress v2 completed for service {v}; v1 endpoint decommissioned." if se else f"Cohort {v} migrated to model v2; v1 license retired.", domain, v)
        b1, b2 = BeliefNode(b1_id, b1t, (e1,)), BeliefNode(b2_id, b2t, (e2,))
        bc = BeliefNode(bc_id, f"Downstream monitor {v} relies on the prior endpoint.", (e1,))
        c1 = ConditionNode(c1_id, f"scope_{ep}_1", ct)
        dep = DependencyEdge(f"dep_{ep}_child", bc_id, c1_id, "system")
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2,
                                        candidate_beliefs=(b1, bc), candidate_replacement_beliefs=(b2,),
                                        candidate_conditions_by_belief=((bc_id, (c1,)),), dependency_edges_by_belief=((bc_id, (dep,)),))
        targets = [
            TypedRevisionTarget(s2, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b2_id, rationale="Routing migration supersedes prior belief.", evidence_ids=(e2,)),
            TypedRevisionTarget(s2, "BLOCKS", target_condition_id=c1_id, rationale="Decommissioned endpoint blocks the prerequisite.", evidence_ids=(e2,)),
        ]
        return (sub1, sub2), targets, {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED", bc_id: "BLOCKED"}, bc_id, "BLOCKED", query

    if f_type == "multi_action_supersedes_releases":
        query = f"Reconcile deploy gate for service {v}" if se else f"Reconcile access gate for study {v}"
        b1_id, b2_id, bd_id, c1_id = f"b_{ep}_1", f"b_{ep}_2", f"b_{ep}_dep", f"c_{ep}_1"
        b1t = f"Service {v} uses secret backend v1." if se else f"Study {v} uses credential vault v1."
        b2t = f"Service {v} uses secret backend v2." if se else f"Study {v} uses credential vault v2."
        bdt = f"Nightly job {v} can read secrets." if se else f"Nightly export {v} can read credentials."
        ct = f"Secret backend lease for job {v} is granted." if se else f"Credential lease for export {v} is granted."
        ev1 = _ev(e1, ep, 1, "Initial config lists secret backend v1; lease granted." if se else "Initial config lists vault v1; lease granted.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Lease for job {v} temporarily revoked during rotation." if se else f"Lease for export {v} temporarily revoked during rotation.", domain, v)
        ev3 = _ev(e3, ep, 3, f"Rotation finished: service {v} now on backend v2 and lease restored." if se else f"Rotation finished: study {v} now on vault v2 and lease restored.", domain, v)
        b1, b2 = BeliefNode(b1_id, b1t, (e1,)), BeliefNode(b2_id, b2t, (e3,))
        bd = BeliefNode(bd_id, bdt, (e1,))
        c1 = ConditionNode(c1_id, f"scope_{ep}_1", ct)
        dep = DependencyEdge(f"dep_{ep}_1", bd_id, c1_id, "system")
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1,
                                        candidate_beliefs=(b1, bd), candidate_conditions_by_belief=((bd_id, (c1,)),), dependency_edges_by_belief=((bd_id, (dep,)),))
        sub2 = FixedCandidateSubmission(s2, "blocker_notifier", "blocker_notifier", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2,
                                        candidate_beliefs=(bd,), candidate_conditions_by_belief=((bd_id, (c1,)),), dependency_edges_by_belief=((bd_id, (dep,)),))
        sub3 = FixedCandidateSubmission(s3, "reviewer", "reviewer", task, f"snapshot_{ep}_2", _T2, f"inst_{ep}_3", f"q_{ep}_3", query, (ev1, ev2, ev3), e3,
                                        candidate_beliefs=(b1, bd), candidate_replacement_beliefs=(b2,),
                                        candidate_conditions_by_belief=((bd_id, (c1,)),), dependency_edges_by_belief=((bd_id, (dep,)),))
        targets = [
            TypedRevisionTarget(s2, "BLOCKS", target_condition_id=c1_id, rationale="Rotation revokes the lease.", evidence_ids=(e2,)),
            TypedRevisionTarget(s3, "SUPERSEDES", target_belief_id=b1_id, replacement_belief_id=b2_id, rationale="Backend migration supersedes prior belief.", evidence_ids=(e3,)),
            TypedRevisionTarget(s3, "RELEASES", target_condition_id=c1_id, rationale="Rotation complete; lease restored.", evidence_ids=(e3,)),
        ]
        return (sub1, sub2, sub3), targets, {b1_id: "SUPERSEDED", b2_id: "AUTHORIZED", bd_id: "AUTHORIZED"}, b2_id, "AUTHORIZED", query

    if f_type == "blocks_uncertain":
        query = f"Triage release readiness for component {v}" if se else f"Triage publication readiness for finding {v}"
        bm_id, bo_id, c1_id = f"b_{ep}_main", f"b_{ep}_other", f"c_{ep}_1"
        bmt = f"Component {v} integration suite passes." if se else f"Finding {v} primary result is reproducible."
        bot = f"Component {v} performance profile is acceptable." if se else f"Finding {v} secondary result is acceptable."
        ct = f"Integration test environment for component {v} is healthy." if se else f"Reproduction environment for finding {v} is healthy."
        ev1 = _ev(e1, ep, 1, "Initial checks pass; environment healthy." if se else "Initial reproduction succeeds; environment healthy.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Test environment for component {v} crashed; perf numbers look inconsistent and unverified." if se else f"Reproduction environment for finding {v} failed; secondary result looks inconsistent and unverified.", domain, v, metadata={"uncertainty_cue": "inconsistent/unverified"})
        bm, bo = BeliefNode(bm_id, bmt, (e1,)), BeliefNode(bo_id, bot, (e1,))
        c1 = ConditionNode(c1_id, f"scope_{ep}_1", ct)
        dep = DependencyEdge(f"dep_{ep}_main", bm_id, c1_id, "system")
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1,
                                        candidate_beliefs=(bm, bo), candidate_conditions_by_belief=((bm_id, (c1,)),), dependency_edges_by_belief=((bm_id, (dep,)),))
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2,
                                        candidate_beliefs=(bm, bo), candidate_conditions_by_belief=((bm_id, (c1,)),), dependency_edges_by_belief=((bm_id, (dep,)),))
        targets = [
            TypedRevisionTarget(s2, "BLOCKS", target_condition_id=c1_id, rationale="Environment crash blocks the prerequisite.", evidence_ids=(e2,)),
            TypedRevisionTarget(s2, "UNCERTAIN", target_belief_id=bo_id, rationale="Inconsistent unverified signal.", evidence_ids=(e2,)),
        ]
        return (sub1, sub2), targets, {bm_id: "BLOCKED", bo_id: "UNRESOLVED"}, bm_id, "BLOCKED", query

    if f_type == "reaffirms_only":
        query = f"Confirm rollback safety for release {v}" if se else f"Confirm dataset integrity for corpus {v}"
        b1_id = f"b_{ep}_1"
        b1t = f"Release {v} rollback procedure is verified safe." if se else f"Corpus {v} integrity checksum matches."
        ev1 = _ev(e1, ep, 1, f"Initial rollback drill for release {v} succeeds." if se else f"Initial checksum for corpus {v} matches.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Independent re-run reconfirms release {v} rollback is safe." if se else f"Independent re-run reconfirms corpus {v} checksum matches.", domain, v)
        b1 = BeliefNode(b1_id, b1t, (e1,))
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2, (b1,), ())
        targets = [TypedRevisionTarget(s2, "REAFFIRMS", target_belief_id=b1_id, rationale="Corroborating evidence reaffirms the belief.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {b1_id: "AUTHORIZED"}, b1_id, "AUTHORIZED", query

    if f_type == "evidence_conflict":
        query = f"Adjudicate sensor reports for system {v}" if se else f"Adjudicate measurement reports for assay {v}"
        b1_id, b2_id = f"b_{ep}_1", f"b_{ep}_2"
        b1t = f"System {v} temperature sensor reads nominal." if se else f"Assay {v} measurement reads within tolerance."
        b2t = f"System {v} backup sensor reads an over-temperature alarm." if se else f"Assay {v} replicate measurement reads out of tolerance."
        ev1 = _ev(e1, ep, 1, "Primary sensor log: nominal." if se else "Primary measurement: in tolerance.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Backup sensor log for system {v} contradicts the primary reading." if se else f"Replicate measurement for assay {v} contradicts the primary reading.", domain, v)
        b1, b2 = BeliefNode(b1_id, b1t, (e1,)), BeliefNode(b2_id, b2t, (e2,))
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2, (b1, b2), ())
        targets = [TypedRevisionTarget(s2, "UNCERTAIN", target_belief_id=b1_id, rationale="Conflicting evidence sources for the same belief.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {b1_id: "UNRESOLVED", b2_id: "AUTHORIZED"}, b1_id, "UNRESOLVED", query

    if f_type == "target_ambiguity":
        query = f"Update config bundle for service {v}" if se else f"Update config bundle for experiment {v}"
        ba_id, ba2_id, bb_id = f"b_{ep}_a", f"b_{ep}_a2", f"b_{ep}_b"
        bat = f"Service {v} config profile A is on schema v1." if se else f"Experiment {v} config profile A is on schema v1."
        ba2t = f"Service {v} config profile A is on schema v2." if se else f"Experiment {v} config profile A is on schema v2."
        bbt = f"Service {v} config profile B is on schema v1." if se else f"Experiment {v} config profile B is on schema v1."
        ev1 = _ev(e1, ep, 1, "Profiles A and B both initialized on schema v1." if se else "Profiles A and B both initialized on schema v1.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Only profile A migrated to schema v2 for {v}; profile B unchanged." if se else f"Only profile A migrated to schema v2 for {v}; profile B unchanged.", domain, v)
        ba, ba2, bb = BeliefNode(ba_id, bat, (e1,)), BeliefNode(ba2_id, ba2t, (e2,)), BeliefNode(bb_id, bbt, (e1,))
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (ba, bb), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2, (ba, bb), (ba2,))
        targets = [TypedRevisionTarget(s2, "SUPERSEDES", target_belief_id=ba_id, replacement_belief_id=ba2_id, rationale="Only profile A is superseded; profile B is a distractor.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {ba_id: "SUPERSEDED", ba2_id: "AUTHORIZED", bb_id: "AUTHORIZED"}, bb_id, "AUTHORIZED", query

    if f_type == "no_revision":
        query = f"Review status note for service {v}" if se else f"Review status note for project {v}"
        b1_id = f"b_{ep}_1"
        b1t = f"Service {v} is running on the approved release channel." if se else f"Project {v} is on the approved analysis plan."
        ev1 = _ev(e1, ep, 1, f"Service {v} confirmed on approved channel." if se else f"Project {v} confirmed on approved plan.", domain, v)
        ev2 = _ev(e2, ep, 2, f"Cosmetic dashboard label for service {v} was renamed; no state change." if se else f"Cosmetic report heading for project {v} was renamed; no state change.", domain, v)
        b1 = BeliefNode(b1_id, b1t, (e1,))
        sub1 = FixedCandidateSubmission(s1, "writer", "writer", task, "snapshot_init", _T0, f"inst_{ep}_1", f"q_{ep}_1", query, (ev1,), e1, (b1,), ())
        sub2 = FixedCandidateSubmission(s2, "reviewer", "reviewer", task, f"snapshot_{ep}_1", _T1, f"inst_{ep}_2", f"q_{ep}_2", query, (ev1, ev2), e2, (b1,), ())
        targets = [TypedRevisionTarget(s2, "NO_REVISION", rationale="Cosmetic change carries no actionable state update.", evidence_ids=(e2,))]
        return (sub1, sub2), targets, {b1_id: "AUTHORIZED"}, b1_id, "AUTHORIZED", query

    raise ValueError(f"Unknown failure type: {f_type}")


def generate_paper1_balanced_episodes() -> List[Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]]:
    """Deterministically generate the balanced paper1 validation episodes."""
    episodes: List[Tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]] = []

    for domain in DOMAINS:
        for f_type in FAILURE_TYPES:
            for v in range(1, VARIANTS + 1):
                episode_id = f"ep_paper1_{domain}_{f_type}_v{v}"
                submissions, targets, gold_statuses, downstream_belief, expected_status, query = _build_case(domain, f_type, v)

                tasks = (
                    DownstreamTask(
                        task_id=f"task_{episode_id}",
                        query=query,
                        expected_answer_or_action=expected_status,
                        relevant_belief_ids=(downstream_belief,) if downstream_belief else (),
                    ),
                )

                gold_snapshot = GoldSnapshotExpectation(
                    belief_statuses=gold_statuses,
                    rationale=f"Deterministic paper1_balanced gold for {f_type}.",
                )

                last_sub = submissions[-1]
                last_targets = tuple(t for t in targets if t.submission_id == last_sub.submission_id)
                checklist = {
                    "has_visible_new_evidence": last_sub.new_evidence_id in [e.evidence_id for e in last_sub.evidence_context],
                    "typed_target_ids_visible": all(
                        check_targets_visible(s, tuple(t for t in targets if t.submission_id == s.submission_id))
                        for s in submissions
                    ),
                    "downstream_task_defined": len(tasks) > 0,
                    "last_submission_has_target": len(last_targets) > 0,
                }

                episode_meta = {
                    "dataset_name": DATASET_NAME,
                    "generator_version": GENERATOR_VERSION,
                    "review_status": "synthetic_internal_validation",
                    "training_eligible": False,
                    "scientific_status": "internal_balanced_validation",
                    "label_source": "deterministic_template_generator",
                    "variant_index": v,
                    "semantic_checklist": checklist,
                    "semantic_validation_status": {
                        "passes_structural_checks": all(checklist.values()),
                        "requires_human_semantic_review": False,
                    },
                    "downstream_target_belief_id": downstream_belief,
                    "expected_target_status": expected_status,
                }

                episode = FixedCandidateInputEpisode(
                    episode_id=episode_id,
                    domain=domain,
                    failure_type_public_or_controlled=f_type,
                    subagent_roles=tuple(sorted({s.producer_role for s in submissions})),
                    submissions=submissions,
                    downstream_tasks=tasks,
                    split="internal_balanced_validation",
                    protocol_mode="fixed_candidate_revision",
                    proposal_source="deterministic_template_generator",
                    metadata=episode_meta,
                )

                gold_record = FixedCandidateGoldRecord(
                    episode_id=episode_id,
                    gold_snapshot=gold_snapshot,
                    gold_typed_targets=tuple(targets),
                    failure_type=f_type,
                    requires_multi_action=len({t.submission_id for t in targets if t.action_type != "NO_REVISION"}) > 0
                    and any(len([t for t in targets if t.submission_id == s.submission_id and t.action_type != "NO_REVISION"]) > 1 for s in submissions),
                    metadata=episode_meta,
                )

                episodes.append((episode, gold_record))

    return episodes
