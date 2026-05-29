from __future__ import annotations

import hashlib
from typing import Any
from retracemem.authorization import authorize
from retracemem.methods.contracts import SharedCandidateView
from retracemem.multiagent.contracts import SubagentMemorySubmission, SharedMemoryCommitResult, SharedMemorySnapshotResult
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm
from retracemem.tms.gate import RevisionGate
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.schemas import AuthorizationStatus, EvidenceEdgeType, EvidenceEdge


def commit_subagent_submission(
    submission: SubagentMemorySubmission,
) -> SharedMemoryCommitResult:
    """Commit a subagent's memory submission to the shared memory basis."""
    # 1. Validate that new_evidence_id identifies exactly one node in evidence_context
    matching_new_ev = [ev for ev in submission.evidence_context if ev.evidence_id == submission.new_evidence_id]
    if len(matching_new_ev) != 1:
        raise ValueError(
            f"new_evidence_id '{submission.new_evidence_id}' must match exactly one "
            f"EvidenceNode in evidence_context (found {len(matching_new_ev)} matches)"
        )
    new_ev = matching_new_ev[0]

    # 2. Validate that any BeliefNode.source_evidence_ids referenced by the submission are represented in evidence_context
    context_ev_ids = {ev.evidence_id for ev in submission.evidence_context}
    for b in submission.candidate_beliefs:
        for evid in b.source_evidence_ids:
            if evid not in context_ev_ids:
                raise ValueError(
                    f"BeliefNode '{b.belief_id}' references source evidence '{evid}' "
                    f"not present in evidence_context"
                )
    for b in submission.candidate_replacement_beliefs:
        for evid in b.source_evidence_ids:
            if evid not in context_ev_ids:
                raise ValueError(
                    f"Replacement BeliefNode '{b.belief_id}' references source evidence '{evid}' "
                    f"not present in evidence_context"
                )

    # 3. Construct SharedCandidateView
    view = SharedCandidateView(
        instance_id=submission.instance_id,
        query_id=submission.query_id,
        query=submission.query,
        evidence_context=submission.evidence_context,
        new_evidence=new_ev,
        candidate_beliefs=submission.candidate_beliefs,
        candidate_replacement_beliefs=submission.candidate_replacement_beliefs,
        candidate_conditions_by_belief=submission.candidate_conditions_by_belief,
        dependency_edges_by_belief=submission.dependency_edges_by_belief,
        metadata=submission.metadata,
    )

    # 4. Call authorize(...)
    audit_metadata = {
        "producer_id": submission.producer_id,
        "producer_role": submission.producer_role,
        "submission_id": submission.submission_id,
        "observed_at": submission.observed_at,
    }
    if submission.task_id:
        audit_metadata["task_id"] = submission.task_id

    auth_res = authorize(
        view=view,
        proposal_batches=submission.proposal_batches,
        audit_metadata=audit_metadata,
    )

    # 5. Derive a deterministic next_snapshot_id from semantic fingerprint + provenance
    hasher = hashlib.sha256()
    hasher.update(view.view_fingerprint.encode("utf-8"))
    hasher.update(submission.submission_id.encode("utf-8"))
    hasher.update(submission.producer_id.encode("utf-8"))
    hasher.update(submission.observed_at.encode("utf-8"))
    next_snapshot_id = f"snap_{hasher.hexdigest()[:16]}"

    # 6. Emit a JSON-serializable commit trace
    commit_trace = {
        "submission_id": submission.submission_id,
        "producer_id": submission.producer_id,
        "producer_role": submission.producer_role,
        "parent_snapshot_id": submission.parent_snapshot_id,
        "next_snapshot_id": next_snapshot_id,
        "observed_at": submission.observed_at,
        "view_fingerprint": view.view_fingerprint,
        "task_id": submission.task_id,
        "auth_trace": auth_res.trace,
    }

    return SharedMemoryCommitResult(
        submission_id=submission.submission_id,
        producer_id=submission.producer_id,
        producer_role=submission.producer_role,
        parent_snapshot_id=submission.parent_snapshot_id,
        next_snapshot_id=next_snapshot_id,
        authorization_result=auth_res,
        commit_trace=commit_trace,
    )


def order_subagent_submissions(
    submissions: tuple[SubagentMemorySubmission, ...],
) -> tuple[SubagentMemorySubmission, ...]:
    """Deterministic order by observed_at, submission_id, producer_id."""
    return tuple(
        sorted(
            submissions,
            key=lambda s: (s.observed_at, s.submission_id, s.producer_id)
        )
    )


def commit_submission_sequence(
    submissions: tuple[SubagentMemorySubmission, ...],
    *,
    final_snapshot_evaluation: bool = True,
) -> SharedMemorySnapshotResult:
    """Execute a sequence of subagent memory submissions deterministically."""
    if not submissions:
        raise ValueError("Cannot commit an empty sequence of submissions.")

    submission_results = []
    initial_snapshot_id = submissions[0].parent_snapshot_id
    final_snapshot_id = initial_snapshot_id

    global_store = BeliefStore()
    global_ledger = EpisodeLedger()
    gate = RevisionGate()

    for sub in submissions:
        # 1. Commit individual subagent submission
        commit_res = commit_subagent_submission(sub)
        submission_results.append(commit_res)
        final_snapshot_id = commit_res.next_snapshot_id

        # 2. Cumulative insertion of evidence nodes
        for ev in sub.evidence_context:
            if ev.evidence_id not in global_ledger:
                global_ledger.append(ev)

        # 3. Cumulative insertion of beliefs
        for b in sub.candidate_beliefs:
            if not global_store.has_belief(b.belief_id):
                global_store.add_belief(b)
        for b in sub.candidate_replacement_beliefs:
            if not global_store.has_belief(b.belief_id):
                global_store.add_belief(b)

        # 4. Cumulative insertion of conditions
        for _bid, conds in sub.candidate_conditions_by_belief:
            for cond in conds:
                if not global_store.has_condition(cond.condition_id):
                    global_store.add_condition(cond)

        # 5. Cumulative insertion of dependency edges
        for _bid, deps in sub.dependency_edges_by_belief:
            for dep in deps:
                if not global_store.has_dependency_edge(dep.edge_id):
                    global_store.add_dependency_edge(dep)

        # 6. Cumulative insertion of admitted evidence edges from proposal batches
        for batch in sub.proposal_batches:
            for edge in batch.edges:
                dec = gate.admit_evidence_edge(edge, global_store)
                if dec.admitted and not global_store.has_evidence_edge(edge.edge_id):
                    global_store.add_evidence_edge(edge)

    # 7. Final DPA evaluation without temporal cutoff
    final_belief_statuses = {}
    final_authorized = []
    final_excluded = []

    if final_snapshot_evaluation:
        dpa = DefeatPathAuthorizationAlgorithm(global_store, global_ledger)
        for b in global_store.all_beliefs():
            trace = dpa.authorize(b.belief_id, as_of_evidence_id=None)
            status_str = "SUPERSEDED"
            if trace.status == AuthorizationStatus.AUTHORIZED:
                status_str = "AUTHORIZED"
                final_authorized.append(b.belief_id)
            elif trace.status == AuthorizationStatus.BLOCKED:
                status_str = "BLOCKED"
                final_excluded.append(b.belief_id)
            elif trace.status == AuthorizationStatus.UNRESOLVED:
                status_str = "UNRESOLVED"
                final_excluded.append(b.belief_id)
            else:
                final_excluded.append(b.belief_id)
            final_belief_statuses[b.belief_id] = status_str

    trace_info = {
        "number_of_submissions": len(submissions),
        "cumulative_beliefs": [b.belief_id for b in global_store.all_beliefs()],
        "cumulative_evidence": global_ledger.ids(),
    }

    return SharedMemorySnapshotResult(
        initial_snapshot_id=initial_snapshot_id,
        final_snapshot_id=final_snapshot_id,
        submission_results=tuple(submission_results),
        final_belief_statuses=final_belief_statuses,
        final_authorized_belief_ids=tuple(final_authorized),
        final_excluded_belief_ids=tuple(final_excluded),
        trace=trace_info,
    )
