from __future__ import annotations

import hashlib
from typing import Any
from retracemem.authorization import authorize
from retracemem.methods.contracts import SharedCandidateView
from retracemem.multiagent.contracts import SubagentMemorySubmission, SharedMemoryCommitResult


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
