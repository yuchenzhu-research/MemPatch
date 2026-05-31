from __future__ import annotations

from typing import Any
from retracemem.multiagent.parser import (
    strip_think_tags,
    strip_code_fences,
    extract_json_array as extract_first_json_array,
    extract_json_object,
    canonicalize_id,
)

def canonicalize_belief_id_with_type(
    returned_id: str,
    valid_belief_ids: set[str],
) -> tuple[str, bool, str]:
    """Backward compatible wrapper around canonicalize_id."""
    res = canonicalize_id(returned_id, valid_belief_ids)
    return res.value, res.sensitivity_applied, res.sensitivity_type

class CandidateActionBuilder:
    """Deterministic builder for structure-allowable candidate revision actions."""

    @staticmethod
    def build_candidates(submission: Any) -> list[dict[str, Any]]:
        """Build all structure-allowable candidate revision actions using method-visible inputs."""
        candidates = []
        new_ev_id = submission.new_evidence_id

        # Find the new evidence text for semantic why_candidate
        new_ev = next((ev for ev in submission.evidence_context if ev.evidence_id == new_ev_id), None)
        new_ev_snippet = new_ev.text[:60].replace("\n", " ") if new_ev else "No evidence text"

        # 1. SUPERSEDES candidates
        for b_old in submission.candidate_beliefs:
            for b_new in submission.candidate_replacement_beliefs:
                candidates.append({
                    "candidate_action_id": f"act_supersedes_{b_old.belief_id}_{b_new.belief_id}",
                    "action_type": "SUPERSEDES",
                    "target_belief_id": b_old.belief_id,
                    "target_condition_id": None,
                    "replacement_belief_id": b_new.belief_id,
                    "evidence_ids": [new_ev_id],
                    "why_candidate": f"Based on evidence '{new_ev_snippet}...', evaluate whether to replace old belief {b_old.belief_id} ({b_old.proposition[:30]}) with new belief {b_new.belief_id} ({b_new.proposition[:30]})."
                })

        # Build condition_id to dependent belief_ids mapping to preserve dependencies context
        cond_to_beliefs: dict[str, list[str]] = {}
        cond_node_map: dict[str, Any] = {}
        for bid, conds in getattr(submission, "candidate_conditions_by_belief", []):
            for c in conds:
                cond_to_beliefs.setdefault(c.condition_id, []).append(bid)
                cond_node_map[c.condition_id] = c

        # 2. BLOCKS and 3. RELEASES candidates
        for cid, dep_bids in cond_to_beliefs.items():
            c = cond_node_map[cid]
            bids_str = ", ".join(dep_bids)
            candidates.append({
                "candidate_action_id": f"act_blocks_{c.condition_id}",
                "action_type": "BLOCKS",
                "target_belief_id": None,
                "target_condition_id": c.condition_id,
                "replacement_belief_id": None,
                "evidence_ids": [new_ev_id],
                "why_candidate": f"Based on evidence '{new_ev_snippet}...', evaluate whether to invalidate prerequisite {c.condition_id} ({c.text[:30]}) [depended on by belief {bids_str}]."
            })
            candidates.append({
                "candidate_action_id": f"act_releases_{c.condition_id}",
                "action_type": "RELEASES",
                "target_belief_id": None,
                "target_condition_id": c.condition_id,
                "replacement_belief_id": None,
                "evidence_ids": [new_ev_id],
                "why_candidate": f"Based on evidence '{new_ev_snippet}...', evaluate whether to restore/release prerequisite {c.condition_id} ({c.text[:30]}) [depended on by belief {bids_str}]."
            })

        # 4. UNCERTAIN candidates & 5. REAFFIRMS candidates
        for b in submission.candidate_beliefs:
            candidates.append({
                "candidate_action_id": f"act_uncertain_{b.belief_id}",
                "action_type": "UNCERTAIN",
                "target_belief_id": b.belief_id,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "evidence_ids": [new_ev_id],
                "why_candidate": f"Based on evidence '{new_ev_snippet}...', evaluate whether to mark belief {b.belief_id} ({b.proposition[:30]}) as uncertain."
            })
            candidates.append({
                "candidate_action_id": f"act_reaffirms_{b.belief_id}",
                "action_type": "REAFFIRMS",
                "target_belief_id": b.belief_id,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "evidence_ids": [new_ev_id],
                "why_candidate": f"Based on evidence '{new_ev_snippet}...', evaluate whether to reaffirm/support belief {b.belief_id} ({b.proposition[:30]})."
            })

        # 6. NO_REVISION fallback
        candidates.append({
            "candidate_action_id": "act_no_revision",
            "action_type": "NO_REVISION",
            "target_belief_id": None,
            "target_condition_id": None,
            "replacement_belief_id": None,
            "evidence_ids": [new_ev_id],
            "why_candidate": "No revision action is warranted by the new evidence"
        })

        return candidates

    @staticmethod
    def validate_proposal(selected_action_ids: list[str]) -> bool:
        """Ensure NO_REVISION cannot be co-selected with other actions."""
        if "act_no_revision" in selected_action_ids and len(selected_action_ids) > 1:
            return False
        return True

def build_candidate_actions(submission: Any) -> list[dict[str, Any]]:
    """Build all structure-allowable candidate revision actions for a submission.
    
    Backward compatibility wrapper around CandidateActionBuilder.
    """
    return CandidateActionBuilder.build_candidates(submission)


def detect_local_conflict(submission: Any) -> tuple[bool, list[str], list[str]]:
    """Deterministically detect a same-scope belief-level conflict.

    A conflict is flagged only when the submission exhibits the structural
    signature of two competing beliefs about the same object with no safe
    structural resolution (no replacement to SUPERSEDES with, no condition to
    BLOCKS/RELEASES). Concretely, all of the following must hold:

    - There are at least two candidate beliefs.
    - There are no candidate replacement beliefs (no SUPERSEDES handle).
    - There are no candidate condition anchors (no BLOCKS/RELEASES handle).
    - The new evidence grounds at least one candidate belief while at least one
      other candidate belief is grounded only in prior evidence (i.e. the new
      submission introduces a belief that competes with a previously
      established one).

    This uses only method-visible structure (candidate beliefs, their
    source-evidence grounding, and the submission's new_evidence_id). It never
    reads gold targets, gold statuses, failure-type labels, or evaluator
    expectations.

    Returns ``(triggered, established_belief_ids, new_belief_ids)`` where
    ``established_belief_ids`` are the previously established beliefs that the
    new evidence does not ground (the contradicted/unresolved candidates) and
    ``new_belief_ids`` are the beliefs introduced by the new evidence.
    """
    candidate_beliefs = list(getattr(submission, "candidate_beliefs", ()) or ())
    if len(candidate_beliefs) < 2:
        return False, [], []
    if getattr(submission, "candidate_replacement_beliefs", ()):
        return False, [], []
    for _, conds in getattr(submission, "candidate_conditions_by_belief", ()) or ():
        if conds:
            return False, [], []

    new_ev_id = submission.new_evidence_id
    new_belief_ids = [
        b.belief_id for b in candidate_beliefs if new_ev_id in (b.source_evidence_ids or ())
    ]
    established_belief_ids = [
        b.belief_id for b in candidate_beliefs if new_ev_id not in (b.source_evidence_ids or ())
    ]
    if new_belief_ids and established_belief_ids:
        return True, established_belief_ids, new_belief_ids
    return False, [], []


def rename_string(s: str | None, old_ns: str, new_ns: str) -> str | None:
    if s is None:
        return None
    return s.replace(old_ns, new_ns)


def rename_submission(sub: Any, old_ns: str, new_ns: str) -> Any:
    """Helper to rewrite a submission to use the augmented namespace."""
    from retracemem.schemas import EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
    from retracemem.evaluation.multiagent.contracts import FixedCandidateSubmission

    new_evidence_context = []
    for ev in sub.evidence_context:
        new_evidence_context.append(
            EvidenceNode(
                evidence_id=rename_string(ev.evidence_id, old_ns, new_ns),
                session_id=rename_string(ev.session_id, old_ns, new_ns),
                timestamp=ev.timestamp,
                text=rename_string(ev.text, old_ns, new_ns),
                source_dataset=ev.source_dataset,
                source_pointer=ev.source_pointer,
                is_raw_source=ev.is_raw_source,
                metadata=ev.metadata,
            )
        )

    new_candidate_beliefs = []
    for b in sub.candidate_beliefs:
        new_candidate_beliefs.append(
            BeliefNode(
                belief_id=rename_string(b.belief_id, old_ns, new_ns),
                proposition=rename_string(b.proposition, old_ns, new_ns),
                source_evidence_ids=tuple(rename_string(eid, old_ns, new_ns) for eid in b.source_evidence_ids),
                source_span=b.source_span,
                extractor_version=b.extractor_version,
                confidence=b.confidence,
                metadata=b.metadata,
            )
        )

    new_candidate_replacement_beliefs = []
    for b in sub.candidate_replacement_beliefs:
        new_candidate_replacement_beliefs.append(
            BeliefNode(
                belief_id=rename_string(b.belief_id, old_ns, new_ns),
                proposition=rename_string(b.proposition, old_ns, new_ns),
                source_evidence_ids=tuple(rename_string(eid, old_ns, new_ns) for eid in b.source_evidence_ids),
                source_span=b.source_span,
                extractor_version=b.extractor_version,
                confidence=b.confidence,
                metadata=b.metadata,
            )
        )

    new_candidate_conditions = []
    for bid, conds in sub.candidate_conditions_by_belief:
        new_conds = []
        for c in conds:
            new_conds.append(
                ConditionNode(
                    condition_id=rename_string(c.condition_id, old_ns, new_ns),
                    scope_id=rename_string(c.scope_id, old_ns, new_ns),
                    text=rename_string(c.text, old_ns, new_ns),
                    metadata=c.metadata,
                )
            )
        new_candidate_conditions.append(
            (rename_string(bid, old_ns, new_ns), tuple(new_conds))
        )

    new_dependency_edges = []
    for bid, deps in sub.dependency_edges_by_belief:
        new_deps = []
        for d in deps:
            new_deps.append(
                DependencyEdge(
                    edge_id=rename_string(d.edge_id, old_ns, new_ns),
                    edge_type=d.edge_type,
                    belief_id=rename_string(d.belief_id, old_ns, new_ns),
                    condition_id=rename_string(d.condition_id, old_ns, new_ns),
                    inducer=d.inducer,
                    supporting_evidence_ids=tuple(rename_string(eid, old_ns, new_ns) for eid in d.supporting_evidence_ids),
                    model_call_trace_id=d.model_call_trace_id,
                    confidence=d.confidence,
                    rationale=rename_string(d.rationale, old_ns, new_ns),
                    metadata=d.metadata,
                )
            )
        new_dependency_edges.append(
            (rename_string(bid, old_ns, new_ns), tuple(new_deps))
        )

    return FixedCandidateSubmission(
        submission_id=rename_string(sub.submission_id, old_ns, new_ns),
        producer_id=sub.producer_id,
        producer_role=sub.producer_role,
        task_id=rename_string(sub.task_id, old_ns, new_ns),
        parent_snapshot_id=rename_string(sub.parent_snapshot_id, old_ns, new_ns),
        observed_at=sub.observed_at,
        instance_id=rename_string(sub.instance_id, old_ns, new_ns),
        query_id=rename_string(sub.query_id, old_ns, new_ns),
        query=rename_string(sub.query, old_ns, new_ns),
        evidence_context=tuple(new_evidence_context),
        new_evidence_id=rename_string(sub.new_evidence_id, old_ns, new_ns),
        candidate_beliefs=tuple(new_candidate_beliefs),
        candidate_replacement_beliefs=tuple(new_candidate_replacement_beliefs),
        candidate_conditions_by_belief=tuple(new_candidate_conditions),
        dependency_edges_by_belief=tuple(new_dependency_edges),
        metadata=sub.metadata,
    )


def format_assistant_response(ex: Any) -> str:
    """Format targets into a strict JSON string."""
    import json
    targets_dict = []
    new_evidence_id = ex.method_visible_input.new_evidence_id
    for t in ex.targets:
        if t.action_type == "NO_REVISION":
            ev_ids = list(t.evidence_ids) if t.evidence_ids else [new_evidence_id]
            targets_dict.append({
                "action_type": "NO_REVISION",
                "target_belief_id": None,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "rationale": t.rationale or "No evidence-grounded revision is warranted.",
                "evidence_ids": ev_ids,
            })
        else:
            targets_dict.append({
                "action_type": t.action_type,
                "target_belief_id": t.target_belief_id,
                "target_condition_id": t.target_condition_id,
                "replacement_belief_id": t.replacement_belief_id,
                "rationale": t.rationale,
                "evidence_ids": list(t.evidence_ids),
            })
    return json.dumps(targets_dict, indent=2)


def format_user_prompt(ex: Any) -> str:
    """Format the method-visible context of a submission for the user message."""
    sub = ex.method_visible_input
    
    # 1. Submission Metadata
    meta_lines = [
        f"- Submission ID: {sub.submission_id}",
        f"- Producer ID: {sub.producer_id}",
        f"- Producer Role: {sub.producer_role}",
        f"- Observed At: {sub.observed_at}",
        f"- Parent Snapshot ID: {sub.parent_snapshot_id}",
    ]
    if sub.task_id:
        meta_lines.append(f"- Task ID: {sub.task_id}")
        
    # 2. Evidence Context
    evidence_lines = []
    for ev in sub.evidence_context:
        source_ptr_str = f", Source: {ev.source_pointer}" if hasattr(ev, "source_pointer") and ev.source_pointer else ""
        timestamp_str = f", Timestamp: {ev.timestamp}" if hasattr(ev, "timestamp") and ev.timestamp else ""
        evidence_lines.append(f"- ID: {ev.evidence_id}{timestamp_str}{source_ptr_str}, Content: {ev.text}")
        
    # 3. Candidate Beliefs
    candidates = []
    for b in sub.candidate_beliefs:
        ev_ids_str = ", ".join(b.source_evidence_ids) if b.source_evidence_ids else "None"
        candidates.append(f"- ID: {b.belief_id}, Proposition: {b.proposition}, Source Evidence: [{ev_ids_str}]")
        
    # 4. Candidate Replacement Beliefs
    replacements = []
    for b in sub.candidate_replacement_beliefs:
        ev_ids_str = ", ".join(b.source_evidence_ids) if b.source_evidence_ids else "None"
        replacements.append(f"- ID: {b.belief_id}, Proposition: {b.proposition}, Source Evidence: [{ev_ids_str}]")
        
    # 5. Candidate Conditions by Belief
    conditions = []
    for bid, conds in sub.candidate_conditions_by_belief:
        for c in conds:
            conditions.append(f"- Owning Belief: {bid}, Condition ID: {c.condition_id}, Scope: {c.scope_id}, Text: {c.text}")
            
    # 6. Pre-existing Dependency Anchors
    dependencies = []
    for bid, deps in sub.dependency_edges_by_belief:
        for d in deps:
            dependencies.append(f"- {d.belief_id} --REQUIRES--> {d.condition_id}")
            
    prompt = (
        f"Episode ID: {ex.episode_id}\n\n"
        "Submission Metadata:\n" + "\n".join(meta_lines) + "\n\n"
        f"Query: {sub.query}\n\n"
        "Evidence Context:\n" + ("\n".join(evidence_lines) if evidence_lines else "None") + "\n\n"
        f"New Evidence ID: {sub.new_evidence_id}\n\n"
        "Candidate Beliefs:\n" + ("\n".join(candidates) if candidates else "None") + "\n\n"
        "Candidate Replacement Beliefs:\n" + ("\n".join(replacements) if replacements else "None") + "\n\n"
        "Candidate Conditions by Belief:\n" + ("\n".join(conditions) if conditions else "None") + "\n\n"
        "Pre-existing Dependency Anchors:\n" + ("\n".join(dependencies) if dependencies else "None") + "\n\n"
        "Identify the correct revision actions. Return a strict JSON array of objects."
    )
    return prompt
