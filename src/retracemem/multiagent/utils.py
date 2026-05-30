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
                    "why_candidate": f"依据证据 '{new_ev_snippet}...'，评估是否将旧信念 {b_old.belief_id} ({b_old.proposition[:30]}) 替换为新信念 {b_new.belief_id} ({b_new.proposition[:30]})。"
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
                "why_candidate": f"依据证据 '{new_ev_snippet}...'，评估是否失效先决条件 {c.condition_id} ({c.text[:30]}) [被信念 {bids_str} 依赖]。"
            })
            candidates.append({
                "candidate_action_id": f"act_releases_{c.condition_id}",
                "action_type": "RELEASES",
                "target_belief_id": None,
                "target_condition_id": c.condition_id,
                "replacement_belief_id": None,
                "evidence_ids": [new_ev_id],
                "why_candidate": f"依据证据 '{new_ev_snippet}...'，评估是否恢复/释领先决条件 {c.condition_id} ({c.text[:30]}) [被信念 {bids_str} 依赖]。"
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
                "why_candidate": f"依据证据 '{new_ev_snippet}...'，评估是否将信念 {b.belief_id} ({b.proposition[:30]}) 标记为不确定。"
            })
            candidates.append({
                "candidate_action_id": f"act_reaffirms_{b.belief_id}",
                "action_type": "REAFFIRMS",
                "target_belief_id": b.belief_id,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "evidence_ids": [new_ev_id],
                "why_candidate": f"依据证据 '{new_ev_snippet}...'，评估是否重新确认/支持信念 {b.belief_id} ({b.proposition[:30]})。"
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


def rename_string(s: str | None, old_ns: str, new_ns: str) -> str | None:
    if s is None:
        return None
    return s.replace(old_ns, new_ns)


def rename_submission(sub: Any, old_ns: str, new_ns: str) -> Any:
    """Helper to rewrite a submission to use the augmented namespace."""
    from retracemem.schemas import EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
    from experiments.multiagent.contracts import FixedCandidateSubmission

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
