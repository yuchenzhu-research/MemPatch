from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple
from retracemem.schemas import EvidenceEdgeType, EvidenceEdge
from retracemem.authorization import EvidenceProposalBatch
from retracemem.multiagent.commit import commit_subagent_submission
from retracemem.multiagent.contracts import SubagentMemorySubmission
from retracemem.evaluation.multiagent.contracts import (
    MultiAgentMemoryEpisode,
    EpisodeMethodResult,
    RevisionEventResult,
    FixedCandidateInputEpisode,
    FixedCandidateEpisodeMethodResult,
    MethodDecisionRecord,
    MethodDecisionArtifact,
)


class SharedMemoryMethod(Protocol):
    @property
    def method_name(self) -> str:
        ...

    def run_episode(self, episode: MultiAgentMemoryEpisode) -> EpisodeMethodResult:
        ...


class FixedCandidateMethod(Protocol):
    """Protocol for methods operating on fixed-candidate episodes."""
    @property
    def method_name(self) -> str:
        ...

    def run_fixed_episode(self, episode: FixedCandidateInputEpisode) -> FixedCandidateEpisodeMethodResult:
        ...


class NaiveLastWriteWinsMethod:
    """A deterministic naive baseline that ignores blocker/uncertain graph logic.
    
    It blindly authorizes candidate beliefs and replacement beliefs, marking any
    superseded targets as SUPERSEDED.
    """
    method_name: str = "Naive_LWW"

    def run_episode(self, episode: MultiAgentMemoryEpisode) -> EpisodeMethodResult:
        active_statuses: Dict[str, str] = {}
        events: List[RevisionEventResult] = []

        for sub in episode.submissions:
            # 1. Authorize candidate beliefs
            for b in sub.candidate_beliefs:
                active_statuses[b.belief_id] = "AUTHORIZED"
            
            for b in sub.candidate_replacement_beliefs:
                active_statuses[b.belief_id] = "AUTHORIZED"

            # 2. Process SUPERSEDES edges naively
            for batch in sub.proposal_batches:
                for edge in batch.edges:
                    if edge.edge_type == EvidenceEdgeType.SUPERSEDES:
                        active_statuses[edge.target_id] = "SUPERSEDED"
                        if edge.replacement_belief_id:
                            active_statuses[edge.replacement_belief_id] = "AUTHORIZED"

            # Log this event
            events.append(
                RevisionEventResult(
                    submission_id=sub.submission_id,
                    producer_id=sub.producer_id,
                    producer_role=sub.producer_role,
                    parent_snapshot_id=sub.parent_snapshot_id,
                    next_snapshot_id=None,
                    belief_statuses=dict(active_statuses),
                    trace_available=False,
                )
            )

        # Naive predictions for downstream tasks
        task_predictions: Dict[str, str] = {}
        for task in episode.downstream_tasks:
            # Check the status of relevant belief ids
            authorized_any = any(
                active_statuses.get(bid) == "AUTHORIZED"
                for bid in task.relevant_belief_ids
            )
            task_predictions[task.task_id] = "AUTHORIZED" if authorized_any else "UNAUTHORIZED"

        return EpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type,
            method_name=self.method_name,
            final_belief_statuses=dict(active_statuses),
            revision_events=tuple(events),
            task_predictions=task_predictions,
        )


class ReTraceMethod:
    """Processes submissions sequentially using commit_subagent_submission."""
    method_name: str = "ReTrace"

    def run_episode(self, episode: MultiAgentMemoryEpisode) -> EpisodeMethodResult:
        events: List[RevisionEventResult] = []
        final_statuses: Dict[str, str] = {}

        for sub in episode.submissions:
            commit_res = commit_subagent_submission(sub)
            
            # Extract fine-grained statuses from ReTrace auth trace
            auth_trace = commit_res.authorization_result.trace
            fine_grained = auth_trace.get("fine_grained_statuses", {})
            for bid, status in fine_grained.items():
                final_statuses[bid] = status

            events.append(
                RevisionEventResult(
                    submission_id=sub.submission_id,
                    producer_id=sub.producer_id,
                    producer_role=sub.producer_role,
                    parent_snapshot_id=sub.parent_snapshot_id,
                    next_snapshot_id=commit_res.next_snapshot_id,
                    belief_statuses=dict(final_statuses),
                    trace_available=True,
                )
            )

        # Task predictions based on authorized status
        task_predictions: Dict[str, str] = {}
        for task in episode.downstream_tasks:
            authorized_any = any(
                final_statuses.get(bid) == "AUTHORIZED"
                for bid in task.relevant_belief_ids
            )
            task_predictions[task.task_id] = "AUTHORIZED" if authorized_any else "UNAUTHORIZED"

        return EpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type,
            method_name=self.method_name,
            final_belief_statuses=final_statuses,
            revision_events=tuple(events),
            task_predictions=task_predictions,
        )


# ===========================================================================
# Fixed-Candidate Methods (Packet 4B)
# ===========================================================================


def _compute_task_predictions(
    episode: FixedCandidateInputEpisode,
    statuses: Dict[str, str],
) -> Dict[str, str]:
    """Shared helper: predict AUTHORIZED/UNAUTHORIZED per downstream task."""
    preds: Dict[str, str] = {}
    for task in episode.downstream_tasks:
        authorized_any = any(
            statuses.get(bid) == "AUTHORIZED"
            for bid in task.relevant_belief_ids
        )
        preds[task.task_id] = "AUTHORIZED" if authorized_any else "UNAUTHORIZED"
    return preds


class NaiveLastWriteWinsFixedCandidateMethod:
    """Deterministic baseline: Blindly authorizes all candidates, naively overwriting previous beliefs sharing the same slot."""
    method_name: str = "Naive_LWW_FC"

    def run_fixed_episode(self, episode: FixedCandidateInputEpisode) -> FixedCandidateEpisodeMethodResult:
        statuses: Dict[str, str] = {}
        decisions: List[MethodDecisionRecord] = []

        def _get_slot_key(prop: str) -> str:
            # Extract first 2 words as slot key for LWW matching
            words = prop.lower().split()
            if len(words) >= 2:
                return " ".join(words[:2])
            return prop.lower()

        for sub in episode.submissions:
            # First authorize candidates
            for b in sub.candidate_beliefs:
                statuses[b.belief_id] = "AUTHORIZED"
                decisions.append(MethodDecisionRecord(
                    belief_id=b.belief_id,
                    decision="AUTHORIZE",
                    rationale="LWW: blindly accept candidates",
                ))

            # Then process replacements by slot key
            for rep_b in sub.candidate_replacement_beliefs:
                rep_slot = _get_slot_key(rep_b.proposition)
                # Look for prior authorized beliefs to overwrite
                for prior_id, status in list(statuses.items()):
                    if prior_id == rep_b.belief_id:
                        continue
                    if status == "AUTHORIZED":
                        prior_prop = None
                        for s_prev in episode.submissions:
                            for c_b in s_prev.candidate_beliefs:
                                if c_b.belief_id == prior_id:
                                    prior_prop = c_b.proposition
                                    break
                            if prior_prop:
                                break
                        
                        if prior_prop and _get_slot_key(prior_prop) == rep_slot:
                            statuses[prior_id] = "SUPERSEDED"
                            decisions.append(MethodDecisionRecord(
                                belief_id=prior_id,
                                decision="SUPERSEDE",
                                replacement_belief_id=rep_b.belief_id,
                                rationale=f"LWW: superseded by newer slot update {rep_b.belief_id}",
                            ))
                statuses[rep_b.belief_id] = "AUTHORIZED"

        return FixedCandidateEpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type_public_or_controlled,
            method_name=self.method_name,
            protocol_mode=episode.protocol_mode,
            proposal_source=episode.proposal_source,
            final_belief_statuses=dict(statuses),
            decisions=tuple(decisions),
            task_predictions=_compute_task_predictions(episode, statuses),
        )


class AppendOnlyLexicalTopKMethod:
    """Keeps top-K candidate beliefs by true token-overlap lexical similarity against query."""
    method_name: str = "AppendOnly_TopK"

    def __init__(self, k: int = 5) -> None:
        self._k = k

    def run_fixed_episode(self, episode: FixedCandidateInputEpisode) -> FixedCandidateEpisodeMethodResult:
        all_beliefs: List[Tuple[BeliefNode, str]] = []
        seen: set[str] = set()
        
        # Collect unique candidate beliefs with query from submission
        for sub in episode.submissions:
            for b in sub.candidate_beliefs:
                if b.belief_id not in seen:
                    all_beliefs.append((b, sub.query))
                    seen.add(b.belief_id)

        # Lexical score calculation
        def _get_overlap_score(prop: str, query: str) -> float:
            import string
            translator = str.maketrans('', '', string.punctuation)
            q_words = set(query.translate(translator).lower().split())
            p_words = set(prop.translate(translator).lower().split())
            return float(len(q_words & p_words))

        # Score and sort beliefs (score descending, then lexicographical order for tie-breaking)
        scored_beliefs = []
        for b, query in all_beliefs:
            score = _get_overlap_score(b.proposition, query)
            scored_beliefs.append((b.belief_id, score))

        scored_beliefs.sort(key=lambda x: (x[1], x[0]), reverse=True)

        statuses: Dict[str, str] = {}
        decisions: List[MethodDecisionRecord] = []
        for idx, (bid, score) in enumerate(scored_beliefs):
            if idx < self._k:
                statuses[bid] = "AUTHORIZED"
                decisions.append(MethodDecisionRecord(
                    belief_id=bid,
                    decision="AUTHORIZE",
                    rationale=f"LexicalTopK: score={score} rank={idx} (k={self._k})",
                ))
            else:
                statuses[bid] = "SUPERSEDED"
                decisions.append(MethodDecisionRecord(
                    belief_id=bid,
                    decision="REJECT",
                    rationale=f"LexicalTopK: rank={idx} exceeds k={self._k}",
                ))

        return FixedCandidateEpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type_public_or_controlled,
            method_name=self.method_name,
            protocol_mode=episode.protocol_mode,
            proposal_source=episode.proposal_source,
            final_belief_statuses=dict(statuses),
            decisions=tuple(decisions),
            task_predictions=_compute_task_predictions(episode, statuses),
        )


class DirectJudgeReplayMethod:
    """Replays direct verdicts from the sidecar MethodDecisionArtifact."""
    method_name: str = "DirectJudge_Replay"

    def run_fixed_episode(
        self,
        episode: FixedCandidateInputEpisode,
        artifact: MethodDecisionArtifact | None = None
    ) -> FixedCandidateEpisodeMethodResult:
        statuses: Dict[str, str] = {}
        decisions = []

        if artifact:
            # Flatten direct verdicts from all submissions
            for sub_id, sub_verdicts in artifact.direct_verdicts_by_submission:
                for dec in sub_verdicts:
                    dec_rec = dec
                    if isinstance(dec, dict):
                        dec_rec = MethodDecisionRecord(**dec)
                    
                    decisions.append(dec_rec)
                    if dec_rec.decision == "AUTHORIZE":
                        statuses[dec_rec.belief_id] = "AUTHORIZED"
                    elif dec_rec.decision == "REJECT":
                        statuses[dec_rec.belief_id] = "BLOCKED"
                    elif dec_rec.decision == "SUPERSEDE":
                        statuses[dec_rec.belief_id] = "SUPERSEDED"
                        if dec_rec.replacement_belief_id:
                            statuses[dec_rec.replacement_belief_id] = "AUTHORIZED"
                    elif dec_rec.decision == "DEFER":
                        statuses[dec_rec.belief_id] = "UNRESOLVED"

        # Fill default for candidates not covered
        for sub in episode.submissions:
            for b in sub.candidate_beliefs:
                if b.belief_id not in statuses:
                    statuses[b.belief_id] = "AUTHORIZED"
                    decisions.append(MethodDecisionRecord(
                        belief_id=b.belief_id,
                        decision="AUTHORIZE",
                        rationale="DirectJudge: default authorized",
                    ))

        return FixedCandidateEpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type_public_or_controlled,
            method_name=self.method_name,
            protocol_mode=episode.protocol_mode,
            proposal_source=episode.proposal_source,
            final_belief_statuses=dict(statuses),
            decisions=tuple(decisions),
            task_predictions=_compute_task_predictions(episode, statuses),
        )


class ReTraceProposalReplayMethod:
    """Replays proposal batches through the deterministic DPA/authorization kernel."""
    method_name: str = "ReTrace_StageA_Replay"

    def run_fixed_episode(
        self,
        episode: FixedCandidateInputEpisode,
        artifact: MethodDecisionArtifact | None = None
    ) -> FixedCandidateEpisodeMethodResult:
        
        # Build proposal batch mapping
        proposal_map = {}
        if artifact:
            for sub_id, batches in artifact.typed_proposal_batches_by_submission:
                proposal_map[sub_id] = batches

        subagent_subs = []
        for sub in episode.submissions:
            # 1. Fetch proposal batches for this submission
            batches = proposal_map.get(sub.submission_id, ())
            
            # Translate dictionary to EvidenceProposalBatch if serialized
            clean_batches = []
            for b in batches:
                if isinstance(b, dict):
                    # Deserialize edges
                    edges = tuple(
                        EvidenceEdge(
                            edge_id=e["edge_id"],
                            edge_type=EvidenceEdgeType(e["edge_type"]) if hasattr(EvidenceEdgeType, e["edge_type"]) else e["edge_type"],
                            evidence_id=e["evidence_id"],
                            target_kind=e["target_kind"],
                            target_id=e["target_id"],
                            verifier=e["verifier"],
                            replacement_belief_id=e.get("replacement_belief_id"),
                        )
                        for e in b["edges"]
                    )
                    clean_batches.append(
                        EvidenceProposalBatch(
                            edges=edges,
                            model_call_trace_id=b.get("model_call_trace_id"),
                            source_belief_id=b.get("source_belief_id"),
                            metadata=b.get("metadata", {}),
                        )
                    )
                else:
                    clean_batches.append(b)

            # 2. Build SubagentMemorySubmission
            subagent_sub = SubagentMemorySubmission(
                submission_id=sub.submission_id,
                producer_id=sub.producer_id,
                producer_role=sub.producer_role,
                parent_snapshot_id=sub.parent_snapshot_id,
                observed_at=sub.observed_at,
                instance_id=sub.instance_id,
                query_id=sub.query_id,
                query=sub.query,
                evidence_context=sub.evidence_context,
                new_evidence_id=sub.new_evidence_id,
                candidate_beliefs=sub.candidate_beliefs,
                candidate_replacement_beliefs=sub.candidate_replacement_beliefs,
                candidate_conditions_by_belief=sub.candidate_conditions_by_belief,
                dependency_edges_by_belief=sub.dependency_edges_by_belief,
                proposal_batches=tuple(clean_batches),
                task_id=sub.task_id,
                metadata=sub.metadata,
            )
            subagent_subs.append(subagent_sub)

        # 3. Call the sequence commit engine in multiagent core
        from retracemem.multiagent.commit import commit_submission_sequence
        seq_res = commit_submission_sequence(tuple(subagent_subs), final_snapshot_evaluation=True)
        active_statuses = seq_res.final_belief_statuses

        decisions: List[MethodDecisionRecord] = []
        # Build method decision records based on final statuses
        for bid, status in active_statuses.items():
            dec = "AUTHORIZE"
            if status == "BLOCKED":
                dec = "REJECT"
            elif status == "SUPERSEDED":
                dec = "SUPERSEDE"
            elif status == "UNRESOLVED":
                dec = "DEFER"
            
            # Find replacement if superseded
            rep_id = None
            if dec == "SUPERSEDE":
                # Find in proposal batches if any replacement was proposed
                for sub_id, batches in proposal_map.items():
                    for batch in batches:
                        edge_list = batch.edges if not isinstance(batch, dict) else batch.get("edges", [])
                        for edge in edge_list:
                            e_tid = edge.target_id if not isinstance(edge, dict) else edge.get("target_id")
                            if e_tid == bid:
                                rep_id = edge.replacement_belief_id if not isinstance(edge, dict) else edge.get("replacement_belief_id")
                                break
                        if rep_id:
                            break
                    if rep_id:
                        break

            decisions.append(
                MethodDecisionRecord(
                    belief_id=bid,
                    decision=dec,
                    replacement_belief_id=rep_id,
                    rationale=f"DPA: determined status {status}",
                )
            )

        return FixedCandidateEpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type_public_or_controlled,
            method_name=self.method_name,
            protocol_mode=episode.protocol_mode,
            proposal_source=episode.proposal_source,
            final_belief_statuses=dict(active_statuses),
            decisions=tuple(decisions),
            task_predictions=_compute_task_predictions(episode, active_statuses),
        )

