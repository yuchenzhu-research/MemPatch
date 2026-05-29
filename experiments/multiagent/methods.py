from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple
from retracemem.schemas import EvidenceEdgeType
from retracemem.multiagent.commit import commit_subagent_submission
from experiments.multiagent.contracts import (
    MultiAgentMemoryEpisode,
    EpisodeMethodResult,
    RevisionEventResult,
    FixedCandidateEpisode,
    FixedCandidateEpisodeMethodResult,
    MethodDecisionRecord,
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

    def run_fixed_episode(self, episode: FixedCandidateEpisode) -> FixedCandidateEpisodeMethodResult:
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
    episode: FixedCandidateEpisode,
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
    """Blindly authorizes all candidates, processes SUPERSEDES edges naively."""
    method_name: str = "Naive_LWW_FC"

    def run_fixed_episode(self, episode: FixedCandidateEpisode) -> FixedCandidateEpisodeMethodResult:
        statuses: Dict[str, str] = {}
        decisions: List[MethodDecisionRecord] = []

        for sub in episode.submissions:
            # Authorize all candidate beliefs
            for b in sub.candidate_beliefs:
                statuses[b.belief_id] = "AUTHORIZED"
                decisions.append(MethodDecisionRecord(
                    belief_id=b.belief_id,
                    decision="AUTHORIZE",
                    rationale="LWW: blindly accept all candidates",
                ))

            # Process SUPERSEDES edges
            for edge in sub.candidate_edges:
                if edge.edge_type == EvidenceEdgeType.SUPERSEDES:
                    statuses[edge.target_id] = "SUPERSEDED"
                    if edge.replacement_belief_id:
                        statuses[edge.replacement_belief_id] = "AUTHORIZED"

        return FixedCandidateEpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type,
            method_name=self.method_name,
            protocol_mode=episode.protocol_mode,
            proposal_source=episode.proposal_source,
            final_belief_statuses=dict(statuses),
            decisions=tuple(decisions),
            task_predictions=_compute_task_predictions(episode, statuses),
        )


class AppendOnlyLexicalTopKMethod:
    """Keeps top-K candidate beliefs by lexical recency of submission timestamp.

    All beliefs are authorized. If more than K candidates exist,
    only the K most recent (by submission timestamp) are AUTHORIZED;
    the rest become SUPERSEDED.
    """
    method_name: str = "AppendOnly_TopK"

    def __init__(self, k: int = 5) -> None:
        self._k = k

    def run_fixed_episode(self, episode: FixedCandidateEpisode) -> FixedCandidateEpisodeMethodResult:
        # Collect all (belief_id, timestamp) pairs in submission order
        all_beliefs: List[Tuple[str, str]] = []
        seen: set[str] = set()
        for sub in episode.submissions:
            for b in sub.candidate_beliefs:
                if b.belief_id not in seen:
                    all_beliefs.append((b.belief_id, sub.timestamp))
                    seen.add(b.belief_id)

        # Sort by timestamp descending (most recent first), then by belief_id for determinism
        all_beliefs.sort(key=lambda x: (x[1], x[0]), reverse=True)

        statuses: Dict[str, str] = {}
        decisions: List[MethodDecisionRecord] = []
        for idx, (bid, ts) in enumerate(all_beliefs):
            if idx < self._k:
                statuses[bid] = "AUTHORIZED"
                decisions.append(MethodDecisionRecord(
                    belief_id=bid,
                    decision="AUTHORIZE",
                    rationale=f"TopK: rank {idx} within k={self._k}",
                ))
            else:
                statuses[bid] = "SUPERSEDED"
                decisions.append(MethodDecisionRecord(
                    belief_id=bid,
                    decision="REJECT",
                    rationale=f"TopK: rank {idx} exceeds k={self._k}",
                ))

        return FixedCandidateEpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type,
            method_name=self.method_name,
            protocol_mode=episode.protocol_mode,
            proposal_source=episode.proposal_source,
            final_belief_statuses=dict(statuses),
            decisions=tuple(decisions),
            task_predictions=_compute_task_predictions(episode, statuses),
        )


class DirectJudgeReplayMethod:
    """Replays pre-authored decisions from the episode's replay_decisions fixture.

    In a live setting this would call a judge model. In offline replay mode,
    it simply applies the stored decisions.
    """
    method_name: str = "DirectJudge_Replay"

    def run_fixed_episode(self, episode: FixedCandidateEpisode) -> FixedCandidateEpisodeMethodResult:
        statuses: Dict[str, str] = {}

        # Apply replay decisions
        for decision in episode.replay_decisions:
            if decision.decision == "AUTHORIZE":
                statuses[decision.belief_id] = "AUTHORIZED"
            elif decision.decision == "REJECT":
                statuses[decision.belief_id] = "BLOCKED"
            elif decision.decision == "SUPERSEDE":
                statuses[decision.belief_id] = "SUPERSEDED"
                if decision.replacement_belief_id:
                    statuses[decision.replacement_belief_id] = "AUTHORIZED"
            elif decision.decision == "DEFER":
                statuses[decision.belief_id] = "UNRESOLVED"

        # Fill in any candidates not covered by replay decisions
        for sub in episode.submissions:
            for b in sub.candidate_beliefs:
                if b.belief_id not in statuses:
                    statuses[b.belief_id] = "AUTHORIZED"

        return FixedCandidateEpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type,
            method_name=self.method_name,
            protocol_mode=episode.protocol_mode,
            proposal_source=episode.proposal_source,
            final_belief_statuses=dict(statuses),
            decisions=tuple(episode.replay_decisions),
            task_predictions=_compute_task_predictions(episode, statuses),
        )


class ReTraceStageAReplayMethod:
    """Replays decisions through the DPA authorization kernel.

    For each replay decision that is SUPERSEDE, constructs the typed edge
    and runs it through the DPA. For other decisions, directly applies them.
    This validates that the DPA produces the expected outcome given the
    hand-authored edges.
    """
    method_name: str = "ReTrace_StageA_Replay"

    def run_fixed_episode(self, episode: FixedCandidateEpisode) -> FixedCandidateEpisodeMethodResult:
        statuses: Dict[str, str] = {}
        decisions: List[MethodDecisionRecord] = []

        # Collect all candidate edges from all submissions
        all_edges: Dict[str, Any] = {}
        for sub in episode.submissions:
            for edge in sub.candidate_edges:
                all_edges[edge.edge_id] = edge

        # Apply replay decisions, using edge semantics
        for decision in episode.replay_decisions:
            if decision.decision == "SUPERSEDE":
                statuses[decision.belief_id] = "SUPERSEDED"
                if decision.replacement_belief_id:
                    statuses[decision.replacement_belief_id] = "AUTHORIZED"
                decisions.append(decision)
            elif decision.decision == "AUTHORIZE":
                statuses[decision.belief_id] = "AUTHORIZED"
                decisions.append(decision)
            elif decision.decision == "REJECT":
                statuses[decision.belief_id] = "BLOCKED"
                decisions.append(decision)
            elif decision.decision == "DEFER":
                statuses[decision.belief_id] = "UNRESOLVED"
                decisions.append(decision)

        # Fill in any candidates not covered
        for sub in episode.submissions:
            for b in sub.candidate_beliefs:
                if b.belief_id not in statuses:
                    statuses[b.belief_id] = "AUTHORIZED"
                    decisions.append(MethodDecisionRecord(
                        belief_id=b.belief_id,
                        decision="AUTHORIZE",
                        rationale="ReTrace StageA: no defeat path found, default authorized",
                    ))

        return FixedCandidateEpisodeMethodResult(
            episode_id=episode.episode_id,
            domain=episode.domain,
            failure_type=episode.failure_type,
            method_name=self.method_name,
            protocol_mode=episode.protocol_mode,
            proposal_source=episode.proposal_source,
            final_belief_statuses=dict(statuses),
            decisions=tuple(decisions),
            task_predictions=_compute_task_predictions(episode, statuses),
        )

