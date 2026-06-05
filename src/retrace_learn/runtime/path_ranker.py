"""Future/optional: Learned defeat-path ranker (safe, auditable).

Not part of the v1 scaffold commit path; advisory, never on the commit path.

This does **not** replace DPA and the LLM never emits a final status directly.
Instead the ranker scores *DPA-legal* candidate defeat paths and selects one.
Safety invariants enforced here:

* the selected path must be one of the enumerated legal candidates (no illegal
  path can ever be chosen);
* every candidate maps deterministically to a DPA final status, so the runtime
  can replay the decision and audit it;
* ``rationale`` is advisory only and never used to decide the outcome.

Candidate path types (:data:`retrace_learn.schemas.CANDIDATE_PATH_TYPES`):
``DIRECT_SUPERSEDE``, ``PREREQUISITE_BLOCK``, ``UNRESOLVED_UNCERTAIN``,
``AUTHORIZED_DEFAULT``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from retracemem.schemas import AuthorizationStatus, DefeatPathType

from retrace_learn.runtime.dpa_runtime import RuntimeResult

# Map each candidate path type to the DPA final status it implies. This is the
# replay contract: selecting a path is equivalent to asserting this status.
PATH_TYPE_TO_STATUS: dict[str, str] = {
    DefeatPathType.DIRECT_SUPERSEDE.value: AuthorizationStatus.SUPERSEDED.value,
    DefeatPathType.PREREQUISITE_BLOCK.value: AuthorizationStatus.BLOCKED.value,
    DefeatPathType.UNRESOLVED_UNCERTAIN.value: AuthorizationStatus.UNRESOLVED.value,
    "AUTHORIZED_DEFAULT": AuthorizationStatus.AUTHORIZED.value,
}

# Canonical precedence priors (higher = stronger defeat), mirroring DPA:
# SUPERSEDES > PREREQUISITE_BLOCK > UNRESOLVED_UNCERTAIN > AUTHORIZED.
_PRECEDENCE_PRIOR: dict[str, float] = {
    DefeatPathType.DIRECT_SUPERSEDE.value: 3.0,
    DefeatPathType.PREREQUISITE_BLOCK.value: 2.0,
    DefeatPathType.UNRESOLVED_UNCERTAIN.value: 1.0,
    "AUTHORIZED_DEFAULT": 0.0,
}


@dataclass(frozen=True)
class CandidatePath:
    path_type: str
    target_belief_id: str
    supporting_evidence_edge_ids: tuple[str, ...] = ()
    supporting_dependency_edge_ids: tuple[str, ...] = ()
    replacement_belief_id: str | None = None
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def implied_status(self) -> str:
        return PATH_TYPE_TO_STATUS[self.path_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_type": self.path_type,
            "target_belief_id": self.target_belief_id,
            "supporting_evidence_edge_ids": list(self.supporting_evidence_edge_ids),
            "supporting_dependency_edge_ids": list(self.supporting_dependency_edge_ids),
            "replacement_belief_id": self.replacement_belief_id,
            "implied_status": self.implied_status,
        }


@dataclass(frozen=True)
class RankResult:
    target_belief_id: str
    selected: CandidatePath
    rejected: tuple[CandidatePath, ...]
    scores: dict[str, float]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_belief_id": self.target_belief_id,
            "selected": self.selected.to_dict(),
            "rejected": [p.to_dict() for p in self.rejected],
            "scores": self.scores,
            "rationale": self.rationale,
        }


def legal_paths_from_runtime(result: RuntimeResult) -> dict[str, list[CandidatePath]]:
    """Enumerate DPA-legal candidate paths per belief from a runtime result.

    ``AUTHORIZED_DEFAULT`` is always a legal candidate. Any accepted defeat path
    in the DPA trace is added as the corresponding defeat candidate. This keeps
    the ranker constrained to the kernel's own admissible set.
    """
    accepted_by_belief: dict[str, dict[str, Any]] = {
        dp["belief_id"]: dp for dp in result.defeat_paths
    }
    candidates: dict[str, list[CandidatePath]] = {}
    for belief_id in result.final_belief_statuses:
        paths = [CandidatePath(path_type="AUTHORIZED_DEFAULT", target_belief_id=belief_id)]
        dp = accepted_by_belief.get(belief_id)
        if dp is not None:
            paths.append(
                CandidatePath(
                    path_type=dp["path_type"],
                    target_belief_id=belief_id,
                    supporting_evidence_edge_ids=tuple(dp.get("evidence_edge_ids", [])),
                    supporting_dependency_edge_ids=tuple(dp.get("dependency_edge_ids", [])),
                    replacement_belief_id=dp.get("replacement_belief_id"),
                )
            )
        candidates[belief_id] = paths
    return candidates


SCORE_FN = Callable[[CandidatePath, dict[str, Any]], float]


class HeuristicPathRanker:
    """Deterministic ranker using canonical precedence priors.

    By construction it agrees with DPA: among the legal candidates it prefers the
    highest-precedence defeat path that is present, falling back to
    ``AUTHORIZED_DEFAULT``. Used as the offline baseline and the safe default.
    """

    ranker_version = "heuristic_precedence_v1"

    def score(self, path: CandidatePath, context: dict[str, Any]) -> float:
        return _PRECEDENCE_PRIOR[path.path_type]


class LearnedPathRanker:
    """Wrap a learned score function, guarded by precedence priors.

    ``score_fn`` maps (path, context) -> a real-valued adjustment that is added
    to the canonical precedence prior. Because selection is restricted to legal
    candidates and ties/priors are deterministic, a poorly-trained model can
    never select an illegal path — at worst it reorders legal ones.
    """

    ranker_version = "learned_v1"

    def __init__(self, score_fn: SCORE_FN, *, prior_weight: float = 1.0) -> None:
        self._score_fn = score_fn
        self._prior_weight = prior_weight

    def score(self, path: CandidatePath, context: dict[str, Any]) -> float:
        return self._prior_weight * _PRECEDENCE_PRIOR[path.path_type] + self._score_fn(
            path, context
        )


def rank_belief(
    belief_id: str,
    candidates: list[CandidatePath],
    ranker: HeuristicPathRanker | LearnedPathRanker,
    *,
    context: dict[str, Any] | None = None,
) -> RankResult:
    """Score and select one legal candidate path for a belief."""
    if not candidates:
        raise ValueError(f"no candidate paths for belief '{belief_id}'")
    context = context or {}
    scores = {p.path_type: ranker.score(p, context) for p in candidates}
    # Deterministic tie-break: higher score, then higher precedence prior.
    ordered = sorted(
        candidates,
        key=lambda p: (scores[p.path_type], _PRECEDENCE_PRIOR[p.path_type]),
        reverse=True,
    )
    selected = ordered[0]
    rejected = tuple(ordered[1:])
    return RankResult(
        target_belief_id=belief_id,
        selected=selected,
        rejected=rejected,
        scores=scores,
        rationale=f"selected {selected.path_type} (score={scores[selected.path_type]:.3f})",
    )


def rank_runtime(
    result: RuntimeResult,
    ranker: HeuristicPathRanker | LearnedPathRanker | None = None,
) -> dict[str, RankResult]:
    """Rank legal paths for every belief in a runtime result."""
    ranker = ranker or HeuristicPathRanker()
    candidates = legal_paths_from_runtime(result)
    return {
        belief_id: rank_belief(belief_id, paths, ranker)
        for belief_id, paths in candidates.items()
    }


def replay_status_from_ranking(ranking: dict[str, RankResult]) -> dict[str, str]:
    """Map a ranking back to DPA final statuses (audit replay)."""
    return {bid: r.selected.implied_status for bid, r in ranking.items()}
