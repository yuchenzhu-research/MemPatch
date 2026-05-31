"""Section E ablation: defeat-path ranker variants vs DPA replay.

Confirms the safety properties of the learned path ranker:

* ``dpa_replay_consistency``: fraction of beliefs whose ranker-selected path
  reproduces the DPA final status. The :class:`HeuristicPathRanker` must score
  1.0 (it mirrors canonical precedence).
* ``legal_selection_rate``: fraction of selections that are among the enumerated
  legal candidates. Must be 1.0 for *any* ranker — even an adversarial one — by
  construction.

A deliberately bad learned ranker (always prefers AUTHORIZED_DEFAULT) is included
to show it can only lower final-status correctness, never select an illegal path.
"""
from __future__ import annotations

import argparse
import json

from retrace_learn.schemas import CANDIDATE_PATH_TYPES
from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes
from retrace_learn.runtime.dpa_runtime import run_actions
from retrace_learn.runtime.path_ranker import (
    HeuristicPathRanker,
    LearnedPathRanker,
    legal_paths_from_runtime,
    rank_runtime,
    replay_status_from_ranking,
)
from retrace_learn.eval.metrics import mean


def _consistency(ranker) -> dict[str, float]:
    replay_hits: list[float] = []
    legal_hits: list[float] = []
    for ep in build_synthetic_episodes():
        view = ep.build_view()
        result = run_actions(view, list(ep.gold_actions))
        legal = legal_paths_from_runtime(result)
        ranking = rank_runtime(result, ranker)
        replay = replay_status_from_ranking(ranking)
        for bid, status in result.final_belief_statuses.items():
            replay_hits.append(1.0 if replay.get(bid) == status else 0.0)
            legal_types = {p.path_type for p in legal[bid]}
            legal_hits.append(1.0 if ranking[bid].selected.path_type in legal_types else 0.0)
    return {
        "dpa_replay_consistency": mean(replay_hits),
        "legal_selection_rate": mean(legal_hits),
    }


def _always_authorized(path, context) -> float:
    # Adversarial: push AUTHORIZED_DEFAULT to the top regardless of precedence.
    return 100.0 if path.path_type == "AUTHORIZED_DEFAULT" else 0.0


def evaluate() -> dict[str, dict[str, float]]:
    return {
        "heuristic": _consistency(HeuristicPathRanker()),
        "adversarial_learned": _consistency(
            LearnedPathRanker(_always_authorized, prior_weight=0.0)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    report = {"candidate_path_types": list(CANDIDATE_PATH_TYPES), **evaluate()}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
