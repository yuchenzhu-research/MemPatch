"""Learned defeat-path ranker safety tests (future/optional extension)."""
from __future__ import annotations

from retrace_learn.data.build_synthetic_raw_dialogue import build_synthetic_episodes, get_smoke_episode
from retrace_learn.runtime.dpa_runtime import run_actions
from retrace_learn.runtime.path_ranker import (
    PATH_TYPE_TO_STATUS,
    HeuristicPathRanker,
    LearnedPathRanker,
    legal_paths_from_runtime,
    rank_runtime,
    replay_status_from_ranking,
)


def _adversarial(path, context):
    return 100.0 if path.path_type == "AUTHORIZED_DEFAULT" else 0.0


def test_path_type_status_mapping_is_canonical():
    assert PATH_TYPE_TO_STATUS["DIRECT_SUPERSEDE"] == "SUPERSEDED"
    assert PATH_TYPE_TO_STATUS["PREREQUISITE_BLOCK"] == "BLOCKED"
    assert PATH_TYPE_TO_STATUS["UNRESOLVED_UNCERTAIN"] == "UNRESOLVED"
    assert PATH_TYPE_TO_STATUS["AUTHORIZED_DEFAULT"] == "AUTHORIZED"


def test_heuristic_ranker_replays_dpa_exactly():
    for ep in build_synthetic_episodes():
        view = ep.build_view()
        result = run_actions(view, list(ep.gold_actions))
        ranking = rank_runtime(result, HeuristicPathRanker())
        replay = replay_status_from_ranking(ranking)
        for bid, status in result.final_belief_statuses.items():
            assert replay[bid] == status


def test_authorized_default_is_always_legal():
    ep = get_smoke_episode()
    result = run_actions(ep.build_view(), list(ep.gold_actions))
    legal = legal_paths_from_runtime(result)
    for paths in legal.values():
        assert any(p.path_type == "AUTHORIZED_DEFAULT" for p in paths)


def test_adversarial_ranker_never_selects_illegal_path():
    ep = get_smoke_episode()
    result = run_actions(ep.build_view(), list(ep.gold_actions))
    legal = legal_paths_from_runtime(result)
    ranking = rank_runtime(result, LearnedPathRanker(_adversarial, prior_weight=0.0))
    for bid, rank in ranking.items():
        legal_types = {p.path_type for p in legal[bid]}
        assert rank.selected.path_type in legal_types
