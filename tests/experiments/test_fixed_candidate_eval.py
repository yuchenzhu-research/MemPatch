from __future__ import annotations

import json
import os
import pytest
from experiments.multiagent.episodes_fc_dev import get_fc_dev_episodes
from experiments.multiagent.methods import (
    NaiveLastWriteWinsFixedCandidateMethod,
    AppendOnlyLexicalTopKMethod,
    DirectJudgeReplayMethod,
    ReTraceStageAReplayMethod,
)
from experiments.multiagent.metrics import (
    compute_fixed_candidate_metrics,
    aggregate_fixed_candidate_metrics,
)
from experiments.multiagent.run_fc_comparison import run_fc_comparison


class TestFixedCandidateEpisodesSchema:
    def test_episode_count_is_14(self):
        episodes = get_fc_dev_episodes()
        assert len(episodes) == 14

    def test_7_failure_types_covered(self):
        episodes = get_fc_dev_episodes()
        failure_types = {ep.failure_type for ep in episodes}
        expected = {
            "cross_agent_conflict",
            "direct_supersession",
            "stale_propagation",
            "scope_expansion",
            "temporary_blocker_recovery",
            "duplicate_evidence",
            "ambiguous_update",
        }
        assert expected == failure_types

    def test_2_domains_covered(self):
        episodes = get_fc_dev_episodes()
        domains = {ep.domain for ep in episodes}
        assert domains == {"software_engineering", "research_workflow"}

    def test_each_failure_type_has_2_domains(self):
        episodes = get_fc_dev_episodes()
        coverage = {}
        for ep in episodes:
            coverage.setdefault(ep.failure_type, set()).add(ep.domain)
        for ft, domains in coverage.items():
            assert len(domains) == 2, f"failure_type {ft} missing domain coverage: {domains}"

    def test_serialization(self):
        episodes = get_fc_dev_episodes()
        for ep in episodes:
            data = ep.to_dict()
            assert isinstance(data, dict)
            assert data["episode_id"] == ep.episode_id
            assert data["protocol_mode"] == "oracle_edge_replay"
            assert data["proposal_source"] == "hand_authored_development"
            assert data["split"] == "development_only"

    def test_all_have_replay_decisions(self):
        episodes = get_fc_dev_episodes()
        for ep in episodes:
            assert len(ep.replay_decisions) > 0, f"{ep.episode_id} has no replay_decisions"

    def test_gold_snapshot_not_empty(self):
        episodes = get_fc_dev_episodes()
        for ep in episodes:
            assert len(ep.gold_snapshot.belief_statuses) > 0


class TestFixedCandidateMethods:
    def test_naive_lww_fc_authorizes_all(self):
        episodes = get_fc_dev_episodes()
        method = NaiveLastWriteWinsFixedCandidateMethod()
        for ep in episodes:
            result = method.run_fixed_episode(ep)
            assert result.method_name == "Naive_LWW_FC"
            assert result.protocol_mode == ep.protocol_mode
            assert result.proposal_source == ep.proposal_source
            # Should have at least one decision per episode
            assert len(result.decisions) > 0

    def test_append_only_topk_respects_k(self):
        episodes = get_fc_dev_episodes()
        method = AppendOnlyLexicalTopKMethod(k=2)
        for ep in episodes:
            result = method.run_fixed_episode(ep)
            authorized = [bid for bid, s in result.final_belief_statuses.items() if s == "AUTHORIZED"]
            # With k=2, at most 2 beliefs should be AUTHORIZED
            assert len(authorized) <= 2

    def test_direct_judge_replay_follows_decisions(self):
        episodes = get_fc_dev_episodes()
        method = DirectJudgeReplayMethod()
        for ep in episodes:
            result = method.run_fixed_episode(ep)
            assert result.method_name == "DirectJudge_Replay"
            # Decisions should match the episode's replay_decisions
            assert len(result.decisions) == len(ep.replay_decisions)

    def test_retrace_stage_a_replay(self):
        episodes = get_fc_dev_episodes()
        method = ReTraceStageAReplayMethod()
        for ep in episodes:
            result = method.run_fixed_episode(ep)
            assert result.method_name == "ReTrace_StageA_Replay"
            assert len(result.decisions) > 0

    def test_replay_methods_match_gold(self):
        """Both replay methods should match gold on all episodes."""
        episodes = get_fc_dev_episodes()
        for method_cls in [DirectJudgeReplayMethod, ReTraceStageAReplayMethod]:
            method = method_cls()
            for ep in episodes:
                result = method.run_fixed_episode(ep)
                for bid, expected in ep.gold_snapshot.belief_statuses.items():
                    actual = result.final_belief_statuses.get(bid)
                    assert actual == expected, (
                        f"{method.method_name} on {ep.episode_id}: "
                        f"belief {bid} expected {expected}, got {actual}"
                    )


class TestFixedCandidateMetrics:
    def test_compute_metrics_per_episode(self):
        episodes = get_fc_dev_episodes()
        method = DirectJudgeReplayMethod()
        for ep in episodes:
            result = method.run_fixed_episode(ep)
            metrics = compute_fixed_candidate_metrics(ep, result)
            assert "authorization_accuracy" in metrics
            assert "stale_propagation_error_rate" in metrics
            assert "decision_count" in metrics
            # DirectJudge replay should have perfect accuracy on gold-aligned episodes
            assert metrics["authorization_accuracy"] == 1.0, (
                f"{ep.episode_id}: auth accuracy should be 1.0, got {metrics['authorization_accuracy']}"
            )

    def test_aggregate_metrics(self):
        episodes = get_fc_dev_episodes()
        results = []
        for method_cls in [NaiveLastWriteWinsFixedCandidateMethod, DirectJudgeReplayMethod]:
            method = method_cls()
            for ep in episodes:
                result = method.run_fixed_episode(ep)
                results.append((ep, result))

        aggregated = aggregate_fixed_candidate_metrics(results)
        assert len(aggregated) > 0
        # DirectJudge replay overall accuracy should be 1.0
        assert "DirectJudge_Replay__overall__all" in aggregated
        assert aggregated["DirectJudge_Replay__overall__all"]["authorization_accuracy"] == 1.0


class TestFixedCandidateRunner:
    def test_run_fc_comparison(self):
        res = run_fc_comparison("offline_replay")
        assert res["results_count"] > 0
        assert os.path.exists(res["jsonl_path"])
        assert os.path.exists(res["summary_path"])
        assert os.path.exists(res["manifest_path"])

        # Verify JSONL schema
        with open(res["jsonl_path"], "r", encoding="utf-8") as f:
            first_row = json.loads(f.readline())
            assert "protocol_mode" in first_row
            assert "proposal_source" in first_row
            assert first_row["protocol_mode"] == "oracle_edge_replay"
            assert first_row["proposal_source"] == "hand_authored_development"

        # Verify manifest
        with open(res["manifest_path"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
            assert manifest["split"] == "development_only"
            assert len(manifest["methods"]) == 4

    def test_result_count_is_correct(self):
        """14 episodes × 4 methods × 8 metrics = 448 rows."""
        res = run_fc_comparison("offline_replay")
        assert res["results_count"] == 14 * 4 * 8
