from __future__ import annotations

import json
import os
import pytest
from retracemem.evaluation.multiagent.data.episodes_fc_dev import get_fc_dev_episodes
from experiments.multiagent.methods import (
    NaiveLastWriteWinsFixedCandidateMethod,
    AppendOnlyLexicalTopKMethod,
    DirectJudgeReplayMethod,
    ReTraceProposalReplayMethod,
)
from experiments.multiagent.metrics import (
    compute_fixed_candidate_metrics,
    aggregate_fixed_candidate_metrics,
)
from experiments.multiagent.legacy.run_fc_comparison import run_fc_comparison


class TestFixedCandidateEpisodesSchema:
    def test_episode_count_is_14(self):
        episodes = get_fc_dev_episodes()
        assert len(episodes) == 14

    def test_7_failure_types_covered(self):
        episodes = get_fc_dev_episodes()
        failure_types = {gold.failure_type for _, gold, _ in episodes}
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
        domains = {ep.domain for ep, _, _ in episodes}
        assert domains == {"software_engineering", "research_workflow"}

    def test_each_failure_type_has_2_domains(self):
        episodes = get_fc_dev_episodes()
        coverage = {}
        for ep, gold, _ in episodes:
            coverage.setdefault(gold.failure_type, set()).add(ep.domain)
        for ft, domains in coverage.items():
            assert len(domains) == 2, f"failure_type {ft} missing domain coverage: {domains}"

    def test_serialization(self):
        episodes = get_fc_dev_episodes()
        for ep, gold, artifact in episodes:
            ep_data = ep.to_dict()
            gold_data = gold.to_dict()
            art_data = artifact.to_dict()

            assert isinstance(ep_data, dict)
            assert isinstance(gold_data, dict)
            assert isinstance(art_data, dict)

            assert ep_data["episode_id"] == ep.episode_id
            assert ep_data["protocol_mode"] == "fixed_candidate_revision"
            assert ep_data["split"] == "development_only"
            assert gold_data["failure_type"] == gold.failure_type

    def test_all_have_replay_decisions(self):
        episodes = get_fc_dev_episodes()
        for ep, _, artifact in episodes:
            # Replay decisions are sidecars inside artifact
            assert len(artifact.direct_verdicts_by_submission) > 0, f"{ep.episode_id} has no replay direct verdicts"

    def test_gold_snapshot_not_empty(self):
        episodes = get_fc_dev_episodes()
        for _, gold, _ in episodes:
            assert len(gold.gold_snapshot.belief_statuses) > 0


class TestFixedCandidateMethods:
    def test_naive_lww_fc_authorizes_all(self):
        episodes = get_fc_dev_episodes()
        method = NaiveLastWriteWinsFixedCandidateMethod()
        for ep, _, _ in episodes:
            result = method.run_fixed_episode(ep)
            assert result.method_name == "Naive_LWW_FC"
            assert result.protocol_mode == ep.protocol_mode
            assert result.proposal_source == ep.proposal_source
            assert len(result.decisions) > 0

    def test_append_only_topk_respects_k(self):
        episodes = get_fc_dev_episodes()
        method = AppendOnlyLexicalTopKMethod(k=2)
        for ep, _, _ in episodes:
            result = method.run_fixed_episode(ep)
            authorized = [bid for bid, s in result.final_belief_statuses.items() if s == "AUTHORIZED"]
            assert len(authorized) <= 2

    def test_direct_judge_replay_follows_decisions(self):
        episodes = get_fc_dev_episodes()
        method = DirectJudgeReplayMethod()
        for ep, _, artifact in episodes:
            result = method.run_fixed_episode(ep, artifact=artifact)
            assert result.method_name == "DirectJudge_Replay"
            expected_len = sum(len(verdicts) for _, verdicts in artifact.direct_verdicts_by_submission)
            assert len(result.decisions) == expected_len

    def test_retrace_proposal_replay(self):
        episodes = get_fc_dev_episodes()
        method = ReTraceProposalReplayMethod()
        for ep, _, artifact in episodes:
            result = method.run_fixed_episode(ep, artifact=artifact)
            assert result.method_name == "ReTrace_StageA_Replay"
            assert len(result.decisions) > 0

    def test_replay_methods_match_gold(self):
        """Both replay methods should match gold on all episodes."""
        episodes = get_fc_dev_episodes()
        for method_cls in [DirectJudgeReplayMethod, ReTraceProposalReplayMethod]:
            method = method_cls()
            for ep, gold, artifact in episodes:
                result = method.run_fixed_episode(ep, artifact=artifact)
                for bid, expected in gold.gold_snapshot.belief_statuses.items():
                    actual = result.final_belief_statuses.get(bid)
                    assert actual == expected, (
                        f"{method.method_name} on {ep.episode_id}: "
                        f"belief {bid} expected {expected}, got {actual}"
                    )


class TestFixedCandidateMetrics:
    def test_compute_metrics_per_episode(self):
        episodes = get_fc_dev_episodes()
        method = DirectJudgeReplayMethod()
        for ep, gold, artifact in episodes:
            result = method.run_fixed_episode(ep, artifact=artifact)
            metrics = compute_fixed_candidate_metrics(gold, ep.downstream_tasks, result)
            assert "authorization_accuracy" in metrics
            assert "stale_propagation_error_rate" in metrics
            assert "decision_count" in metrics
            assert metrics["authorization_accuracy"] == 1.0, (
                f"{ep.episode_id}: auth accuracy should be 1.0, got {metrics['authorization_accuracy']}"
            )

    def test_aggregate_metrics(self):
        episodes = get_fc_dev_episodes()
        results = []
        for method_cls in [NaiveLastWriteWinsFixedCandidateMethod, DirectJudgeReplayMethod]:
            method = method_cls()
            for ep, gold, artifact in episodes:
                if method.method_name == "DirectJudge_Replay":
                    result = method.run_fixed_episode(ep, artifact=artifact)
                else:
                    result = method.run_fixed_episode(ep)
                results.append((gold, ep, result))

        aggregated = aggregate_fixed_candidate_metrics(results)
        assert len(aggregated) > 0
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
            assert first_row["protocol_mode"] == "fixed_candidate_revision"
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

    def test_output_rows_contain_all_fields(self):
        res = run_fc_comparison("offline_replay")
        required_fields = {
            "run_id", "episode_id", "domain", "failure_type", "protocol_mode",
            "scientific_status", "split", "method_name", "backbone_model", "proposal_source",
            "candidate_source", "number_of_subagents", "number_of_submissions", "role_diversity",
            "conflict_density", "delay_depth", "recovery_present", "metric_name", "metric_value",
            "trace_available", "calls", "tokens", "latency_ms",
            "policy_variant", "checkpoint_id", "training_split", "training_step",
            "training_examples_seen", "reward_variant", "authorization_reward",
            "downstream_task_reward", "scope_expansion_penalty", "stale_penalty", "total_reward"
        }
        with open(res["jsonl_path"], "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for fld in required_fields:
                    assert fld in row, f"Missing field {fld} in output row"


class TestPlotDataValidator:
    def test_validator_accepts_valid_inputs(self):
        from experiments.multiagent.legacy.validate_plot_inputs import validate_plot_inputs
        validate_plot_inputs(
            results_path="outputs/fc_method_results.jsonl",
            details_path="outputs/fc_run_details.json",
            official=False
        )

    def test_validator_fails_on_official_with_development_data(self):
        from experiments.multiagent.legacy.validate_plot_inputs import validate_plot_inputs
        with pytest.raises(SystemExit) as excinfo:
            validate_plot_inputs(
                results_path="outputs/fc_method_results.jsonl",
                details_path="outputs/fc_run_details.json",
                official=True
            )
        assert excinfo.value.code == 1
