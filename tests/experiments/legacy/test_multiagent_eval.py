from __future__ import annotations

import json
import os
import pytest
from experiments.multiagent.episodes_dev import get_dev_episodes
from experiments.multiagent.methods import NaiveLastWriteWinsMethod, ReTraceMethod
from experiments.multiagent.metrics import compute_episode_metrics, aggregate_metrics
from experiments.multiagent.legacy.run_method_comparison import run_comparison


def test_episodes_schema_and_serialization():
    episodes = get_dev_episodes()
    assert len(episodes) == 10

    # Ensure all 7 failure types are covered
    failure_types = {ep.failure_type for ep in episodes}
    expected_failures = {
        "direct_supersession",
        "stale_propagation",
        "scope_expansion",
        "cross_agent_conflict",
        "temporary_blocker_recovery",
        "duplicate_evidence",
        "ambiguous_update",
    }
    assert expected_failures.issubset(failure_types)

    # Validate serialization works and does not raise errors
    for ep in episodes:
        data = ep.to_dict()
        assert isinstance(data, dict)
        assert data["episode_id"] == ep.episode_id
        assert data["domain"] in ("software_engineering", "research_workflow")
        assert "gold_snapshot" in data


def test_methods_and_metrics_calculations():
    episodes = get_dev_episodes()
    naive_method = NaiveLastWriteWinsMethod()
    retrace_method = ReTraceMethod()

    results = []
    for ep in episodes:
        naive_res = naive_method.run_episode(ep)
        retrace_res = retrace_method.run_episode(ep)

        assert naive_res.method_name == "Naive_LWW"
        assert retrace_res.method_name == "ReTrace"

        naive_metrics = compute_episode_metrics(ep, naive_res)
        retrace_metrics = compute_episode_metrics(ep, retrace_res)

        assert "authorization_accuracy" in retrace_metrics
        assert "stale_propagation_error_rate" in retrace_metrics
        assert "scope_expansion_error_rate" in retrace_metrics

        results.append((ep, naive_res))
        results.append((ep, retrace_res))

    aggregated = aggregate_metrics(results)
    assert len(aggregated) > 0
    # Overall ReTrace authorization accuracy should be correct
    assert "ReTrace__overall__all" in aggregated
    assert "authorization_accuracy" in aggregated["ReTrace__overall__all"]


def test_run_comparison_output_artifacts():
    res = run_comparison("offline_replay")
    assert res["results_count"] > 0
    assert os.path.exists(res["jsonl_path"])
    assert os.path.exists(res["summary_path"])
    assert os.path.exists(res["manifest_path"])

    # Load and verify JSONL rows
    with open(res["jsonl_path"], "r", encoding="utf-8") as f:
        first_row = json.loads(f.readline())
        assert "run_id" in first_row
        assert "metric_name" in first_row
        assert "metric_value" in first_row
        assert "domain" in first_row

    # Load manifest and verify
    with open(res["manifest_path"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
        assert manifest["split"] == "development_only"
        assert "code_commit_sha" in manifest
