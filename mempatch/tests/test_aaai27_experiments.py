from experiments.aaai27.analyze import _bootstrap_delta, _cluster_wtl
from experiments.aaai27.methods import build_method_view


def _view():
    return {
        "scenario_id": "case-1",
        "workflow_context": "resolve alpha release status",
        "public_input": {
            "initial_memory": [{"memory_id": "m1", "text": "alpha is pending"}],
            "event_trace": [
                {"event_id": "e1", "timestamp_order": 1, "text": "unrelated beta note"},
                {"event_id": "e2", "timestamp_order": 2, "text": "alpha release is verified"},
                {"event_id": "e3", "timestamp_order": 3, "text": "recent neutral audit"},
            ],
        },
    }


def test_context_methods_are_deterministic_and_do_not_mutate_input():
    source = _view()
    lexical = build_method_view("lexical_rag", source, retrieval_k=1)
    time_aware = build_method_view("time_aware_rag", source, retrieval_k=1)
    summary = build_method_view("summary_memory", source, retrieval_k=1)

    assert source["public_input"]["event_trace"][0]["event_id"] == "e1"
    assert lexical["public_input"]["event_trace"][0]["event_id"] == "e2"
    assert time_aware["public_input"]["event_trace"][0]["event_id"] in {"e2", "e3"}
    assert summary["public_input"]["event_trace"] == []
    assert "[e1]" in summary["memory_summary"]["text"]
    assert "[e3]" in summary["memory_summary"]["text"]


def test_cluster_bootstrap_and_wtl_are_paired_and_reproducible():
    scenarios = [
        {"scenario_id": "a", "metadata": {"decision_variant": "v1"}},
        {"scenario_id": "b", "metadata": {"decision_variant": "v1"}},
        {"scenario_id": "c", "metadata": {"decision_variant": "v2"}},
    ]
    left = {"a": 1.0, "b": 1.0, "c": 0.0}
    right = {"a": 0.0, "b": 0.5, "c": 0.0}

    first = _bootstrap_delta(scenarios, left, right, "decision_variant", 200, 42)
    second = _bootstrap_delta(scenarios, left, right, "decision_variant", 200, 42)

    assert first == second
    assert first["delta"] > 0
    assert _cluster_wtl(scenarios, left, right, "decision_variant") == {
        "wins": 1,
        "ties": 1,
        "losses": 0,
    }
