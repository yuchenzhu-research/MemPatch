"""Tests for model-facing public view sanitization."""

from __future__ import annotations

import json

from benchmark.retrace_bench.public_view import (
    INTERNAL_ONLY_FIELDS,
    public_scenario_view,
    sanitize_public_input,
)


def test_sanitize_public_input_removes_internal_memory_fields():
    public_input = {
        "initial_memory": [
            {
                "memory_id": "m1",
                "text": "note",
                "scope": "workspace-stable",
                "source_event_ids": ["e1"],
                "is_distractor": True,
            }
        ],
        "event_trace": [
            {
                "event_id": "e1",
                "timestamp_order": 1,
                "actor_role": "user",
                "trust_level": "trusted",
                "visibility_scope": "workspace-stable",
                "event_type": "comment",
                "text": "visible event",
                "related_memory_ids": ["m1"],
                "timestamp": "2027-01-01T09:00:00Z",
                "metadata": {"hidden": True},
            }
        ],
        "metadata": {"internal": True},
    }
    cleaned = sanitize_public_input(public_input)
    assert "metadata" not in cleaned
    assert "is_distractor" not in cleaned["initial_memory"][0]
    assert cleaned["initial_memory"][0] == {
        "memory_id": "m1",
        "text": "note",
        "scope": "workspace-stable",
        "source_event_ids": ["e1"],
    }
    assert "metadata" not in cleaned["event_trace"][0]


def test_public_scenario_view_strips_top_level_internal_fields():
    scenario = {
        "scenario_id": "rt-hard-000001",
        "domain": "software_engineering_agent",
        "difficulty": "L4",
        "workflow_context": "Audit release state.",
        "pattern": "merged_but_unreleased",
        "primary_failure_mode": "stale_memory_reuse",
        "source_pointers": [{"url_or_id": "blueprint-x-1"}],
        "hidden_gold": {"expected_decision": "use_current_memory"},
        "metadata": {"pattern_trap_type": "trap"},
        "public_input": {
            "initial_memory": [
                {
                    "memory_id": "m1",
                    "text": "state",
                    "scope": "workspace-stable",
                    "source_event_ids": [],
                    "is_distractor": False,
                }
            ],
            "event_trace": [],
        },
        "black_box_task": {"prompt": "What should the agent do?"},
    }
    view = public_scenario_view(scenario)
    blob = json.dumps(view)
    for field in INTERNAL_ONLY_FIELDS:
        assert field not in view
    assert "is_distractor" not in blob
    assert view["black_box_task"]["prompt"] == "What should the agent do?"
