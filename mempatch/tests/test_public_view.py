from __future__ import annotations

from mempatch.benchmark.public_view import public_scenario_view


def test_public_scenario_view_strips_internal_task_fields() -> None:
    scenario = {
        "scenario_id": "case_leak_check",
        "difficulty": "challenge",
        "workflow_context": "Routine workflow",
        "public_input": {
            "initial_memory": [{"memory_id": "m1", "text": "value", "is_distractor": False}],
            "event_trace": [{"event_id": "e1", "text": "update"}],
        },
        "black_box_task": {
            "prompt": "What is the current value?",
            "hidden_gold": {"expected_decision": "use_current_memory"},
            "metadata": {"split": "train"},
        },
    }

    view = public_scenario_view(scenario)
    task = view["black_box_task"]

    assert "hidden_gold" not in task
    assert "metadata" not in task
    assert task["prompt"] == "What is the current value?"
    assert "difficulty" not in view
    assert "is_distractor" not in view["public_input"]["initial_memory"][0]


def test_public_scenario_view_keeps_public_memory_text_for_distractor_heuristics() -> None:
    scenario = {
        "scenario_id": "case_dist",
        "public_input": {
            "initial_memory": [
                {
                    "memory_id": "m-case-000003-distractor",
                    "text": "Distractor info: separate workspace config",
                    "is_distractor": True,
                }
            ],
            "event_trace": [{"event_id": "e1", "text": "note"}],
        },
        "black_box_task": {"prompt": "Question?"},
    }

    view = public_scenario_view(scenario)
    memory = view["public_input"]["initial_memory"][0]

    assert memory["memory_id"] == "m-case-000003-distractor"
    assert memory["text"].startswith("Distractor info:")
    assert "is_distractor" not in memory


def test_public_scenario_view_strips_hidden_taxonomy_values_from_tags() -> None:
    scenario = {
        "scenario_id": "case_pattern_tag",
        "public_input": {
            "initial_memories": [
                {
                    "memory_id": "m1",
                    "content": "A public memory.",
                    "tags": ["software_release", "temporal_supersession"],
                    "metadata": {"resolverTrace": {"rule": "private"}},
                }
            ],
            "events": [{"event_id": "e1", "content": "note"}],
        },
    }

    view = public_scenario_view(scenario)
    memory = view["public_input"]["initial_memories"][0]
    assert memory["tags"] == ["software_release"]
    assert "metadata" not in memory
