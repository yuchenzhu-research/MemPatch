from __future__ import annotations

import pytest

from mempatch.benchmark.method_names import FINAL_METHODS, METHOD_ALIASES, normalize_method_name
from scripts.server.methods import build_method_view


def _view() -> dict:
    return {
        "scenario_id": "tiny",
        "workflow_context": "Choose the correct current release note.",
        "public_input": {
            "initial_memories": [{"memory_id": "m1", "content": "Old release note"}],
            "events": [
                {"event_id": "e1", "timestamp_order": 1, "content": "The old release note was superseded."},
                {"event_id": "e2", "timestamp_order": 2, "content": "The new release note is current."},
                {"event_id": "e3", "timestamp_order": 3, "content": "Unrelated support note."},
            ],
        },
        "tasks": {"black_box_task": {"query": "Which release note is current?"}},
    }


def test_every_final_method_is_recognized() -> None:
    for method in FINAL_METHODS:
        assert normalize_method_name(method) == method


def test_every_legacy_alias_maps_to_final_name() -> None:
    for alias, final in METHOD_ALIASES.items():
        assert normalize_method_name(alias) == final


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_method_name("vanilla")


def test_dense_rag_json_runs_on_tiny_fixture() -> None:
    view = build_method_view("dense_rag_json", _view(), retrieval_k=2)
    events = view["public_input"]["events"]
    assert len(events) == 2
    assert view["retrieval_metadata"]["method"] == "dense_rag_json"
    assert view["retrieval_metadata"]["retrieved_event_count"] == 2


def test_legacy_alias_builds_final_bm25_view() -> None:
    view = build_method_view("lexical_rag", _view(), retrieval_k=2)
    assert view["retrieval_metadata"]["method"] == "bm25_rag_json"
