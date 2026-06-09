"""Tests for external memory baseline prompt builders."""

from __future__ import annotations

import json

from benchmark.api import load_scenarios
from benchmark.model_runner import build_prompt
from scripts.memory.context_builders import (
    BASELINE_IDS,
    PAPER_MAIN_BASELINE_IDS,
    build_baseline_prompt,
    build_baseline_view,
)


def _first_scenario() -> dict:
    return load_scenarios("hf_release/mempatch/test/scenarios.jsonl")[0]


def test_all_baseline_ids_build_prompt() -> None:
    scenario = _first_scenario()
    for baseline in BASELINE_IDS:
        prompt = build_baseline_prompt(scenario, baseline)
        payload = json.loads(prompt)
        assert "required_output_schema" in payload
        assert payload.get("scenario_id") == scenario["scenario_id"]


def test_paper_main_baselines_subset() -> None:
    assert set(PAPER_MAIN_BASELINE_IDS).issubset(set(BASELINE_IDS))


def test_vanilla_rag_filters_events() -> None:
    scenario = _first_scenario()
    full = len(scenario["public_input"]["event_trace"])
    view = build_baseline_view(scenario, "vanilla_rag", rag_top_k=3)
    assert len(view["public_input"]["event_trace"]) <= min(3, full)


def test_structured_direct_matches_build_prompt() -> None:
    scenario = _first_scenario()
    from benchmark.public_view import public_scenario_view

    direct = build_prompt(public_scenario_view(scenario))
    baseline = build_baseline_prompt(scenario, "structured_direct")
    assert json.loads(direct) == json.loads(baseline)
