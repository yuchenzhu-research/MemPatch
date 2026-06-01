"""Validation tests for the committed hard calibration split sample_80_hard_en.

These assert structural integrity, taxonomy coverage, public-text hygiene, and
that the hidden gold is internally consistent (a gold-equal prediction scores
perfectly and is never falsely flagged as stale reuse).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.retrace_bench.general_taxonomy import (
    DOMAINS,
    FAILURE_MODES,
    MEMORY_STATUSES,
    PUBLIC_FORBIDDEN_TERMS,
    TASK_TYPES,
)
from benchmark.retrace_bench.scorers_general import score_prediction
from scripts.generate_retrace_bench_hard import build_scenario
from scripts.validate_retrace_bench_dataset import validate_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "retrace_bench" / "sample_80_hard_en" / "scenarios.jsonl"


def _load() -> list[dict]:
    assert DATA_PATH.exists(), f"missing committed fixture: {DATA_PATH}"
    return [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def scenarios() -> list[dict]:
    return _load()


def _public_text(scenario: dict) -> str:
    pieces = [scenario.get("workflow_context", "")]
    public = scenario.get("public_input", {})
    for event in public.get("event_trace", []):
        pieces.append(event.get("text", ""))
        pieces.append(event.get("source", ""))
    for memory in public.get("initial_memory", []):
        pieces.append(memory.get("text", ""))
    for task in scenario.get("tasks", []):
        pieces.append(task.get("prompt", ""))
    return "\n".join(pieces).lower()


def _all_memory_ids(scenario: dict) -> set[str]:
    public = scenario.get("public_input", {})
    ids = {m["memory_id"] for m in public.get("initial_memory", [])}
    introduced = scenario["hidden_gold"].get("rubric", {}).get("introduced_memories", {})
    return ids | set(introduced.keys())


def test_committed_fixture_matches_generator():
    """The committed file is exactly what the deterministic generator emits."""
    regenerated = [build_scenario(i) for i in range(80)]
    assert _load() == regenerated


def test_count_is_80(scenarios):
    assert len(scenarios) == 80
    assert len({s["scenario_id"] for s in scenarios}) == 80


def test_all_domains_present(scenarios):
    assert {s["domain"] for s in scenarios} == set(DOMAINS)


def test_all_failure_modes_present(scenarios):
    assert {s["primary_failure_mode"] for s in scenarios} == set(FAILURE_MODES)


def test_every_scenario_has_four_task_types(scenarios):
    for s in scenarios:
        types = [t["task_type"] for t in s["tasks"]]
        assert len(types) >= 4
        assert set(types) == set(TASK_TYPES), s["scenario_id"]


def test_hidden_gold_event_ids_exist(scenarios):
    for s in scenarios:
        event_ids = {e["event_id"] for e in s["public_input"]["event_trace"]}
        for eid in s["hidden_gold"]["expected_evidence_event_ids"]:
            assert eid in event_ids, f"{s['scenario_id']}: evidence {eid} missing"
        introduced = s["hidden_gold"].get("rubric", {}).get("introduced_memories", {})
        for mem in introduced.values():
            assert mem["introduced_by_event_id"] in event_ids, s["scenario_id"]


def test_expected_memory_state_ids_exist(scenarios):
    for s in scenarios:
        valid_ids = _all_memory_ids(s)
        for mid, status in s["hidden_gold"]["expected_memory_state"].items():
            assert mid in valid_ids, f"{s['scenario_id']}: state references unknown memory {mid}"
            assert status in MEMORY_STATUSES, f"{s['scenario_id']}: bad status {status}"


def test_memory_source_event_ids_exist(scenarios):
    for s in scenarios:
        event_ids = {e["event_id"] for e in s["public_input"]["event_trace"]}
        for mem in s["public_input"]["initial_memory"]:
            for eid in mem.get("source_event_ids", []):
                assert eid in event_ids, f"{s['scenario_id']}: memory cites missing event {eid}"


def test_event_related_memory_ids_exist(scenarios):
    for s in scenarios:
        valid_ids = _all_memory_ids(s)
        for event in s["public_input"]["event_trace"]:
            for mid in event.get("related_memory_ids", []):
                assert mid in valid_ids, f"{s['scenario_id']}: event cites missing memory {mid}"


def test_public_text_has_no_forbidden_terms(scenarios):
    for s in scenarios:
        text = _public_text(s)
        for term in PUBLIC_FORBIDDEN_TERMS:
            assert term not in text, f"{s['scenario_id']}: forbidden term '{term}'"


def test_dataset_coverage_thresholds(scenarios):
    """Hard-split coverage requirements (stricter than the base validator)."""
    report = validate_dataset(scenarios)
    assert report["errors"] == [], report["errors"]
    rates = report["rates"]
    assert rates["events_ge_7"] >= 0.70
    assert rates["memories_ge_3"] >= 0.60
    assert rates["distractors"] >= 0.50
    assert rates["cross_scope"] >= 0.40
    assert rates["verified_over_trusted"] >= 0.40
    assert rates["false_premise"] >= 0.30
    assert rates["non_answer"] >= 0.25


def test_gold_prediction_scores_perfectly_and_not_stale(scenarios):
    """A gold-equal prediction must be perfect and never flagged stale."""
    for s in scenarios:
        g = s["hidden_gold"]
        prediction = {
            "response": {
                "answer": g["expected_answer"],
                "decision": g["expected_decision"],
                "memory_state": g["expected_memory_state"],
                "evidence_event_ids": g["expected_evidence_event_ids"],
                "failure_diagnosis": g["expected_failure_diagnosis"],
            }
        }
        m = score_prediction(s, prediction)
        sid = s["scenario_id"]
        assert m["answer_key_fact_accuracy"] == 1.0, sid
        assert m["answer_exact_match"] == 1.0, sid
        assert m["memory_state_accuracy"] == 1.0, sid
        assert m["evidence_f1"] == 1.0, sid
        assert m["black_box_decision_accuracy"] == 1.0, sid
        assert m["failure_diagnosis_accuracy"] == 1.0, sid
        assert m["stale_reuse_rate"] == 0.0, sid


def test_paraphrased_stale_answer_is_flagged(scenarios):
    """Reusing the stale answer (verbatim or paraphrased) is detected."""
    from benchmark.retrace_bench.scorers_general import _is_stale_reuse

    flagged = 0
    for s in scenarios:
        g = s["hidden_gold"]
        stale_list = g.get("stale_or_wrong_answers", [])
        if not stale_list:
            continue
        # Paraphrase the first stale answer: add a prefix and swap a function
        # word so it is not string-equal but keeps the distinctive content.
        stale = stale_list[0]
        paraphrase = "As before, " + stale.replace("should", "must")
        assert paraphrase.strip() != stale.strip()
        assert _is_stale_reuse(paraphrase, stale_list, expected_answer=g["expected_answer"]), s["scenario_id"]
        flagged += 1
    assert flagged >= 70
