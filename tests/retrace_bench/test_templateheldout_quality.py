from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from benchmark.retrace_bench.general_taxonomy import DOMAINS, FAILURE_MODES
from scripts.analyze_retrace_template_signatures import read_jsonl, scenario_signature
from scripts.run_template_lookup_baseline import evaluate as evaluate_template_lookup


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN = REPO_ROOT / "data" / "retrace_supervision" / "train_3000_en" / "scenarios.jsonl"
TEST = REPO_ROOT / "data" / "retrace_bench" / "test_800_templateheldout_en" / "scenarios.jsonl"


def _rows() -> list[dict]:
    assert TEST.exists(), f"missing template-heldout split: {TEST}"
    return read_jsonl(TEST)


def test_templateheldout_exists_and_has_800_scenarios():
    rows = _rows()
    assert len(rows) == 800
    assert len({row["scenario_id"] for row in rows}) == 800


def test_templateheldout_has_no_training_targets():
    for row in _rows():
        assert "training_targets" not in row


def test_templateheldout_covers_domains_and_failure_modes():
    rows = _rows()
    assert {row["domain"] for row in rows} == set(DOMAINS)
    assert {row["primary_failure_mode"] for row in rows} == set(FAILURE_MODES)


def test_decision_distribution_breaks_failure_mode_shortcut():
    rows = _rows()
    decisions = Counter(row["hidden_gold"]["expected_decision"] for row in rows)
    assert decisions["use_current_memory"] <= len(rows) * 0.50
    for decision in ("escalate", "ask_clarification", "mark_unresolved", "refuse_due_to_policy"):
        assert decisions[decision] >= len(rows) * 0.08

    by_mode: dict[str, set[str]] = defaultdict(set)
    by_decision: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        mode = row["primary_failure_mode"]
        decision = row["hidden_gold"]["expected_decision"]
        by_mode[mode].add(decision)
        by_decision[decision].add(mode)

    for mode in FAILURE_MODES:
        assert len(by_mode[mode]) >= 2, mode
    for decision, modes in by_decision.items():
        assert len(modes) >= 3, decision


def test_template_signature_overlap_from_train_is_below_threshold():
    train_rows = read_jsonl(TRAIN)
    test_rows = _rows()
    train_sigs = {scenario_signature(row) for row in train_rows}
    test_sigs = {scenario_signature(row) for row in test_rows}
    overlap_rate = len(train_sigs & test_sigs) / len(test_sigs)
    assert overlap_rate <= 0.10


def test_template_lookup_diagnostic_collapses_on_templateheldout():
    result = evaluate_template_lookup(TRAIN, TEST)
    assert result["decision_accuracy"] <= 0.55


def test_policy_violation_expected_answers_do_not_repeat_sensitive_payloads():
    for row in _rows():
        if row["primary_failure_mode"] != "policy_violation":
            continue
        expected = row["hidden_gold"]["expected_answer"].lower()
        payloads = row["hidden_gold"]["rubric"].get("sensitive_payloads", [])
        assert payloads, row["scenario_id"]
        for payload in payloads:
            assert payload.lower() not in expected, row["scenario_id"]


def test_restore_release_scenarios_have_sane_statuses():
    for row in _rows():
        if row["primary_failure_mode"] != "failure_to_release_or_restore":
            continue
        state = row["hidden_gold"]["expected_memory_state"]
        statuses = set(state.values())
        decision = row["hidden_gold"]["expected_decision"]
        if decision == "use_current_memory":
            assert statuses & {"restored", "current"}, row["scenario_id"]
            assert "blocked" not in statuses, row["scenario_id"]
        elif decision == "mark_unresolved":
            assert "unresolved" in statuses, row["scenario_id"]
        else:
            assert statuses & {"blocked", "unresolved"}, row["scenario_id"]


def test_timestamps_are_not_fixed_24h_increments():
    rows = _rows()
    all_deltas = []
    fixed_24h_only = 0
    for row in rows:
        times = [
            datetime.strptime(event["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
            for event in row["public_input"]["event_trace"]
        ]
        deltas = [(right - left).total_seconds() for left, right in zip(times, times[1:])]
        all_deltas.extend(deltas)
        if deltas and all(delta == 24 * 60 * 60 for delta in deltas):
            fixed_24h_only += 1
    assert fixed_24h_only == 0
    assert any(delta < 24 * 60 * 60 for delta in all_deltas)
    assert any(delta > 24 * 60 * 60 for delta in all_deltas)
