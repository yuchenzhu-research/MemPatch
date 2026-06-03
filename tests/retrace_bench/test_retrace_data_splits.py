"""Regression tests for the ReTrace-Bench v1.0 paper-facing split package.

These guard the v1.0 release structure so it cannot silently regress: the four
canonical splits (``main`` / ``hard`` / ``realistic`` / ``calibration``), their
counts, cross-split disjointness, schema, the hard-split structural rules, the
de-actionalization (decision-word leakage) guarantee on authoritative records,
and the realistic split's pending annotation status.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.retrace_bench.general_taxonomy import (
    DECISIONS,
    DOMAINS,
    FAILURE_MODES,
    MEMORY_STATUSES,
    TASK_TYPES,
)
from benchmark.retrace_bench.generation.release_manifest import (
    BENCHMARK_VERSION,
    scenario_leaks_decision_word,
)

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "data" / "retrace_bench"

# (split dir, public split name, expected count)
SPLITS = (
    ("main_3000_en", "main", 3000),
    ("hard_300_en", "hard", 300),
    ("realistic_100_en", "realistic", 100),
    ("calibration_80_en", "calibration", 80),
)

REQUIRED_TASK_TYPES = set(TASK_TYPES)
REQUIRED_FIELDS = {
    "scenario_id", "split", "domain", "primary_failure_mode",
    "secondary_failure_modes", "difficulty", "workflow_context",
    "public_input", "tasks", "hidden_gold", "metadata",
}


def _load(dir_name: str) -> list[dict]:
    path = BENCH / dir_name / "scenarios.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    return {public: _load(dir_name) for dir_name, public, _ in SPLITS}


def test_split_files_exist():
    for dir_name, _, _ in SPLITS:
        base = BENCH / dir_name
        assert (base / "scenarios.jsonl").exists(), f"missing scenarios for {dir_name}"
        assert (base / "manifest.json").exists(), f"missing manifest for {dir_name}"
        assert (base / "README.md").exists(), f"missing README for {dir_name}"


def test_only_canonical_splits_present():
    present = {p.name for p in BENCH.iterdir() if p.is_dir()}
    assert present == {dir_name for dir_name, _, _ in SPLITS}, present


def test_split_counts(splits):
    for dir_name, public, expected in SPLITS:
        assert len(splits[public]) == expected, f"{public} size {len(splits[public])} != {expected}"


def test_public_split_label_matches(splits):
    for _, public, _ in SPLITS:
        assert all(r.get("split") == public for r in splits[public])


def test_manifests_use_public_split_names_and_version():
    forbidden = {"train", "dev", "validation", "test"}
    for dir_name, public, expected in SPLITS:
        manifest = json.loads((BENCH / dir_name / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["split"] == public
        assert manifest["split"] not in forbidden
        assert manifest["version"] == BENCHMARK_VERSION
        assert manifest["scenario_count"] == expected


def test_scenario_ids_unique_across_all_splits(splits):
    seen: dict[str, str] = {}
    for public, rows in splits.items():
        for r in rows:
            sid = r["scenario_id"]
            assert sid not in seen, f"scenario_id {sid} in both {seen.get(sid)} and {public}"
            seen[sid] = public


def test_memory_and_event_ids_disjoint_across_splits(splits):
    mem_owner: dict[str, str] = {}
    ev_owner: dict[str, str] = {}
    for public, rows in splits.items():
        for r in rows:
            for m in r["public_input"]["initial_memory"]:
                mid = m["memory_id"]
                assert mid not in mem_owner or mem_owner[mid] == public, f"memory id {mid} shared"
                mem_owner[mid] = public
            for e in r["public_input"]["event_trace"]:
                eid = e["event_id"]
                assert eid not in ev_owner or ev_owner[eid] == public, f"event id {eid} shared"
                ev_owner[eid] = public


def test_all_rows_parse_and_have_required_fields(splits):
    for public, rows in splits.items():
        for r in rows:
            assert REQUIRED_FIELDS <= set(r), f"{public} {r['scenario_id']} missing {REQUIRED_FIELDS - set(r)}"
            assert r["domain"] in DOMAINS
            assert r["primary_failure_mode"] in FAILURE_MODES
            task_types = {t["task_type"] for t in r["tasks"]}
            assert task_types == REQUIRED_TASK_TYPES, f"{public} {r['scenario_id']} task types {task_types}"


def test_no_training_targets_in_benchmark_rows(splits):
    for public, rows in splits.items():
        for r in rows:
            assert "training_targets" not in r, f"{public} {r['scenario_id']} leaks SFT training_targets"


def test_evidence_ids_exist_in_event_trace(splits):
    for public, rows in splits.items():
        for r in rows:
            event_ids = {e["event_id"] for e in r["public_input"]["event_trace"]}
            for eid in r["hidden_gold"].get("expected_evidence_event_ids", []):
                assert eid in event_ids, f"{public} {r['scenario_id']} evidence {eid} missing"


def test_expected_decisions_and_diagnoses_are_valid_enums(splits):
    for public, rows in splits.items():
        for r in rows:
            decision = r["hidden_gold"].get("expected_decision", "")
            diagnosis = r["hidden_gold"].get("expected_failure_diagnosis", "")
            if decision:  # realistic split is pending annotation (empty gold)
                assert decision in DECISIONS, f"{public} {r['scenario_id']} decision {decision}"
            if diagnosis:
                assert diagnosis in FAILURE_MODES, f"{public} {r['scenario_id']} diagnosis {diagnosis}"
            for mid, status in r["hidden_gold"].get("expected_memory_state", {}).items():
                assert status in MEMORY_STATUSES, f"{public} {r['scenario_id']} status {status}"


def test_no_decision_word_leakage_in_authoritative_records(splits):
    for public, rows in splits.items():
        leaks = {r["scenario_id"]: hits for r in rows if (hits := scenario_leaks_decision_word(r))}
        assert not leaks, f"{public} has decision-word leakage: {dict(list(leaks.items())[:3])}"


# ---- hard-split structural rules ----------------------------------------

def test_hard_event_counts_within_20_100(splits):
    for r in splits["hard"]:
        n = len(r["public_input"]["event_trace"])
        assert 20 <= n <= 100, f"{r['scenario_id']} has {n} events"


def test_hard_has_at_least_five_memories(splits):
    for r in splits["hard"]:
        assert len(r["public_input"]["initial_memory"]) >= 5, r["scenario_id"]


def test_hard_has_at_least_two_evidence_events(splits):
    for r in splits["hard"]:
        assert len(r["hidden_gold"]["expected_evidence_event_ids"]) >= 2, r["scenario_id"]


def test_hard_satisfies_at_least_three_hard_criteria(splits):
    for r in splits["hard"]:
        assert r["metadata"]["hard_criteria_satisfied"] >= 3, r["scenario_id"]


def test_hard_length_mix():
    counts = [len(r["public_input"]["event_trace"]) for r in _load("hard_300_en")]
    short = sum(1 for c in counts if c <= 35)
    medium = sum(1 for c in counts if 36 <= c <= 60)
    long = sum(1 for c in counts if c >= 61)
    assert (short, medium, long) == (100, 120, 80), (short, medium, long)


# ---- realistic-split annotation status ----------------------------------

def test_realistic_annotation_pending_and_no_gold(splits):
    for r in splits["realistic"]:
        assert r["metadata"]["annotation_status"] == "pending", r["scenario_id"]
        assert r["metadata"]["source_type"] == "realistic_style_synthetic"
        gold = r["hidden_gold"]
        assert not gold.get("expected_decision")
        assert not gold.get("expected_memory_state")
        assert not gold.get("expected_evidence_event_ids")


def test_realistic_category_mix():
    rows = _load("realistic_100_en")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["domain"]] = counts.get(r["domain"], 0) + 1
    assert counts == {
        "software_engineering_agent": 40,
        "customer_support_crm": 20,
        "research_knowledge_work": 15,
        "calendar_task_workflow": 15,
        "enterprise_multi_tool_workflow": 10,
    }, counts


def test_realistic_annotations_template_one_row_per_scenario():
    rows = _load("realistic_100_en")
    template_path = BENCH / "realistic_100_en" / "annotations_template.jsonl"
    template = [json.loads(line) for line in template_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(template) == len(rows)
    template_ids = {t["scenario_id"] for t in template}
    assert template_ids == {r["scenario_id"] for r in rows}
    for t in template:
        assert t["expected_decision"] == "" and t["evidence_event_ids"] == []
