"""Regression tests for the canonical ReTrace-Bench (internal "v1.1") split package.

These guard the canonical release structure under ``data/retrace_bench_v1_1/`` so
it cannot silently regress: the four public splits
(``main`` / ``hard`` / ``realistic`` / ``calibration``) plus the optional private
``private_hidden`` split, their counts, cross-split disjointness, the general
schema, the validator (schema + leakage + pattern-semantic) gate, the hard-split
distribution rules, and the realistic split's unreviewed synthetic-gold status.

The legacy v1.0 layout is intentionally not exercised here; it lives under
``data_legacy/retrace_bench_v1_0/`` and is recoverable from Git history.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from benchmark.retrace_bench.general_taxonomy import (
    DECISIONS,
    DOMAINS,
    FAILURE_MODES,
    MEMORY_STATUSES,
    PATTERNS,
)
from benchmark.retrace_bench.generation.pattern_spec import infer_pattern
from benchmark.retrace_bench.generation.release_manifest import BENCHMARK_VERSION
from scripts.validate_retrace_bench_dataset import validate_dataset

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "data" / "retrace_bench_v1_1"

# (split dir, public split name, expected count)
PUBLIC_SPLITS = (
    ("main_3000_en", "main", 3000),
    ("hard_500_en", "hard", 500),
    ("realistic_200_en", "realistic", 200),
    ("calibration_80_en", "calibration", 80),
)
PRIVATE_SPLITS = (("private_hidden_200_en", "private_hidden", 200),)
ALL_SPLITS = PUBLIC_SPLITS + PRIVATE_SPLITS

# General-schema required fields (v1.1). The legacy v1.0 ``tasks`` / ``split`` /
# ``secondary_failure_modes`` fields are deliberately *not* required here.
REQUIRED_FIELDS = {
    "scenario_id",
    "public_split_name",
    "domain",
    "primary_failure_mode",
    "workflow_context",
    "public_input",
    "hidden_gold",
    "metadata",
}
REQUIRED_TASK_KEYS = (
    "black_box_task",
    "memory_state_task",
    "evidence_retrieval_task",
    "diagnostic_task",
)

# Approximate hard-split decision targets (Part 3 spec).
HARD_DECISION_RANGES = {
    "use_current_memory": (0.45, 0.55),
    "mark_unresolved": (0.15, 0.25),
    "ask_clarification": (0.10, 0.15),
    "refuse_due_to_policy": (0.08, 0.12),
    "escalate": (0.05, 0.10),
}


def _load(dir_name: str) -> list[dict]:
    path = BENCH / dir_name / "scenarios.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    """Load every canonical split whose ``scenarios.jsonl`` is present.

    The public splits must always be present. ``private_hidden`` scenarios are
    intentionally not committed to GitHub, so they are loaded only when available
    (e.g. locally after generation) and otherwise quietly skipped.
    """
    loaded: dict[str, list[dict]] = {}
    for dir_name, public, _ in ALL_SPLITS:
        path = BENCH / dir_name / "scenarios.jsonl"
        if path.exists():
            loaded[public] = _load(dir_name)
        elif public != "private_hidden":
            pytest.skip(f"required public split not present in checkout: {dir_name}")
    return loaded


def test_split_files_exist():
    # Public splits are committed and must always be present. ``private_hidden``
    # is intentionally not committed to GitHub, so it is only checked when
    # present locally (e.g. right after generation).
    for dir_name, public, _ in ALL_SPLITS:
        base = BENCH / dir_name
        if public == "private_hidden" and not base.exists():
            continue
        assert (base / "scenarios.jsonl").exists(), f"missing scenarios for {dir_name}"
        assert (base / "manifest.json").exists(), f"missing manifest for {dir_name}"
        assert (base / "README.md").exists(), f"missing README for {dir_name}"


def test_only_canonical_splits_present():
    present = {p.name for p in BENCH.iterdir() if p.is_dir()}
    canonical = {dir_name for dir_name, _, _ in ALL_SPLITS}
    public_dirs = {dir_name for dir_name, public, _ in ALL_SPLITS if public != "private_hidden"}
    # No unexpected directories, and every public split is present. The private
    # split may or may not be present depending on the checkout.
    assert present <= canonical, f"unexpected split dirs: {present - canonical}"
    assert public_dirs <= present, f"missing public split dirs: {public_dirs - present}"


def test_split_counts(splits):
    for dir_name, public, expected in ALL_SPLITS:
        if public not in splits:
            continue
        assert len(splits[public]) == expected, f"{public} size {len(splits[public])} != {expected}"


def test_public_split_label_matches(splits):
    for _, public, _ in ALL_SPLITS:
        if public not in splits:
            continue
        assert all(r.get("public_split_name") == public for r in splits[public])


def test_manifests_use_public_split_names_and_version():
    forbidden = {"train", "dev", "validation", "test"}
    for dir_name, public, expected in ALL_SPLITS:
        manifest_path = BENCH / dir_name / "manifest.json"
        if public == "private_hidden" and not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["split"] == public
        assert manifest["split"] not in forbidden
        assert manifest["version"] == BENCHMARK_VERSION
        assert manifest["scenario_count"] == expected


def test_validator_passes_with_no_errors(splits):
    """The official validator must report zero *errors* for every canonical split.

    Realistic is unreviewed synthetic gold, so warnings there are acceptable.
    """
    for dir_name, public, _ in ALL_SPLITS:
        if public not in splits:
            continue
        data_path = BENCH / dir_name / "scenarios.jsonl"
        report = validate_dataset(splits[public], data_path=data_path)
        assert report["errors"] == [], f"{public}: {report['errors'][:5]}"


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
            for tkey in REQUIRED_TASK_KEYS:
                assert tkey in r, f"{public} {r['scenario_id']} missing task view {tkey}"


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
            if decision:
                assert decision in DECISIONS, f"{public} {r['scenario_id']} decision {decision}"
            if diagnosis:
                assert diagnosis in FAILURE_MODES, f"{public} {r['scenario_id']} diagnosis {diagnosis}"
            for mid, status in r["hidden_gold"].get("expected_memory_state", {}).items():
                assert status in MEMORY_STATUSES, f"{public} {r['scenario_id']} status {status}"


# ---- hard-split distribution rules (Part 3 spec) ------------------------

def test_hard_is_l3_l4_only(splits):
    for r in splits["hard"]:
        diff = r.get("difficulty") or r.get("difficulty_level")
        assert diff in ("L3", "L4"), f"{r['scenario_id']} difficulty {diff}"


def test_hard_covers_all_fifteen_patterns(splits):
    seen = {infer_pattern(r) for r in splits["hard"]}
    assert set(PATTERNS) <= seen, f"missing patterns: {set(PATTERNS) - seen}"


def test_hard_no_single_pattern_dominates(splits):
    rows = splits["hard"]
    counts = collections.Counter(infer_pattern(r) for r in rows)
    n = len(rows)
    for pattern, c in counts.items():
        assert c / n <= 0.25 + 1e-9, f"pattern {pattern} share {c / n:.3f} exceeds 25%"


def test_hard_decision_distribution_in_target_ranges(splits):
    rows = splits["hard"]
    n = len(rows)
    counts = collections.Counter(r["hidden_gold"]["expected_decision"] for r in rows)
    for decision, (lo, hi) in HARD_DECISION_RANGES.items():
        share = counts[decision] / n
        assert lo - 1e-9 <= share <= hi + 1e-9, f"{decision} share {share:.3f} outside [{lo}, {hi}]"


def test_hard_average_evidence_above_one(splits):
    rows = splits["hard"]
    ev = [len(r["hidden_gold"]["expected_evidence_event_ids"]) for r in rows]
    assert sum(ev) / len(ev) > 1.0, sum(ev) / len(ev)


# ---- realistic-split annotation status ----------------------------------

def test_realistic_is_unreviewed_synthetic_gold(splits):
    for r in splits["realistic"]:
        status = r.get("annotation_status") or r.get("metadata", {}).get("annotation_status")
        assert status == "synthetic_gold_unreviewed", f"{r['scenario_id']} annotation_status={status!r}"
        # Realistic must NEVER be auto-marked reviewed.
        assert status != "reviewed"
