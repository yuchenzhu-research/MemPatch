"""Validation for the internal ReTrace train / dev / test split package."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from benchmark.retrace_bench.general_taxonomy import (
    DOMAINS,
    FAILURE_MODES,
    PUBLIC_FORBIDDEN_TERMS,
    TASK_TYPES,
)
from benchmark.retrace_bench.scorers_general import score_prediction
from scripts.check_retrace_split_leakage import check

REPO = Path(__file__).resolve().parents[2]

SPLITS = {
    "train": (REPO / "data/retrace_supervision/train_3000_en/scenarios.jsonl", 3000, True),
    "dev": (REPO / "data/retrace_supervision/dev_400_en/scenarios.jsonl", 400, True),
    "test": (REPO / "data/retrace_bench/test_800_en/scenarios.jsonl", 800, False),
}

REQUIRED_TASK_TYPES = {
    "black_box_task",
    "memory_state_task",
    "evidence_retrieval_task",
    "diagnostic_task",
}

CHECK_FORBIDDEN = ("retrace", "dpa", "authorization court", "benchmark", "gold label", "hidden truth")


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    return {name: _load(path) for name, (path, _, _) in SPLITS.items()}


def test_split_files_exist():
    for name, (path, _, _) in SPLITS.items():
        assert path.exists(), f"missing {name} split at {path}"
        assert (path.parent / "manifest.json").exists()
        assert (path.parent / "README.md").exists()


def test_split_sizes(splits):
    for name, (_, expected, _) in SPLITS.items():
        assert len(splits[name]) == expected, f"{name} size {len(splits[name])} != {expected}"


def test_all_domains_present(splits):
    for name, rows in splits.items():
        present = {r["domain"] for r in rows}
        assert present == set(DOMAINS), f"{name} domains {present}"


def test_all_failure_modes_present(splits):
    for name, rows in splits.items():
        present = {r["primary_failure_mode"] for r in rows}
        assert present == set(FAILURE_MODES), f"{name} modes {present}"


def test_required_schema_fields(splits):
    required = {
        "scenario_id", "domain", "primary_failure_mode", "secondary_failure_modes",
        "difficulty", "workflow_context", "public_input", "tasks", "hidden_gold", "metadata",
    }
    for name, rows in splits.items():
        for r in rows[:200]:
            assert required <= set(r), f"{name} {r['scenario_id']} missing {required - set(r)}"
            assert "event_trace" in r["public_input"] and "initial_memory" in r["public_input"]
            task_types = {t["task_type"] for t in r["tasks"]}
            assert REQUIRED_TASK_TYPES <= task_types
            assert task_types <= set(TASK_TYPES)


def test_training_targets_presence(splits):
    for name, (_, _, has_targets) in SPLITS.items():
        for r in splits[name]:
            if has_targets:
                tt = r.get("training_targets")
                assert tt is not None, f"{name} {r['scenario_id']} missing training_targets"
                assert tt["typed_revision_actions"], "empty typed_revision_actions"
                assert isinstance(tt["target_memory_state"], dict) and tt["target_memory_state"]
                assert tt["supporting_evidence_event_ids"]
            else:
                assert "training_targets" not in r, f"{name} test scenario leaks training_targets"


def test_no_scenario_id_overlap(splits):
    errors, _ = check(splits["train"], splits["dev"], splits["test"])
    assert not any(e.startswith("scenario_id") for e in errors), errors


def test_no_memory_id_overlap(splits):
    errors, _ = check(splits["train"], splits["dev"], splits["test"])
    assert not any(e.startswith("memory_id") for e in errors), errors


def test_no_event_id_overlap(splits):
    errors, _ = check(splits["train"], splits["dev"], splits["test"])
    assert not any(e.startswith("event_id") for e in errors), errors


def test_no_public_event_text_overlap(splits):
    errors, _ = check(splits["train"], splits["dev"], splits["test"])
    assert not any(e.startswith("public_event_text") for e in errors), errors


def test_no_expected_answer_overlap(splits):
    errors, _ = check(splits["train"], splits["dev"], splits["test"])
    assert not any(e.startswith("expected_answer") for e in errors), errors


def test_leakage_checker_clean(splits):
    errors, _ = check(splits["train"], splits["dev"], splits["test"])
    assert errors == [], errors


def test_hidden_gold_evidence_event_ids_exist(splits):
    for name, rows in splits.items():
        for r in rows:
            event_ids = {e["event_id"] for e in r["public_input"]["event_trace"]}
            for eid in r["hidden_gold"]["expected_evidence_event_ids"]:
                assert eid in event_ids, f"{name} {r['scenario_id']} evidence {eid} missing"


def test_expected_memory_state_ids_exist(splits):
    for name, rows in splits.items():
        for r in rows:
            known = {m["memory_id"] for m in r["public_input"]["initial_memory"]}
            known |= set(r["hidden_gold"].get("rubric", {}).get("introduced_memories", {}))
            for mid in r["hidden_gold"]["expected_memory_state"]:
                assert mid in known, f"{name} {r['scenario_id']} state id {mid} missing"


def test_public_text_has_no_forbidden_terms(splits):
    for name, rows in splits.items():
        for r in rows:
            blob_parts = [r["workflow_context"]]
            blob_parts += [t["prompt"] for t in r["tasks"]]
            for m in r["public_input"]["initial_memory"]:
                blob_parts.append(m["text"])
            for e in r["public_input"]["event_trace"]:
                blob_parts.append(e["text"])
            blob = " ".join(blob_parts).lower()
            for term in CHECK_FORBIDDEN + tuple(PUBLIC_FORBIDDEN_TERMS):
                assert term not in blob, f"{name} {r['scenario_id']} contains forbidden term {term!r}"


def test_gold_equal_prediction_scores_perfectly(splits):
    rng = random.Random(20260601)
    for name, rows in splits.items():
        for r in rng.sample(rows, 25):
            g = r["hidden_gold"]
            pred = {
                "decision": g["expected_decision"],
                "answer": g["expected_answer"],
                "evidence_event_ids": list(g["expected_evidence_event_ids"]),
                "memory_state": dict(g["expected_memory_state"]),
                "failure_diagnosis": g["expected_failure_diagnosis"],
            }
            sc = score_prediction(r, pred)
            assert sc["black_box_decision_accuracy"] == 1.0
            assert sc["memory_state_accuracy"] == 1.0
            assert sc["evidence_f1"] == 1.0
            assert sc["failure_diagnosis_accuracy"] == 1.0
            assert sc["stale_reuse_rate"] == 0.0


def test_stale_paraphrase_detection(splits):
    rng = random.Random(7)
    for name, rows in splits.items():
        candidates = [r for r in rows if r["hidden_gold"].get("stale_or_wrong_answers")]
        for r in rng.sample(candidates, 15):
            g = r["hidden_gold"]
            pred = {
                "decision": g["expected_decision"],
                "answer": g["stale_or_wrong_answers"][0],
                "evidence_event_ids": list(g["expected_evidence_event_ids"]),
                "memory_state": dict(g["expected_memory_state"]),
                "failure_diagnosis": g["expected_failure_diagnosis"],
            }
            sc = score_prediction(r, pred)
            assert sc["stale_reuse_rate"] == 1.0, f"{name} {r['scenario_id']} stale not detected"


def test_test_split_coverage_minimums(splits):
    rows = splits["test"]
    n = len(rows)

    def rate(key: str) -> float:
        return sum(r["metadata"][key] for r in rows) / n

    assert sum(len(r["public_input"]["event_trace"]) >= 7 for r in rows) / n >= 0.70
    assert sum(len(r["public_input"]["initial_memory"]) >= 3 for r in rows) / n >= 0.60
    assert rate("has_distractor") >= 0.50
    assert rate("has_cross_scope_trap") >= 0.40
    assert rate("verified_contradicts_trusted_note") >= 0.40
    assert rate("requires_rejecting_false_premise") >= 0.30
    assert rate("requires_non_answer_action") >= 0.25
