"""Tests for the hardened template-held-out **v2** generator.

These guard the design fixes from the v1 model-output audit
(``docs/retrace_bench/templateheldout_v1_model_audit.md``): no decision-word
leakage in the verified record, localized diagnostic prompts, conditional
cross-scope distractors, varied evidence labels, and atomic-fact rubrics.
"""

from __future__ import annotations

from collections import Counter

import pytest

from benchmark.retrace_bench.general_taxonomy import DOMAINS, FAILURE_MODES
from scripts.generate_retrace_templateheldout_v2 import build_scenario
from scripts.validate_retrace_bench_dataset import validate_dataset

ACTION_WORD_PREFIXES = (
    "escalate", "refuse", "ask for clarification", "ask ", "mark ",
    "use ", "keep ", "restore", "delete", "do not create",
)
NON_ANSWER_VERB = {
    "escalate": "escalate",
    "ask_clarification": "ask",
    "mark_unresolved": "mark",
    "refuse_due_to_policy": "refuse",
}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [build_scenario(i) for i in range(800)]


def _verified_text(row: dict) -> str:
    eid = row["hidden_gold"]["expected_evidence_event_ids"][0]
    for event in row["public_input"]["event_trace"]:
        if event["event_id"] == eid:
            return event["text"]
    raise AssertionError(f"{row['scenario_id']}: evidence event not found")


def _verified_body(row: dict) -> str:
    text = _verified_text(row)
    return text.split(": ", 1)[1] if ": " in text else text


def test_v2_writes_800_unique_scenarios(rows):
    assert len(rows) == 800
    assert len({r["scenario_id"] for r in rows}) == 800


def test_v2_passes_dataset_validation(rows):
    report = validate_dataset(rows)
    assert report["errors"] == [], report["errors"]
    assert report["count"] == 800


def test_v2_covers_all_domains_and_failure_modes(rows):
    assert {r["domain"] for r in rows} == set(DOMAINS)
    assert {r["primary_failure_mode"] for r in rows} == set(FAILURE_MODES)


def test_v2_no_public_forbidden_terms(rows):
    # validate_dataset already enforces PUBLIC_FORBIDDEN_TERMS; assert no errors
    # mention 'forbidden' as a defensive double-check.
    report = validate_dataset(rows)
    assert not any("forbidden" in e for e in report["errors"])


def test_v2_verified_record_has_no_decision_word_leakage(rows):
    # No verified-record body may begin with a final action verb (audit §3.1).
    for r in rows:
        body = _verified_body(r).lower()
        assert not any(body.startswith(p) for p in ACTION_WORD_PREFIXES), (
            f"{r['scenario_id']}: verified body leaks action verb: {body[:60]!r}"
        )


def test_v2_non_answer_evidence_does_not_start_with_action_verb(rows):
    for r in rows:
        dec = r["hidden_gold"]["expected_decision"]
        if dec in NON_ANSWER_VERB:
            body = _verified_body(r).lower()
            assert not body.startswith(NON_ANSWER_VERB[dec]), (
                f"{r['scenario_id']}: {dec} evidence starts with its action verb"
            )


def test_v2_diagnostic_task_references_concrete_event(rows):
    for r in rows:
        prompt = next(t["prompt"] for t in r["tasks"] if t["task_type"] == "diagnostic_task")
        focus = r["hidden_gold"]["diagnostic_focus_event_id"]
        contrast = r["hidden_gold"]["diagnostic_contrast_event_id"]
        assert focus in prompt and contrast in prompt, (
            f"{r['scenario_id']}: diagnostic prompt missing concrete event ids"
        )
        # The contrast event must be the verified/evidence event.
        assert contrast == r["hidden_gold"]["expected_evidence_event_ids"][0]


def test_v2_cross_scope_distractors_are_not_universal(rows):
    total = len(rows)
    with_cross = sum(1 for r in rows if r["metadata"]["has_cross_scope_trap"])
    nonscope = [r for r in rows if r["primary_failure_mode"] != "scope_leakage"]
    ns_cross = sum(1 for r in nonscope if r["metadata"]["has_cross_scope_trap"])
    assert with_cross / total < 0.50
    assert (ns_cross / len(nonscope)) <= 0.30
    # scope_leakage must always carry the cross-scope cue.
    for r in rows:
        if r["primary_failure_mode"] == "scope_leakage":
            assert r["metadata"]["has_cross_scope_trap"]


def test_v2_evidence_prefixes_are_varied(rows):
    labels = Counter(_verified_text(r).split(":", 1)[0] for r in rows)
    assert len(labels) >= 5
    # No single label may dominate, and "Authoritative record" must not be the
    # universal greppable prefix.
    assert all(count < len(rows) * 0.40 for count in labels.values())
    assert sum(1 for r in rows if _verified_text(r).startswith("Authoritative record:")) == 0


def test_v2_rubric_must_include_is_atomic(rows):
    long_entries = 0
    total = 0
    for r in rows:
        for entry in r["hidden_gold"]["rubric"]["must_include"]:
            total += 1
            if len(entry.split()) >= 8:
                long_entries += 1
    assert total > 0
    # Overwhelmingly atomic: no whole-sentence must_include entries.
    assert long_entries == 0


def test_v2_renderer_metadata_and_non_destructive_naming(rows):
    for r in rows:
        assert r["metadata"]["renderer"] == "templateheldout_v2"
        assert r["metadata"]["split"] == "test_800_templateheldout_v2_en"
        assert r["scenario_id"].startswith("rt-templateheldout-v2-")
