"""Tests for the Stage A vs Stage B failure-analysis module + run wiring.

Covers: --max-cases slicing, manifest generation (env-var name only, no key
leakage), CSV/markdown generation from mocked Stage A/B outputs, the failure
classifier, and Stage B strict-vs-canonicalized separation.
"""
from __future__ import annotations

import json
import os

import pytest

from retracemem.evaluation.multiagent.cases import load_eval_cases
from retracemem.evaluation.multiagent.failure_analysis import (
    FAILURE_CATEGORIES,
    BeliefAnalysis,
    build_failure_rows,
    build_manifest,
    classify_failure,
    rows_to_csv,
    rows_to_markdown,
    summarize_categories,
)


# --------------------------------------------------------------------------- #
# --max-cases
# --------------------------------------------------------------------------- #
def test_max_cases_slices_dataset():
    assert len(load_eval_cases(max_cases=3)) == 3
    assert len(load_eval_cases(max_cases=8)) == 8
    # max_cases beyond the dev set size loads only what exists (dev70).
    assert len(load_eval_cases(max_cases=10_000)) == 70


# --------------------------------------------------------------------------- #
# Manifest generation + no key leakage
# --------------------------------------------------------------------------- #
def _manifest(**over):
    base = dict(
        git_commit="abc123",
        branch="eval/live-stageab-failure-analysis",
        provider="siliconflow",
        provider_mode="openai-chat",
        model="deepseek-ai/DeepSeek-V3",
        temperature=0.0,
        constrained=True,
        cache_enabled=True,
        prompt_version="hash123",
        data_split="dev_expansion",
        max_cases=8,
        timestamp="2026-05-31T00:00:00Z",
        api_key_env="SILICONFLOW_API_KEY",
        api_key_present=True,
        live_api_run=True,
    )
    base.update(over)
    return build_manifest(**base)


def test_manifest_has_all_required_fields():
    m = _manifest()
    for key in (
        "git_commit", "branch", "provider", "provider_mode", "model",
        "temperature", "constrained", "cache_enabled", "prompt_version",
        "data_split", "max_cases", "timestamp", "api_key_env",
        "api_key_present", "live_api_run",
    ):
        assert key in m


def test_manifest_records_env_var_name_only_never_value(monkeypatch):
    secret = "sk-super-secret-value-DO-NOT-LEAK-9999"
    monkeypatch.setenv("SILICONFLOW_API_KEY", secret)
    m = _manifest(api_key_present=bool(os.getenv("SILICONFLOW_API_KEY")))
    text = json.dumps(m)
    # The env var NAME is recorded; the VALUE never is.
    assert m["api_key_env"] == "SILICONFLOW_API_KEY"
    assert m["api_key_present"] is True
    assert secret not in text
    # And the rendered markdown report must not leak it either.
    rows = build_failure_rows([_belief()])
    assert secret not in rows_to_markdown(rows, manifest=m)


# --------------------------------------------------------------------------- #
# CSV / markdown from mocked Stage A/B outputs
# --------------------------------------------------------------------------- #
def _belief(**over) -> BeliefAnalysis:
    base = dict(
        case_id="case_000",
        episode_id="ep1",
        belief_id="b1",
        gold_status="AUTHORIZED",
        a_status="AUTHORIZED",
        b_raw_verdict="USABLE",
        b_canonical_verdict="USABLE",
        b_strict_verdict="USABLE",
        a_actions=(),
        a_gate_decisions=(),
        a_parse_error=False,
    )
    base.update(over)
    return BeliefAnalysis(**base)


def test_csv_has_header_and_required_columns():
    rows = build_failure_rows([_belief()])
    csv_text = rows_to_csv(rows)
    header = csv_text.splitlines()[0]
    for col in (
        "case_id", "episode_id", "stage_a_final_status", "stage_b_final_status",
        "gold_final_status", "a_correct", "b_correct", "a_typed_actions",
        "a_gate_decisions", "b_raw_verdict", "b_canonicalized_verdict",
        "failure_category", "suggested_fix",
    ):
        assert col in header


def test_all_categories_are_in_canonical_set():
    # A spread of synthetic beliefs; every emitted category must be canonical.
    beliefs = [
        _belief(),  # none
        _belief(gold_status="SUPERSEDED", a_status="AUTHORIZED", a_actions=()),  # NO_REVISION_overuse
        _belief(
            gold_status="SUPERSEDED", a_status="AUTHORIZED",
            a_actions=({"action_type": "REAFFIRMS", "target_belief_id": "b1"},),
        ),  # missed_SUPERSEDES
        _belief(gold_status="AUTHORIZED", a_status="SUPERSEDED",
                a_actions=({"action_type": "SUPERSEDES", "target_belief_id": "b1"},)),  # over_update
        _belief(gold_status="UNRESOLVED", a_status="AUTHORIZED"),  # uncertainty_collapse
        _belief(a_parse_error=True, a_status=None),  # parse_error
    ]
    rows = build_failure_rows(beliefs)
    for r in rows:
        assert r["failure_category"] in FAILURE_CATEGORIES


# --------------------------------------------------------------------------- #
# Classifier specifics
# --------------------------------------------------------------------------- #
def test_classify_correct_is_none():
    assert classify_failure(_belief()) == "none"


# In genuine Stage-A-miss cases Stage B judged correctly (and thus disagrees
# with A), so they are distinct from gold-label disputes where A and B agree.
def test_classify_missed_supersedes():
    b = _belief(
        gold_status="SUPERSEDED", a_status="AUTHORIZED",
        b_canonical_verdict="NOT_USABLE", b_strict_verdict="NOT_USABLE",
        b_raw_verdict="NOT_USABLE",
        a_actions=({"action_type": "REAFFIRMS", "target_belief_id": "b1"},),
    )
    assert classify_failure(b) == "missed_SUPERSEDES"


def test_classify_no_revision_overuse():
    b = _belief(
        gold_status="SUPERSEDED", a_status="AUTHORIZED", a_actions=(),
        b_canonical_verdict="NOT_USABLE", b_strict_verdict="NOT_USABLE",
        b_raw_verdict="NOT_USABLE",
    )
    assert classify_failure(b) == "NO_REVISION_overuse"


def test_classify_over_update():
    b = _belief(
        gold_status="AUTHORIZED", a_status="SUPERSEDED",
        b_canonical_verdict="USABLE", b_strict_verdict="USABLE", b_raw_verdict="USABLE",
        a_actions=({"action_type": "SUPERSEDES", "target_belief_id": "b1"},),
    )
    assert classify_failure(b) == "over_update"


def test_classify_uncertainty_collapse():
    b = _belief(
        gold_status="UNRESOLVED", a_status="AUTHORIZED",
        b_canonical_verdict="NOT_USABLE", b_strict_verdict="NOT_USABLE",
    )
    assert classify_failure(b) == "uncertainty_collapse"


def test_classify_parse_error():
    b = _belief(a_parse_error=True, a_status=None, gold_status="SUPERSEDED")
    assert classify_failure(b) == "parse_error"


def test_classify_invalid_target_from_gate_reject():
    b = _belief(
        gold_status="SUPERSEDED", a_status="AUTHORIZED",
        b_canonical_verdict="NOT_USABLE", b_strict_verdict="NOT_USABLE",
        a_gate_decisions=({"edge_type": "SUPERSEDES", "admitted": False,
                           "reason": "target id unknown"},),
    )
    assert classify_failure(b) == "invalid_target"


def test_classify_missing_new_evidence_from_gate_reject():
    b = _belief(
        gold_status="SUPERSEDED", a_status="AUTHORIZED",
        b_canonical_verdict="NOT_USABLE", b_strict_verdict="NOT_USABLE",
        a_gate_decisions=({"edge_type": "SUPERSEDES", "admitted": False,
                           "reason": "missing new evidence grounding"},),
    )
    assert classify_failure(b) == "missing_new_evidence"


def test_classify_possible_gold_issue_when_a_and_b_agree():
    # Both independent methods say USABLE; gold says NOT_USABLE -> suspect gold.
    b = _belief(
        gold_status="SUPERSEDED", a_status="AUTHORIZED",
        b_canonical_verdict="USABLE", b_strict_verdict="USABLE", b_raw_verdict="USABLE",
        a_actions=({"action_type": "REAFFIRMS", "target_belief_id": "b1"},),
    )
    assert classify_failure(b) == "possible_gold_issue"


# --------------------------------------------------------------------------- #
# Stage B strict vs canonicalized separation
# --------------------------------------------------------------------------- #
def test_stage_b_canonicalization_advantage_separated_from_strict():
    # Stage A wrong; Stage B strict ALSO wrong (UNCERTAIN), but canonicalization
    # rescued Stage B's canonical verdict to the correct USABLE.
    b = _belief(
        gold_status="AUTHORIZED",
        a_status="SUPERSEDED",
        a_actions=({"action_type": "SUPERSEDES", "target_belief_id": "b1"},),
        b_strict_verdict="UNCERTAIN",     # strict B is wrong
        b_canonical_verdict="USABLE",     # canonical B is right
        b_raw_verdict="USABLE",
    )
    rows = build_failure_rows([b])
    row = rows[0]
    assert row["b_correct"] is True                 # canonical B correct
    assert row["failure_category"] == "Stage_B_canonicalization_advantage"
    # strict and canonical are reported distinctly (not conflated).
    assert row["b_canonicalized_verdict"] == "USABLE"


def test_summarize_categories_excludes_none_and_sorts():
    rows = build_failure_rows([
        _belief(),  # none
        _belief(gold_status="SUPERSEDED", a_status="AUTHORIZED",
                b_canonical_verdict="NOT_USABLE", b_strict_verdict="NOT_USABLE"),  # NO_REVISION_overuse
        _belief(belief_id="b2", gold_status="SUPERSEDED", a_status="AUTHORIZED",
                b_canonical_verdict="NOT_USABLE", b_strict_verdict="NOT_USABLE"),  # NO_REVISION_overuse
    ])
    summary = summarize_categories(rows)
    assert ("none", pytest.approx(1)) not in summary
    assert summary[0][0] == "NO_REVISION_overuse"
    assert summary[0][1] == 2
