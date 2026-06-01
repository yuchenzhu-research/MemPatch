"""Regression tests for the ReTrace-Bench scoring-leakage fixes.

These pin the behaviour described in the P0 scorer repairs:
* key-fact accuracy can no longer be won by substring stuffing;
* stale reuse is detected on paraphrases, not only exact strings;
* enum decisions require strict equality unless explicit aliases exist;
* scope/policy anchors raise per-instance diagnostic hits.
"""

from __future__ import annotations

from typing import Any

from benchmark.retrace_bench.scorers_general import (
    _is_stale_reuse,
    decision_matches,
    key_fact_matches,
    score_prediction,
    token_f1_match,
)


EXPECTED = "C-1000 should follow the updated release blocker path for PROJ-A17."


def _scenario(gold: dict[str, Any], primary: str = "stale_memory_reuse") -> dict[str, Any]:
    return {
        "scenario_id": "rb-test",
        "primary_failure_mode": primary,
        "hidden_gold": gold,
    }


def test_key_fact_not_substring_cheat():
    """A long answer that merely embeds the expected string must not pass."""
    # Diverse filler (like retrieve_all concatenating many distinct events)
    # dwarfs the expected token set and drives F1 below threshold.
    filler = " ".join(f"unrelated{i} context{i}" for i in range(120))
    stuffed = f"{EXPECTED} {filler}"

    # Exact answer still passes; stuffed answer fails the F1 gate.
    assert key_fact_matches(EXPECTED, EXPECTED) is True
    assert key_fact_matches(stuffed, EXPECTED) is False
    assert token_f1_match(stuffed, EXPECTED) is False

    metrics = score_prediction(
        _scenario({"expected_answer": EXPECTED}),
        {"response": {"answer": stuffed, "decision": "use_current_memory"}},
    )
    assert metrics["answer_key_fact_accuracy"] == 0.0


def test_key_fact_rubric_must_not_include():
    rubric = {"must_not_include": ["earlier release blocker path"]}
    metrics = score_prediction(
        _scenario({"expected_answer": EXPECTED, "rubric": rubric}),
        {"response": {"answer": "C-1000 should follow the earlier release blocker path for PROJ-A17."}},
    )
    assert metrics["answer_key_fact_accuracy"] == 0.0


def test_stale_reuse_paraphrase():
    """Paraphrased reuse of a stale policy answer should be detected."""
    stale = ["Approve the refund under the previous 30 day refund window."]
    pred = "Approve because the previous 30 day refund window still applies."
    assert _is_stale_reuse(pred, stale) is True

    metrics = score_prediction(
        _scenario(
            {
                "expected_answer": "Reject: the refund window is now 14 days.",
                "stale_or_wrong_answers": stale,
            },
            primary="stale_memory_reuse",
        ),
        {"response": {"answer": pred, "decision": "use_current_memory"}},
    )
    assert metrics["stale_reuse_rate"] == 1.0


def test_stale_reuse_does_not_flag_correct_answer():
    """A correct answer lexically close to a stale one is not flagged."""
    stale = ["C-1000 should follow the earlier release blocker path for PROJ-A17."]
    metrics = score_prediction(
        _scenario({"expected_answer": EXPECTED, "stale_or_wrong_answers": stale}),
        {"response": {"answer": EXPECTED, "decision": "use_current_memory"}},
    )
    assert metrics["stale_reuse_rate"] == 0.0


def test_decision_matches_strict():
    assert decision_matches("reject_refund", "reject_refund") is True
    # Negation / substring must not match.
    assert decision_matches("do_not_reject_refund", "reject_refund") is False
    assert decision_matches("I think we should reject_refund now", "reject_refund") is False
    # Aliases honored only when explicitly provided.
    assert decision_matches("deny_refund", "reject_refund") is False
    assert decision_matches("deny_refund", "reject_refund", {"reject_refund": ["deny_refund"]}) is True


def test_scope_leakage_anchor_hit():
    gold = {
        "expected_answer": "Use only workspace-A memory for C-1000.",
        "scope_leakage_anchors": ["memory from workspace-B for another customer"],
    }
    leaked = "Reuse the memory from workspace-B for another customer to answer."
    metrics = score_prediction(
        _scenario(gold, primary="scope_leakage"),
        {"response": {"answer": leaked, "decision": "use_current_memory"}},
    )
    assert metrics["scope_leakage_anchor_hit_rate"] == 1.0


def test_no_duplicate_headline_metrics():
    metrics = score_prediction(
        _scenario({"expected_answer": EXPECTED}),
        {"response": {"answer": EXPECTED, "decision": "use_current_memory"}},
    )
    assert "answer_accuracy" not in metrics
    assert "decision_accuracy" not in metrics
    assert "black_box_decision_accuracy" in metrics
