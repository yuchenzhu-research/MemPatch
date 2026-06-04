"""Shared manifest + leakage-audit helpers for ReTrace-Bench v1.0 splits.

These are used by the four paper-facing split generators (``main``, ``hard``,
``realistic``, ``calibration``) and by the release regression tests so the
leakage definition is single-sourced.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from benchmark.retrace_bench.general_taxonomy import DECISIONS

BENCHMARK_VERSION = "1.0.0"

# Direct decision-action phrases strongly tied to one of the five canonical
# gold DECISIONS. A verified/authoritative record (trust_level in
# {"verified", "trusted"}) must never contain one of these, otherwise the gold
# decision is recoverable by string match rather than by reasoning over the
# described state (decision-word leakage).
#
# NOTE: lifecycle words (delete / restore / forget) are deliberately NOT in this
# list. They denote canonical memory *statuses* (deleted / restored), not one of
# the five decisions, and the de-actionalized design intentionally *describes*
# such lifecycle facts as state. The list targets only the decision verbs plus
# "do not store" (the exact phrase that leaked in the legacy train_3000_en pool).
DECISION_WORD_PHRASES = (
    "ask for clarification",
    "ask the requester to confirm",
    "ask the requester to clarify",
    "please confirm",
    "mark unresolved",
    "mark it unresolved",
    "mark as unresolved",
    "escalate",
    "refuse",
    "do not store",
    "use current memory",
    "use the current memory",
)


def authoritative_records(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Verified/trusted events — the records whose wording could leak gold."""
    events = scenario.get("public_input", {}).get("event_trace", [])
    return [e for e in events if e.get("trust_level") in {"verified", "trusted"}]


def scenario_leaks_decision_word(scenario: dict[str, Any]) -> list[str]:
    """Return decision-action phrases found in this scenario's authoritative records."""
    hits: list[str] = []
    for event in authoritative_records(scenario):
        text = event.get("text", "").lower()
        for phrase in DECISION_WORD_PHRASES:
            if phrase in text and phrase not in hits:
                hits.append(phrase)
    return hits


def leakage_audit(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    leaky = 0
    phrase_counts: Counter = Counter()
    for row in rows:
        hits = scenario_leaks_decision_word(row)
        if hits:
            leaky += 1
            phrase_counts.update(hits)
    return {
        "method": "exact decision-action phrase match over verified/trusted records",
        "scanned_records": "authoritative (verified+trusted) event text only",
        "decision_word_phrases_checked": list(DECISION_WORD_PHRASES),
        "scenarios_with_decision_word_leak": leaky,
        "leak_phrase_counts": dict(sorted(phrase_counts.items())),
        "clean": leaky == 0,
    }


def _evidence_count(row: dict[str, Any]) -> int:
    gold = row.get("hidden_gold", {})
    return len(
        gold.get("expected_evidence_event_ids")
        or gold.get("minimal_evidence_event_ids")
        or []
    )


def _event_count(row: dict[str, Any]) -> int:
    return len(row.get("public_input", {}).get("event_trace", []))


def _memory_count(row: dict[str, Any]) -> int:
    return len(row.get("public_input", {}).get("initial_memory", []))


def build_manifest(
    rows: list[dict[str, Any]],
    *,
    split: str,
    source_type: str,
    annotation_status: str,
    role: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical v1.0 manifest for a split."""
    n = len(rows) or 1
    domains = Counter(r["domain"] for r in rows)
    modes = Counter(r["primary_failure_mode"] for r in rows)
    decisions = Counter(r["hidden_gold"]["expected_decision"] for r in rows)
    difficulty = Counter(r["difficulty"] for r in rows)
    event_counts = [_event_count(r) for r in rows]
    memory_counts = [_memory_count(r) for r in rows]
    evidence_counts = [_evidence_count(r) for r in rows]

    manifest = {
        "version": BENCHMARK_VERSION,
        "split": split,
        "scenario_count": len(rows),
        "schema_version": "retrace_bench_general_1",
        "role": role,
        "training_targets": False,
        "domains": dict(sorted(domains.items())),
        "failure_modes": dict(sorted(modes.items())),
        "expected_decisions": {d: decisions.get(d, 0) for d in DECISIONS},
        "difficulty": dict(sorted(difficulty.items())),
        "avg_event_count": round(sum(event_counts) / n, 3),
        "max_event_count": max(event_counts) if event_counts else 0,
        "avg_memory_count": round(sum(memory_counts) / n, 3),
        "max_memory_count": max(memory_counts) if memory_counts else 0,
        "avg_required_evidence_count": round(sum(evidence_counts) / n, 3),
        "source_type": source_type,
        "annotation_status": annotation_status,
        "leakage_audit_summary": leakage_audit(rows),
    }
    if extra:
        manifest.update(extra)
    return manifest
