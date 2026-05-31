"""Stage A vs Stage B failure analysis (pure, I/O-free core).

Given parsed Stage A (typed-action -> RevisionGate -> DPA) and Stage B
(DirectJudge) outputs plus gold final statuses, produce a per-belief comparison
table that attributes *why* a belief's Stage A status diverges from gold, and
whether Stage B got it via canonicalization where Stage A did not.

This module is deterministic and does no network or disk I/O so it can be unit
tested directly from mocked Stage A/B records. The CLI wrapper
(`scripts/build_failure_analysis.py`) loads a finished run directory, calls
``build_failure_rows``, and writes ``failure_analysis.csv`` / ``.md`` /
``manifest.json``.

Stage A and Stage B are independent methods, never a pipeline: Stage B has no
typed actions / RevisionGate / DPA. The shared {USABLE, NOT_USABLE, UNCERTAIN}
comparability space (``STATUS_MAP_A_TO_COMPARABLE``) is only used to score both
methods against the same gold.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

from retracemem.evaluation.multiagent.metrics import STATUS_MAP_A_TO_COMPARABLE

# Canonical, closed set of failure categories. The analysis must label every
# belief with exactly one of these (``none`` for a correct Stage A belief).
FAILURE_CATEGORIES: tuple[str, ...] = (
    "none",
    "parse_error",
    "invalid_action",
    "invalid_target",
    "missing_new_evidence",
    "wrong_action_type",
    "missed_SUPERSEDES",
    "missed_BLOCKS",
    "missed_RELEASES",
    "NO_REVISION_overuse",
    "scope_leakage",
    "over_update",
    "under_update",
    "uncertainty_collapse",
    "Stage_B_canonicalization_advantage",
    "candidate_context_missing_anchor",
    "prompt_weakness",
    "possible_gold_issue",
    "unknown",
)

# Suggested-fix hints keyed by category (engineering-facing, not marketing).
_SUGGESTED_FIX: dict[str, str] = {
    "none": "",
    "parse_error": "Tighten output schema / enable --repair-on-parse-error; add few-shot JSON example.",
    "invalid_action": "Constrain action vocabulary in prompt; reject non-canonical action types pre-gate.",
    "invalid_target": "Surface valid target ids in the candidate view; remind proposer targets must exist.",
    "missing_new_evidence": "Require evidence_ids grounding; reject ungrounded actions before the gate.",
    "wrong_action_type": "Clarify action semantics (SUPERSEDES vs BLOCKS vs REAFFIRMS) in the prompt.",
    "missed_SUPERSEDES": "Add supersession cues / exemplars; emphasize replacement_belief_id requirement.",
    "missed_BLOCKS": "Add prerequisite-blocking cues; show BLOCKS targets a condition, not a belief.",
    "missed_RELEASES": "Add recovery/release cues; show RELEASES re-enables a previously blocked condition.",
    "NO_REVISION_overuse": "Penalize default NO_REVISION; require explicit justification when evidence conflicts.",
    "scope_leakage": "Restrict candidate view to in-scope beliefs; do not leak unrelated targets.",
    "over_update": "Proposer revised a belief gold marks usable; tighten revision threshold / add counter-exemplars.",
    "under_update": "Proposer left a stale belief authorized; strengthen conflict detection in the prompt.",
    "uncertainty_collapse": "Proposer collapsed UNCERTAIN to a hard status; add UNCERTAIN affordance/exemplars.",
    "Stage_B_canonicalization_advantage": "Metric artifact: Stage B matched only via belief-id canonicalization; report strict separately.",
    "candidate_context_missing_anchor": "Add the missing REQUIRES/condition anchor to the candidate view.",
    "prompt_weakness": "General prompt revision needed (Stage A wrong with no localizable structural cause).",
    "possible_gold_issue": "Inspect the gold label; A and B agree with each other but disagree with gold.",
    "unknown": "Manual inspection required.",
}


@dataclass(frozen=True)
class BeliefAnalysis:
    """Resolved per-belief inputs for one (episode, belief) comparison.

    All fields are method-visible run outputs plus gold; nothing here is fed
    back into a method (no leakage). ``a_actions`` / ``a_gate_decisions`` are the
    Stage A typed actions and RevisionGate decisions that *target this belief*.
    """

    case_id: str
    episode_id: str
    belief_id: str
    gold_status: str
    a_status: str | None
    b_raw_verdict: str | None
    b_canonical_verdict: str | None
    b_strict_verdict: str | None
    a_actions: tuple[dict[str, Any], ...] = ()
    a_gate_decisions: tuple[dict[str, Any], ...] = ()
    a_parse_error: bool = False


def _comparable(status: str | None) -> str:
    """Map a DPA status into the shared usability space (pass through if already there)."""
    if status is None:
        return "UNCERTAIN"
    if status in STATUS_MAP_A_TO_COMPARABLE:
        return STATUS_MAP_A_TO_COMPARABLE[status]
    return status  # already comparable (USABLE / NOT_USABLE / UNCERTAIN)


def classify_failure(b: BeliefAnalysis) -> str:
    """Return the single canonical failure category for one belief.

    Deterministic, structure-first attribution. Stage A correctness is judged in
    the shared comparability space against gold; the category localizes the
    *cause* using the typed actions and gate decisions that target the belief.
    """
    gold_c = _comparable(b.gold_status)
    a_c = _comparable(b.a_status)

    if b.a_parse_error:
        return "parse_error"

    if a_c == gold_c:
        return "none"

    # Stage A diverges from gold. Inspect gate decisions first (hard structural
    # rejections of a proposed action).
    rejected = [d for d in b.a_gate_decisions if not d.get("admitted", True)]
    for d in rejected:
        reason = (d.get("reason") or "").lower()
        if "evidence" in reason and ("missing" in reason or "ground" in reason):
            return "missing_new_evidence"
        if "target" in reason or "unknown" in reason or "exist" in reason:
            return "invalid_target"
        if "condition" in reason or "belief" in reason:
            return "invalid_action"
        return "invalid_action"

    # If the two independent methods agree with each other but both disagree with
    # gold, the gold label itself is the prime suspect (not a method failure).
    b_c = _comparable(b.b_canonical_verdict)
    if b.b_canonical_verdict is not None and a_c == b_c and a_c != gold_c:
        return "possible_gold_issue"

    real_actions = [a for a in b.a_actions if a.get("action_type") not in (None, "NO_REVISION")]
    action_types = {a.get("action_type") for a in real_actions}

    # Gold says the belief should NOT be usable (superseded/blocked) but Stage A
    # kept it usable -> a miss. Attribute to the specific missing action when the
    # gold status implies it.
    if gold_c == "NOT_USABLE" and a_c == "USABLE":
        if not real_actions:
            return "NO_REVISION_overuse"
        if b.gold_status == "SUPERSEDED" and "SUPERSEDES" not in action_types:
            return "missed_SUPERSEDES"
        if b.gold_status == "BLOCKED" and "BLOCKS" not in action_types:
            return "missed_BLOCKS"
        return "under_update"

    # Gold says usable but Stage A revised it away -> over-update.
    if gold_c == "USABLE" and a_c == "NOT_USABLE":
        if "RELEASES" in action_types or b.gold_status == "AUTHORIZED":
            # Expected a release/no-op but a blocking/superseding action stuck.
            return "over_update"
        return "over_update"

    # Uncertainty handling mismatches.
    if gold_c == "UNCERTAIN" and a_c != "UNCERTAIN":
        return "uncertainty_collapse"
    if a_c == "UNCERTAIN" and gold_c != "UNCERTAIN":
        if b.gold_status == "AUTHORIZED":
            return "under_update"
        return "uncertainty_collapse"

    if real_actions and gold_c == "USABLE":
        return "wrong_action_type"

    return "unknown"


def build_failure_rows(beliefs: list[BeliefAnalysis]) -> list[dict[str, Any]]:
    """Build the ordered failure-analysis table rows from resolved belief inputs."""
    rows: list[dict[str, Any]] = []
    for b in beliefs:
        gold_c = _comparable(b.gold_status)
        a_c = _comparable(b.a_status)
        b_canon_c = _comparable(b.b_canonical_verdict)
        b_strict_c = _comparable(b.b_strict_verdict)
        a_correct = a_c == gold_c
        b_correct = b_canon_c == gold_c
        category = classify_failure(b)
        # Promote the metric-artifact label when Stage A failed, Stage B's strict
        # verdict also failed, but canonicalization rescued Stage B.
        if (
            category != "none"
            and not a_correct
            and b_correct
            and b_strict_c != gold_c
        ):
            category = "Stage_B_canonicalization_advantage"

        rows.append({
            "case_id": b.case_id,
            "episode_id": b.episode_id,
            "belief_id": b.belief_id,
            "stage_a_final_status": b.a_status or "MISSING",
            "stage_b_final_status": b_canon_c,
            "gold_final_status": b.gold_status,
            "a_correct": a_correct,
            "b_correct": b_correct,
            "a_typed_actions": _format_actions(b.a_actions),
            "a_gate_decisions": _format_gate(b.a_gate_decisions),
            "b_raw_verdict": b.b_raw_verdict or "MISSING",
            "b_canonicalized_verdict": b.b_canonical_verdict or "MISSING",
            "failure_category": category,
            "suggested_fix": _SUGGESTED_FIX.get(category, ""),
        })
    return rows


def _format_actions(actions: tuple[dict[str, Any], ...]) -> str:
    parts = []
    for a in actions:
        t = a.get("action_type", "?")
        tgt = a.get("target_belief_id") or a.get("target_condition_id") or "-"
        repl = a.get("replacement_belief_id")
        parts.append(f"{t}->{tgt}" + (f"=>{repl}" if repl else ""))
    return "; ".join(parts) if parts else "NONE"


def _format_gate(decisions: tuple[dict[str, Any], ...]) -> str:
    parts = []
    for d in decisions:
        verdict = "admit" if d.get("admitted", True) else "reject"
        parts.append(f"{d.get('edge_type', '?')}:{verdict}({d.get('reason', '')})")
    return "; ".join(parts) if parts else "NONE"


_COLUMNS: tuple[str, ...] = (
    "case_id",
    "episode_id",
    "belief_id",
    "stage_a_final_status",
    "stage_b_final_status",
    "gold_final_status",
    "a_correct",
    "b_correct",
    "a_typed_actions",
    "a_gate_decisions",
    "b_raw_verdict",
    "b_canonicalized_verdict",
    "failure_category",
    "suggested_fix",
)


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    """Serialize rows to CSV text with a stable column order."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def summarize_categories(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Return (category, count) pairs sorted by descending count, excluding 'none'."""
    counts: dict[str, int] = {}
    for r in rows:
        cat = r["failure_category"]
        if cat == "none":
            continue
        counts[cat] = counts.get(cat, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def rows_to_markdown(rows: list[dict[str, Any]], *, manifest: dict[str, Any] | None = None) -> str:
    """Render the failure-analysis report as Markdown (summary + full table)."""
    lines: list[str] = ["# Stage A vs Stage B failure analysis", ""]
    if manifest is not None:
        lines += [
            f"- commit: `{manifest.get('git_commit')}` (branch `{manifest.get('branch')}`)",
            f"- provider: `{manifest.get('provider')}` (mode `{manifest.get('provider_mode')}`), "
            f"model `{manifest.get('model')}`, temperature {manifest.get('temperature')}",
            f"- constrained: {manifest.get('constrained')}, cache: {manifest.get('cache_enabled')}, "
            f"dataset: `{manifest.get('data_split')}`, max_cases: {manifest.get('max_cases')}",
            f"- live API run: {manifest.get('live_api_run')}; key env var: "
            f"`{manifest.get('api_key_env')}` (present: {manifest.get('api_key_present')})",
            "",
        ]
    total = len(rows)
    a_correct = sum(1 for r in rows if r["a_correct"])
    b_correct = sum(1 for r in rows if r["b_correct"])
    a_only_wrong = sum(1 for r in rows if not r["a_correct"] and r["b_correct"])
    lines += [
        "## Summary",
        "",
        f"- beliefs compared: {total}",
        f"- Stage A correct: {a_correct}/{total}",
        f"- Stage B correct: {b_correct}/{total}",
        f"- Stage A wrong while Stage B correct: {a_only_wrong}",
        "",
        "### Top Stage A failure modes",
        "",
    ]
    cat_summary = summarize_categories(rows)
    if cat_summary:
        lines.append("| failure_category | count |")
        lines.append("| --- | --- |")
        for cat, n in cat_summary:
            lines.append(f"| {cat} | {n} |")
    else:
        lines.append("_No Stage A failures (all beliefs correct)._")
    lines += ["", "## Per-belief table", ""]
    lines.append("| " + " | ".join(_COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in _COLUMNS) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in _COLUMNS) + " |")
    lines.append("")
    return "\n".join(lines)


def build_manifest(
    *,
    git_commit: str,
    branch: str,
    provider: str,
    provider_mode: str,
    model: str,
    temperature: float,
    constrained: bool,
    cache_enabled: bool,
    prompt_version: str,
    data_split: str,
    max_cases: int | None,
    timestamp: str,
    api_key_env: str,
    api_key_present: bool,
    live_api_run: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the run manifest.

    SECURITY: only the *name* of the API key env var is recorded (``api_key_env``)
    plus a boolean ``api_key_present``. The key value is never accepted or stored.
    """
    manifest = {
        "git_commit": git_commit,
        "branch": branch,
        "provider": provider,
        "provider_mode": provider_mode,
        "model": model,
        "temperature": temperature,
        "constrained": constrained,
        "cache_enabled": cache_enabled,
        "prompt_version": prompt_version,
        "data_split": data_split,
        "max_cases": max_cases,
        "timestamp": timestamp,
        "api_key_env": api_key_env,
        "api_key_present": api_key_present,
        "live_api_run": live_api_run,
    }
    if extra:
        manifest.update(extra)
    return manifest
