#!/usr/bin/env python3
"""Audit MemPatch scenario JSONL for learnable decision boundaries in public_input."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)

from benchmark.general_taxonomy import DECISIONS, canonical_hidden_gold_fields
from scripts.workflows.validate_mempatch_bench_dataset import _is_background_event

NON_ANSWER_DECISIONS = ("ask_clarification", "escalate", "mark_unresolved")
DEFAULT_JACCARD_WARN = 0.35
MIN_TOTAL_NORMALIZED_TEMPLATES = 50
MIN_TEMPLATES_PER_DECISION_VARIANT = 3
CORE_EVENT_FIELDS = ("text", "actor_role", "trust_level", "visibility_scope", "event_type")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: invalid JSON: {exc}") from exc
    return rows


def resolve_data_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "scenarios.jsonl"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"no scenarios.jsonl in directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"scenarios file not found: {path}")
    return path


def infer_split_label(path: Path, scenario: dict[str, Any]) -> str:
    split = scenario.get("public_split_name") or scenario.get("metadata", {}).get("split")
    if split:
        return str(split)
    parent = path.parent.name
    if parent in {"train", "test"}:
        return parent
    return path.stem


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"case-\d{6}", "case-<id>", text)
    text = re.sub(r"case-\d+", "case-<id>", text)
    return text


def core_events(scenario: dict[str, Any], *, max_events: int = 3) -> list[dict[str, Any]]:
    events = scenario.get("public_input", {}).get("event_trace", [])
    picked: list[dict[str, Any]] = []
    for event in events:
        if _is_background_event(event):
            continue
        picked.append(event)
        if len(picked) >= max_events:
            break
    return picked


def event_feature_tuple(event: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        normalize_text(str(event.get(field) or "")) if field == "text" else str(event.get(field) or "")
        for field in CORE_EVENT_FIELDS
    )


def core_event_signature(scenario: dict[str, Any], *, max_events: int = 3) -> dict[str, Any]:
    events = core_events(scenario, max_events=max_events)
    features = [event_feature_tuple(event) for event in events]
    readable_parts: list[str] = []
    for event, feat in zip(events, features):
        readable_parts.append(
            "|".join(
                [
                    f"role={feat[1]}",
                    f"trust={feat[2]}",
                    f"scope={feat[3]}",
                    f"type={feat[4]}",
                    f"text={feat[0][:80]}",
                ]
            )
        )
    readable = " ;; ".join(readable_parts) if readable_parts else "<empty>"
    payload = json.dumps(features, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "hash": digest,
        "readable": readable,
        "event_count": len(events),
        "events_preview": [
            {
                "event_id": event.get("event_id"),
                "actor_role": event.get("actor_role"),
                "trust_level": event.get("trust_level"),
                "visibility_scope": event.get("visibility_scope"),
                "event_type": event.get("event_type"),
                "text": event.get("text"),
            }
            for event in events
        ],
    }


def token_bigrams(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    if len(tokens) < 2:
        return set(tokens)
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def collect_public_event_text(scenarios: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for scenario in scenarios:
        for event in scenario.get("public_input", {}).get("event_trace", []):
            if _is_background_event(event):
                continue
            chunks.append(str(event.get("text") or ""))
    return "\n".join(chunks)


def bigram_jaccard(text_a: str, text_b: str) -> float:
    a = token_bigrams(text_a)
    b = token_bigrams(text_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pattern_value(scenario: dict[str, Any], key: str) -> str | None:
    meta = scenario.get("metadata") or {}
    value = meta.get(key) or scenario.get(key)
    return str(value) if value is not None else None


def pattern_cross_table(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        pattern = pattern_value(row, key)
        if not pattern:
            continue
        gold = canonical_hidden_gold_fields(row.get("hidden_gold") or {})
        decision = gold.get("expected_decision") or "<missing>"
        table[pattern][decision] += 1
    return {pattern: dict(counts) for pattern, counts in sorted(table.items())}


def sanitized_public_view(scenario: dict[str, Any]) -> str:
    """Full public view with ids/timestamps normalized for collision detection."""
    public = scenario.get("public_input") or {}
    parts: list[str] = [normalize_text(str(scenario.get("workflow_context") or ""))]
    for event in public.get("event_trace") or []:
        if _is_background_event(event):
            continue
        parts.append(
            "|".join(
                [
                    normalize_text(str(event.get("text") or "")),
                    str(event.get("actor_role") or ""),
                    str(event.get("trust_level") or ""),
                    str(event.get("visibility_scope") or ""),
                    str(event.get("event_type") or ""),
                ]
            )
        )
    for memory in public.get("initial_memory") or []:
        parts.append(normalize_text(str(memory.get("text") or "")))
    return "\n".join(parts)


def public_view_hash(scenario: dict[str, Any]) -> str:
    payload = sanitized_public_view(scenario)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decision_variant_values() -> set[str]:
    try:
        from benchmark.generation.blueprints import all_variants

        return {variant.variant_id for _family, variant in all_variants()}
    except ImportError:
        return set()


def marker_leakage_violations(scenario: dict[str, Any], *, decision_variants: set[str]) -> list[str]:
    """Return public event text leakage findings for release-blocking markers."""
    sid = str(scenario.get("scenario_id") or "<unknown>")
    variants = decision_variants | {
        str((scenario.get("metadata") or {}).get("decision_variant") or "")
    }
    variants.discard("")
    checks: list[tuple[str, re.Pattern[str]]] = [
        ("literal [trigger:", re.compile(r"\[trigger:", re.I)),
        ("literal trigger:", re.compile(r"trigger:", re.I)),
    ]
    for decision in DECISIONS:
        checks.append((f"decision enum {decision}", re.compile(rf"\b{re.escape(decision)}\b", re.I)))
    for variant in variants:
        checks.append((f"decision_variant {variant}", re.compile(rf"\b{re.escape(variant)}\b", re.I)))

    findings: list[str] = []
    for event in scenario.get("public_input", {}).get("event_trace", []) or []:
        text = str(event.get("text") or "")
        for label, pattern in checks:
            if pattern.search(text):
                event_id = event.get("event_id") or "<missing-event-id>"
                findings.append(f"{sid}/{event_id}: public event text leaks {label}")
    return findings


def metadata_triggers(scenario: dict[str, Any]) -> list[str]:
    meta = scenario.get("metadata") or {}
    raw = meta.get("decision_triggers")
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if raw is not None:
        return [str(raw)]
    return []


def is_mark_ci_derived(scenario: dict[str, Any]) -> bool:
    meta = scenario.get("metadata") or {}
    if meta.get("mark_ci_derived") is True:
        return True
    triggers = metadata_triggers(scenario)
    ci_prefix = ("ci_second_verified_contradiction", "ci_passive_monitor_gap", "ci_no_authority_path")
    return any(t in ci_prefix for t in triggers)


def detect_triggers_in_public(scenario: dict[str, Any]) -> set[str]:
    try:
        from benchmark.generation.decision_resolver import detect_triggers

        return detect_triggers(scenario.get("public_input") or {})
    except ImportError:
        return set()


ASK_ANSWER_RE = re.compile(r"\b(clarif|ask (the )?user|need.*(scope|target|confirm))\b", re.I)
ESCALATE_ANSWER_RE = re.compile(
    r"\b(escalat|human review|approval required|policy block|compliance block)\b", re.I
)
UNRESOLVED_ANSWER_RE = re.compile(
    r"\b(unresolved|cannot determine|conflict|insufficient evidence|no authority)\b", re.I
)


def audit_dataset(
    path: Path,
    *,
    max_core_events: int,
    jaccard_warn: float,
) -> dict[str, Any]:
    data_path = resolve_data_path(path)
    rows = read_jsonl(data_path)
    split_label = infer_split_label(data_path, rows[0]) if rows else data_path.parent.name

    decision_counts = Counter()
    signature_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signature_meta: dict[str, dict[str, Any]] = {}
    public_view_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    template_hashes_by_decision: dict[str, set[str]] = defaultdict(set)
    template_hashes_by_variant: dict[str, set[str]] = defaultdict(set)
    audited_rows: list[dict[str, Any]] = []
    leakage_violations: list[str] = []
    known_decision_variants = decision_variant_values()
    trigger_coverage: dict[str, dict[str, int]] = {
        d: {"with_metadata": 0, "with_public_trigger": 0, "total": 0}
        for d in NON_ANSWER_DECISIONS
    }
    mark_unresolved_breakdown = Counter({"ci_derived": 0, "non_ci": 0})
    gold_consistency_violations: list[str] = []

    for row in rows:
        gold = canonical_hidden_gold_fields(row.get("hidden_gold") or {})
        decision = gold.get("expected_decision") or "<missing>"
        decision_counts[decision] += 1
        sig = core_event_signature(row, max_events=max_core_events)
        signature_to_rows[sig["hash"]].append(
            {
                "scenario_id": row.get("scenario_id"),
                "expected_decision": decision,
                "signature": sig,
            }
        )
        signature_meta[sig["hash"]] = sig
        pv_hash = public_view_hash(row)
        meta = row.get("metadata") or {}
        variant = str(meta.get("decision_variant") or "<missing>")
        template_hashes_by_decision[decision].add(pv_hash)
        template_hashes_by_variant[variant].add(pv_hash)
        public_view_to_rows[pv_hash].append(
            {"scenario_id": row.get("scenario_id"), "expected_decision": decision}
        )
        audited_rows.append(row)
        leakage_violations.extend(
            marker_leakage_violations(row, decision_variants=known_decision_variants)
        )

        if decision in NON_ANSWER_DECISIONS:
            trigger_coverage[decision]["total"] += 1
            meta_triggers = metadata_triggers(row)
            public_triggers = detect_triggers_in_public(row)
            if meta_triggers:
                trigger_coverage[decision]["with_metadata"] += 1
            if public_triggers:
                trigger_coverage[decision]["with_public_trigger"] += 1

        if decision == "mark_unresolved":
            mark_unresolved_breakdown["ci_derived" if is_mark_ci_derived(row) else "non_ci"] += 1

        answer = str(gold.get("expected_answer") or "")
        public_triggers = detect_triggers_in_public(row)
        sid = str(row.get("scenario_id") or "<unknown>")
        try:
            from benchmark.generation.blueprints import (
                ASK_TRIGGERS,
                ESCALATE_TRIGGERS,
                MARK_CI_TRIGGERS,
                MARK_NON_CI_TRIGGERS,
            )
        except ImportError:
            ASK_TRIGGERS = ESCALATE_TRIGGERS = MARK_CI_TRIGGERS = MARK_NON_CI_TRIGGERS = ()  # type: ignore

        if ASK_ANSWER_RE.search(answer):
            if not (public_triggers & set(ASK_TRIGGERS)):
                gold_consistency_violations.append(
                    f"{sid}: answer implies ask/clarify but no ask trigger in public_input"
                )
        if ESCALATE_ANSWER_RE.search(answer):
            if not (public_triggers & set(ESCALATE_TRIGGERS)):
                gold_consistency_violations.append(
                    f"{sid}: answer implies escalate/approval but no escalation trigger in public_input"
                )
        if UNRESOLVED_ANSWER_RE.search(answer):
            mark_triggers = set(MARK_NON_CI_TRIGGERS) | set(MARK_CI_TRIGGERS)
            if not (public_triggers & mark_triggers):
                gold_consistency_violations.append(
                    f"{sid}: answer implies unresolved but no mark trigger in public_input"
                )

    signature_decisions: dict[str, set[str]] = {
        sig: {entry["expected_decision"] for entry in entries}
        for sig, entries in signature_to_rows.items()
    }
    collision_groups: list[dict[str, Any]] = []
    for sig, decisions in signature_decisions.items():
        if len(decisions) < 2:
            continue
        entries = signature_to_rows[sig]
        collision_groups.append(
            {
                "core_event_signature_hash": sig,
                "core_event_signature_readable": signature_meta[sig]["readable"],
                "involved_decisions": sorted(decisions),
                "scenario_ids": [entry["scenario_id"] for entry in entries],
                "count": len(entries),
                "events_preview": signature_meta[sig]["events_preview"],
            }
        )
    collision_groups.sort(key=lambda item: (-item["count"], item["core_event_signature_hash"]))

    non_answer_collisions = [
        group
        for group in collision_groups
        if len(set(group["involved_decisions"]) & set(NON_ANSWER_DECISIONS)) >= 2
    ]

    public_view_collisions: list[dict[str, Any]] = []
    for pv_hash, entries in public_view_to_rows.items():
        decisions = {e["expected_decision"] for e in entries}
        if len(decisions) < 2:
            continue
        public_view_collisions.append(
            {
                "public_view_hash": pv_hash,
                "involved_decisions": sorted(decisions),
                "scenario_ids": [e["scenario_id"] for e in entries],
                "count": len(entries),
            }
        )
    public_view_collisions.sort(key=lambda item: (-item["count"], item["public_view_hash"]))

    non_answer_public_collisions = [
        g
        for g in public_view_collisions
        if len(set(g["involved_decisions"]) & set(NON_ANSWER_DECISIONS)) >= 2
    ]

    ask_rows = [r for r in audited_rows if canonical_hidden_gold_fields(r.get("hidden_gold") or {}).get("expected_decision") == "ask_clarification"]
    esc_rows = [r for r in audited_rows if canonical_hidden_gold_fields(r.get("hidden_gold") or {}).get("expected_decision") == "escalate"]
    ask_esc_jaccard = bigram_jaccard(
        collect_public_event_text(ask_rows),
        collect_public_event_text(esc_rows),
    )
    jaccard_flags: list[str] = []
    if ask_rows and esc_rows and ask_esc_jaccard >= jaccard_warn:
        jaccard_flags.append(
            f"ask_clarification vs escalate bigram Jaccard={ask_esc_jaccard:.3f} >= {jaccard_warn:.3f}"
        )

    ask_esc_signature_overlap = sorted(
        sig
        for sig, decisions in signature_decisions.items()
        if "ask_clarification" in decisions and "escalate" in decisions
    )

    pattern_tables: dict[str, Any] = {}
    for key in ("pattern", "pattern_trap_type", "decision_variant", "decision_triggers"):
        table = pattern_cross_table(audited_rows, key)
        if table:
            pattern_tables[key] = table

    fatal_issues: list[str] = []
    if ask_esc_signature_overlap:
        fatal_issues.append(
            f"{split_label}: ask_clarification and escalate share {len(ask_esc_signature_overlap)} core_event_signature hash(es)"
        )
    if non_answer_collisions:
        fatal_issues.append(
            f"{split_label}: {len(non_answer_collisions)} core_event_signature collision(s) among ask/escalate/mark"
        )
    if non_answer_public_collisions:
        fatal_issues.append(
            f"{split_label}: {len(non_answer_public_collisions)} full public-view collision(s) among ask/escalate/mark"
        )
    for decision, stats in trigger_coverage.items():
        if stats["total"] and stats["with_public_trigger"] < stats["total"]:
            missing = stats["total"] - stats["with_public_trigger"]
            fatal_issues.append(
                f"{split_label}: {decision} missing public trigger coverage on {missing}/{stats['total']} rows"
            )
    for violation in gold_consistency_violations[:20]:
        fatal_issues.append(f"{split_label}: gold_public_consistency: {violation}")
    if len(gold_consistency_violations) > 20:
        fatal_issues.append(
            f"{split_label}: gold_public_consistency: ... +{len(gold_consistency_violations) - 20} more"
        )
    for violation in leakage_violations[:20]:
        fatal_issues.append(f"{split_label}: public_marker_leakage: {violation}")
    if len(leakage_violations) > 20:
        fatal_issues.append(
            f"{split_label}: public_marker_leakage: ... +{len(leakage_violations) - 20} more"
        )

    return {
        "split": split_label,
        "path": str(data_path),
        "count": len(rows),
        "decision_counts": {d: decision_counts.get(d, 0) for d in DECISIONS},
        "extra_decisions": {
            k: v for k, v in sorted(decision_counts.items()) if k not in DECISIONS
        },
        "core_event_signature_collisions": collision_groups,
        "non_answer_decision_collisions": non_answer_collisions,
        "public_view_collisions": public_view_collisions,
        "non_answer_public_view_collisions": non_answer_public_collisions,
        "ask_escalate_bigram_jaccard": ask_esc_jaccard if ask_rows and esc_rows else None,
        "ask_escalate_jaccard_flags": jaccard_flags,
        "ask_escalate_shared_signatures": ask_esc_signature_overlap,
        "pattern_tables": pattern_tables,
        "decision_triggers_coverage": trigger_coverage,
        "mark_unresolved_breakdown": dict(mark_unresolved_breakdown),
        "gold_public_consistency_violations": gold_consistency_violations,
        "public_marker_leakage_violations": leakage_violations,
        "normalized_public_view_template_count": len(public_view_to_rows),
        "normalized_public_view_template_hashes": sorted(public_view_to_rows),
        "templates_per_decision": {
            decision: len(template_hashes_by_decision.get(decision, set()))
            for decision in DECISIONS
        },
        "template_hashes_by_decision": {
            decision: sorted(template_hashes_by_decision.get(decision, set()))
            for decision in DECISIONS
        },
        "templates_per_decision_variant": {
            variant: len(hashes) for variant, hashes in sorted(template_hashes_by_variant.items())
        },
        "template_hashes_by_decision_variant": {
            variant: sorted(hashes) for variant, hashes in sorted(template_hashes_by_variant.items())
        },
        "fatal_issues": fatal_issues,
        "top_collision_groups": collision_groups[:10],
        "top_public_view_collisions": public_view_collisions[:10],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# MemPatch decision boundary audit",
        "",
        f"- datasets: {len(report['datasets'])}",
        f"- total scenarios: {report['summary']['total_scenarios']}",
        f"- total signature collisions (cross-decision): {report['summary']['total_collision_groups']}",
        f"- ask↔escalate shared signatures: {report['summary']['ask_escalate_shared_signature_count']}",
        f"- non-answer public-view collisions: {report['summary']['non_answer_public_view_collision_count']}",
        f"- normalized public-view templates: {report['summary']['normalized_public_view_template_count']}",
        f"- fatal issues: {len(report['summary']['fatal_issues'])}",
        "",
    ]
    if report["summary"]["fatal_issues"]:
        lines.append("## Fatal issues")
        for issue in report["summary"]["fatal_issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    lines.append("## Template diversity")
    lines.append("")
    lines.append(
        f"- total normalized public-view templates: "
        f"{report['summary']['normalized_public_view_template_count']}"
    )
    for pair, count in report["summary"].get("split_template_overlap", {}).items():
        lines.append(f"- {pair}: {count}")
    lines.append("")
    lines.append("### templates per decision")
    lines.append("")
    for decision, count in report["summary"].get("templates_per_decision", {}).items():
        lines.append(f"- {decision}: {count}")
    lines.append("")
    lines.append("### templates per decision_variant")
    lines.append("")
    for variant, count in report["summary"].get("templates_per_decision_variant", {}).items():
        lines.append(f"- {variant}: {count}")
    lines.append("")

    for dataset in report["datasets"]:
        lines.extend(
            [
                f"## Split: {dataset['split']} (`{dataset['path']}`)",
                "",
                f"- rows: {dataset['count']}",
                "",
                "### expected_decision",
                "",
            ]
        )
        for decision in DECISIONS:
            count = dataset["decision_counts"].get(decision, 0)
            if count:
                lines.append(f"- {decision}: {count}")
        lines.append("")

        if dataset.get("ask_escalate_bigram_jaccard") is not None:
            lines.append(
                f"- ask↔escalate bigram Jaccard: **{dataset['ask_escalate_bigram_jaccard']:.3f}**"
            )
            for flag in dataset.get("ask_escalate_jaccard_flags", []):
                lines.append(f"- ⚠ {flag}")
            lines.append("")

        if dataset.get("decision_triggers_coverage"):
            lines.append("### decision_triggers coverage")
            lines.append("")
            for decision, stats in dataset["decision_triggers_coverage"].items():
                lines.append(
                    f"- {decision}: public={stats['with_public_trigger']}/{stats['total']}, "
                    f"metadata={stats['with_metadata']}/{stats['total']}"
                )
            lines.append("")

        lines.append("### normalized public-view templates")
        lines.append("")
        lines.append(f"- total templates in split: {dataset['normalized_public_view_template_count']}")
        for decision, count in dataset.get("templates_per_decision", {}).items():
            lines.append(f"- {decision}: {count}")
        lines.append("")

        if dataset.get("mark_unresolved_breakdown"):
            lines.append("### mark_unresolved breakdown")
            lines.append("")
            for key, count in dataset["mark_unresolved_breakdown"].items():
                lines.append(f"- {key}: {count}")
            lines.append("")

        if dataset.get("top_public_view_collisions"):
            lines.append("### Top public-view collision groups")
            lines.append("")
            for group in dataset["top_public_view_collisions"][:5]:
                lines.append(f"- hash `{group['public_view_hash'][:16]}...`")
                lines.append(f"  - decisions: {', '.join(group['involved_decisions'])}")
                lines.append(f"  - count: {group['count']}")
                lines.append("")
        if dataset.get("pattern_tables"):
            lines.append("### pattern × decision")
            lines.append("")
            for key, table in dataset["pattern_tables"].items():
                lines.append(f"#### {key}")
                lines.append("")
                for pattern, counts in table.items():
                    total = sum(counts.values())
                    parts = ", ".join(f"{decision}={count}" for decision, count in sorted(counts.items()))
                    lines.append(f"- `{pattern}` (n={total}): {parts}")
                lines.append("")

        if dataset.get("top_collision_groups"):
            lines.append("### Top collision groups")
            lines.append("")
            for group in dataset["top_collision_groups"][:5]:
                lines.append(f"- hash `{group['core_event_signature_hash'][:16]}...`")
                lines.append(f"  - decisions: {', '.join(group['involved_decisions'])}")
                lines.append(f"  - count: {group['count']}")
                lines.append(f"  - scenario_ids: {', '.join(group['scenario_ids'][:8])}")
                if len(group["scenario_ids"]) > 8:
                    lines.append(f"  - ... +{len(group['scenario_ids']) - 8} more")
                lines.append(f"  - signature: {group['core_event_signature_readable'][:200]}")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MemPatch decision boundaries in scenario JSONL.")
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        type=Path,
        help="scenarios.jsonl or directory containing it (repeatable)",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument(
        "--max-core-events",
        type=int,
        default=3,
        help="Number of leading non-background events in core signature (default: 3)",
    )
    parser.add_argument(
        "--jaccard-warn",
        type=float,
        default=DEFAULT_JACCARD_WARN,
        help=f"Flag ask vs escalate bigram Jaccard at or above this threshold (default: {DEFAULT_JACCARD_WARN})",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 even when release-blocking audit failures are found",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = [
        audit_dataset(
            path,
            max_core_events=max(2, min(args.max_core_events, 3)),
            jaccard_warn=args.jaccard_warn,
        )
        for path in args.data
    ]

    fatal_issues: list[str] = []
    total_collisions = 0
    ask_esc_shared = 0
    non_answer_public_collisions = 0
    total_rows = 0
    pattern_decision_split: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    template_hashes_by_split: dict[str, set[str]] = {}
    template_hashes_by_variant: dict[str, set[str]] = defaultdict(set)
    template_hashes_by_decision: dict[str, set[str]] = defaultdict(set)

    for dataset in datasets:
        fatal_issues.extend(dataset["fatal_issues"])
        total_collisions += len(dataset["core_event_signature_collisions"])
        ask_esc_shared += len(dataset["ask_escalate_shared_signatures"])
        non_answer_public_collisions += len(dataset.get("non_answer_public_view_collisions", []))
        total_rows += dataset["count"]
        split = dataset["split"]
        template_hashes_by_split[split] = set(dataset.get("normalized_public_view_template_hashes", []))
        for decision, hashes in dataset.get("template_hashes_by_decision", {}).items():
            template_hashes_by_decision[decision].update(hashes)
        for variant, hashes in dataset.get("template_hashes_by_decision_variant", {}).items():
            template_hashes_by_variant[variant].update(hashes)
        for key in ("pattern", "decision_variant"):
            table = dataset.get("pattern_tables", {}).get(key, {})
            for pattern, counts in table.items():
                for decision, count in counts.items():
                    pattern_decision_split[f"{key}:{pattern}"][split][decision] += count

    all_template_hashes = set().union(*template_hashes_by_split.values()) if template_hashes_by_split else set()
    split_template_overlap: dict[str, int] = {}
    split_names = sorted(template_hashes_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            split_template_overlap[f"{left} ∩ {right}"] = len(
                template_hashes_by_split[left] & template_hashes_by_split[right]
            )

    templates_per_variant = {
        variant: len(hashes) for variant, hashes in sorted(template_hashes_by_variant.items())
    }
    templates_per_decision = {
        decision: len(template_hashes_by_decision.get(decision, set()))
        for decision in DECISIONS
    }
    low_diversity_variants = {
        variant: count
        for variant, count in templates_per_variant.items()
        if count < MIN_TEMPLATES_PER_DECISION_VARIANT
    }
    if len(all_template_hashes) < MIN_TOTAL_NORMALIZED_TEMPLATES:
        fatal_issues.append(
            f"release: normalized public-view templates {len(all_template_hashes)} "
            f"< {MIN_TOTAL_NORMALIZED_TEMPLATES}"
        )
    if low_diversity_variants:
        fatal_issues.append(
            "release: decision_variant template diversity below "
            f"{MIN_TEMPLATES_PER_DECISION_VARIANT}: {low_diversity_variants}"
        )

    report = {
        "summary": {
            "total_scenarios": total_rows,
            "total_collision_groups": total_collisions,
            "ask_escalate_shared_signature_count": ask_esc_shared,
            "non_answer_public_view_collision_count": non_answer_public_collisions,
            "normalized_public_view_template_count": len(all_template_hashes),
            "split_template_overlap": split_template_overlap,
            "templates_per_decision": templates_per_decision,
            "templates_per_decision_variant": templates_per_variant,
            "fatal_issues": fatal_issues,
            "jaccard_warn_threshold": args.jaccard_warn,
            "pattern_decision_split_matrix": {
                key: {split: dict(counts) for split, counts in splits.items()}
                for key, splits in sorted(pattern_decision_split.items())
            },
        },
        "datasets": datasets,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.out_json}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote {args.out_md}")

    print(
        f"Audited {len(datasets)} dataset(s), {total_rows} scenarios, "
        f"{total_collisions} signature collision group(s), "
        f"{ask_esc_shared} ask↔escalate shared signature(s), "
        f"{non_answer_public_collisions} non-answer public-view collision group(s)"
    )
    for dataset in datasets:
        print(f"  {dataset['split']}: collisions={len(dataset['core_event_signature_collisions'])}")
        if dataset["ask_escalate_jaccard_flags"]:
            for flag in dataset["ask_escalate_jaccard_flags"]:
                print(f"    {flag}")

    if fatal_issues and not args.no_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
