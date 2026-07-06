#!/usr/bin/env python
"""Audit real-seeded MemPatch public/label files and generate stats."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mempatch.benchmark.leakage import audit_public_rows
from tools.real_seeded.common import (
    FORBIDDEN_PUBLIC_FIELDS,
    PUBLIC_ALLOWED_TOP_LEVEL,
    VALID_DECISIONS,
    VALID_FAILURE_MODES,
    VALID_OPERATIONS,
    VALID_STATUSES,
    PipelineError,
    public_github_url,
    read_jsonl,
    sensitive_findings,
    sha256_file,
    summarize_counter,
)


def _walk_keys(value: Any, prefix: str = "$") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            out.append((path, str(key)))
            out.extend(_walk_keys(item, path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            out.extend(_walk_keys(item, f"{prefix}[{idx}]"))
    return out


def forbidden_field_violations(public_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = audit_public_rows(public_rows)
    for row in public_rows:
        paths = []
        for key in row:
            if key not in PUBLIC_ALLOWED_TOP_LEVEL:
                paths.append(f"$.{key}")
        for path, key in _walk_keys(row):
            if key in FORBIDDEN_PUBLIC_FIELDS or key.startswith("expected_"):
                paths.append(path)
        if paths:
            violations.append({"scenario_id": row.get("scenario_id"), "paths": sorted(set(paths))})
    return violations


def evidence_id_violations(public_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_by_id = {row["scenario_id"]: row for row in public_rows}
    violations: list[dict[str, Any]] = []
    for label in label_rows:
        sid = label.get("scenario_id")
        public = public_by_id.get(sid)
        if not public:
            violations.append({"scenario_id": sid, "reason": "missing public row"})
            continue
        events = {
            str(event.get("event_id"))
            for event in (public.get("public_input") or {}).get("events") or []
        }
        unknown = [eid for eid in label.get("expected_evidence_event_ids") or [] if str(eid) not in events]
        if unknown:
            violations.append({"scenario_id": sid, "unknown_evidence_event_ids": unknown})
    return violations


def source_url_violations(label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in label_rows:
        urls = list(row.get("source_urls") or [])
        for pointer in row.get("source_pointers") or []:
            if isinstance(pointer, dict) and pointer.get("url_or_id"):
                urls.append(pointer["url_or_id"])
        bad = [url for url in urls if not public_github_url(url)]
        if bad or not urls:
            violations.append({"scenario_id": row.get("scenario_id"), "bad_urls": bad, "url_count": len(urls)})
    return violations


def provenance_violations(label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = []
    for row in label_rows:
        if not row.get("provenance_license_notes"):
            violations.append({"scenario_id": row.get("scenario_id"), "reason": "missing provenance/license note"})
        for pointer in row.get("source_pointers") or []:
            if isinstance(pointer, dict) and not pointer.get("license_or_terms_note"):
                violations.append({"scenario_id": row.get("scenario_id"), "reason": "source pointer missing license note"})
    return violations


def duplicate_source_violations(label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, str] = {}
    violations: list[dict[str, Any]] = []
    for row in label_rows:
        sid = str(row.get("scenario_id"))
        source_key = str(row.get("original_candidate_id") or "")
        if not source_key:
            urls = row.get("source_urls") or []
            source_key = str(urls[0]) if urls else sid
        if source_key in seen:
            violations.append(
                {"scenario_id": sid, "duplicate_source": source_key, "first_scenario_id": seen[source_key]}
            )
        else:
            seen[source_key] = sid
    return violations


def state_transition_violations(label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in label_rows:
        reasons: list[str] = []
        if row.get("expected_decision") not in VALID_DECISIONS:
            reasons.append("invalid expected_decision")
        if row.get("expected_memory_operation") not in VALID_OPERATIONS:
            reasons.append("invalid expected_memory_operation")
        if row.get("expected_failure_diagnosis") not in VALID_FAILURE_MODES:
            reasons.append("invalid expected_failure_diagnosis")
        if row.get("failure_mode") not in VALID_FAILURE_MODES:
            reasons.append("invalid failure_mode")
        states = row.get("expected_memory_states")
        if not isinstance(states, list) or not states:
            reasons.append("missing expected_memory_states")
        else:
            for state in states:
                if not isinstance(state, dict) or state.get("status") not in VALID_STATUSES:
                    reasons.append("invalid memory status")
                    break
        if reasons:
            violations.append({"scenario_id": row.get("scenario_id"), "reasons": sorted(set(reasons))})
    return violations


def balance_warnings(label_rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    total = len(label_rows)
    if not total:
        return ["no label rows"]
    repo_counts = Counter(str(row.get("source_repo")) for row in label_rows)
    failure_counts = Counter(str(row.get("failure_mode")) for row in label_rows)
    operation_counts = Counter(str(row.get("expected_memory_operation")) for row in label_rows)
    if repo_counts and repo_counts.most_common(1)[0][1] / total > 0.35:
        warnings.append("per-repo balance is skewed above 35% for one repository")
    if len(failure_counts) < 6:
        warnings.append("fewer than 6 distinct failure modes")
    if len(operation_counts) < 4:
        warnings.append("fewer than 4 distinct memory operations")
    return warnings


def audit(public_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> dict[str, Any]:
    public_by_id = {row.get("scenario_id") for row in public_rows}
    label_by_id = {row.get("scenario_id") for row in label_rows}
    sensitive_public = [
        {"scenario_id": row.get("scenario_id"), "findings": sensitive_findings(row)}
        for row in public_rows
        if sensitive_findings(row)
    ]
    sensitive_labels = [
        {"scenario_id": row.get("scenario_id"), "findings": sensitive_findings(row)}
        for row in label_rows
        if sensitive_findings(row)
    ]
    return {
        "schema_version": "mempatch_bench_final",
        "public_rows": len(public_rows),
        "label_rows": len(label_rows),
        "public_private_id_mismatches": sorted(public_by_id ^ label_by_id),
        "public_forbidden_field_violations": forbidden_field_violations(public_rows),
        "evidence_id_violations": evidence_id_violations(public_rows, label_rows),
        "source_url_violations": source_url_violations(label_rows),
        "provenance_violations": provenance_violations(label_rows),
        "sensitive_content_violations": sensitive_public + sensitive_labels,
        "duplicate_source_violations": duplicate_source_violations(label_rows),
        "state_transition_violations": state_transition_violations(label_rows),
        "per_repo_distribution": summarize_counter(label_rows, "source_repo"),
        "per_failure_mode_distribution": summarize_counter(label_rows, "failure_mode"),
        "per_operation_distribution": summarize_counter(label_rows, "expected_memory_operation"),
        "balance_warnings": balance_warnings(label_rows),
    }


def has_blocking_violations(report: dict[str, Any]) -> bool:
    blocking_keys = [
        "public_private_id_mismatches",
        "public_forbidden_field_violations",
        "evidence_id_violations",
        "source_url_violations",
        "provenance_violations",
        "sensitive_content_violations",
        "duplicate_source_violations",
        "state_transition_violations",
    ]
    return any(report.get(key) for key in blocking_keys)


def markdown_stats(report: dict[str, Any], public_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> str:
    event_counts = [
        len((row.get("public_input") or {}).get("events") or [])
        for row in public_rows
    ]
    average_events = sum(event_counts) / len(event_counts) if event_counts else 0
    lines = [
        "# Real-Seeded Audit Stats",
        "",
        f"- Public rows: {len(public_rows)}",
        f"- Label rows: {len(label_rows)}",
        f"- Average events per case: {average_events:.2f}",
        f"- Blocking audit violations: {'yes' if has_blocking_violations(report) else 'no'}",
        "",
        "## Per-Repo Distribution",
        "",
    ]
    for key, count in report["per_repo_distribution"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Per-Failure-Mode Distribution", ""])
    for key, count in report["per_failure_mode_distribution"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Per-Operation Distribution", ""])
    for key, count in report["per_operation_distribution"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Balance Warnings", ""])
    for warning in report["balance_warnings"] or ["none"]:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def checksums(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths if path.exists() and path.is_file()}


def default_stats_path(out: Path) -> Path:
    return out.with_name("stats_real_seeded.md")


def default_checksums_path(out: Path) -> Path:
    return out.parent.parent / "manifests" / "checksums_real_seeded.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--checksums", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        public_rows = read_jsonl(args.public)
        label_rows = read_jsonl(args.labels)
        report = audit(public_rows, label_rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stats_path = args.stats or default_stats_path(args.out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(markdown_stats(report, public_rows, label_rows), encoding="utf-8")
        checksum_path = args.checksums or default_checksums_path(args.out)
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path.write_text(
            json.dumps(checksums([args.public, args.labels, args.out, stats_path]), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Audit complete: {len(public_rows)} public rows, {len(label_rows)} label rows.")
        if has_blocking_violations(report):
            print("Audit found blocking violations; inspect the JSON report.")
            return 1
    except PipelineError as exc:
        print(f"Audit failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
