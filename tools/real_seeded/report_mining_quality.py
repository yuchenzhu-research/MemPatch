#!/usr/bin/env python
"""Summarize raw GitHub mining quality before filtering or normalization."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.real_seeded.common import read_jsonl, sanitize_text

AUTHORITATIVE_TYPES = {"maintainer_comment", "pr_merged", "release_note", "changelog", "docs"}
RELEASE_DOC_TYPES = {"release_note", "changelog", "docs"}


def _events(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in row.get("raw_events") or [] if isinstance(event, dict)]


def _group(row: dict[str, Any]) -> str:
    terms = row.get("retrieval_query_terms") or {}
    return str(terms.get("group") or "unknown")


def _primary_source(row: dict[str, Any]) -> str:
    urls = row.get("source_urls") or []
    if urls:
        return str(urls[0])
    return f"{row.get('source_repo')}#{row.get('issue_number')}"


def _load_json(path: Path | None) -> Any:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8") or "{}")


def mining_quality(rows: list[dict[str, Any]], mining_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    mining_summary = mining_summary or {}
    event_counts = [len(_events(row)) for row in rows]
    source_counts = Counter(_primary_source(row) for row in rows)
    duplicates = sum(count - 1 for count in source_counts.values() if count > 1)

    def has_event_type(row: dict[str, Any], types: set[str]) -> bool:
        return any(str(event.get("source_type")) in types for event in _events(row))

    def has_linked_pr(row: dict[str, Any]) -> bool:
        return bool(row.get("pr_numbers")) or has_event_type(row, {"pr_merged"})

    open_count = sum(1 for row in rows if str(row.get("state") or "").lower() == "open")
    merged_pr_count = sum(1 for row in rows if has_event_type(row, {"pr_merged"}))
    closed_or_merged = sum(
        1
        for row in rows
        if str(row.get("state") or "").lower() == "closed" or has_event_type(row, {"pr_merged"})
    )
    open_only = sum(
        1
        for row in rows
        if str(row.get("state") or "").lower() == "open" and not has_event_type(row, AUTHORITATIVE_TYPES)
    )
    release_doc_count = sum(1 for row in rows if has_event_type(row, RELEASE_DOC_TYPES))
    maintainer_count = sum(1 for row in rows if has_event_type(row, {"maintainer_comment"}))
    authoritative_count = sum(1 for row in rows if has_event_type(row, AUTHORITATIVE_TYPES))
    linked_pr_count = sum(1 for row in rows if has_linked_pr(row))
    raw_count = len(rows)
    return {
        "raw_candidates": raw_count,
        "per_repo": dict(sorted(Counter(str(row.get("source_repo") or "unknown") for row in rows).items())),
        "per_query_group": dict(sorted(Counter(_group(row) for row in rows).items())),
        "average_raw_events": round(sum(event_counts) / raw_count, 3) if raw_count else 0.0,
        "candidates_with_maintainer_evidence": maintainer_count,
        "candidates_with_merged_pr_evidence": merged_pr_count,
        "candidates_with_release_changelog_docs_evidence": release_doc_count,
        "candidates_with_any_authoritative_evidence": authoritative_count,
        "linked_pr_candidates": linked_pr_count,
        "linked_pr_ratio": round(linked_pr_count / raw_count, 3) if raw_count else 0.0,
        "release_changelog_docs_ratio": round(release_doc_count / raw_count, 3) if raw_count else 0.0,
        "closed_issue_or_merged_pr_candidates": closed_or_merged,
        "closed_issue_or_merged_pr_ratio": round(closed_or_merged / raw_count, 3) if raw_count else 0.0,
        "open_issue_candidates": open_count,
        "open_issue_ratio": round(open_count / raw_count, 3) if raw_count else 0.0,
        "open_only_candidates": open_only,
        "duplicate_source_count": duplicates,
        "api_failure_counts": mining_summary.get("api_failures") or {},
        "stop_reason": mining_summary.get("stop_reason"),
        "target_raw": mining_summary.get("target_raw"),
        "target_reached": mining_summary.get("target_reached"),
    }


def gate_status(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_candidates_at_least_300": report["raw_candidates"] >= 300,
        "average_raw_events_at_least_3": report["average_raw_events"] >= 3,
        "authoritative_candidates_at_least_100": report["candidates_with_any_authoritative_evidence"] >= 100,
        "duplicate_sources_not_dominant": report["duplicate_source_count"] <= max(10, report["raw_candidates"] * 0.1),
        "not_open_only_dominant": report["open_only_candidates"] <= max(20, report["raw_candidates"] * 0.3),
    }


def markdown(report: dict[str, Any]) -> str:
    gate = gate_status(report)
    lines = [
        "# Real-Seeded Mining Quality Report",
        "",
        f"- Raw candidates: {report['raw_candidates']}",
        f"- Average raw_events per candidate: {report['average_raw_events']}",
        f"- Candidates with linked PR: {report['linked_pr_candidates']} ({report['linked_pr_ratio']:.1%})",
        f"- Candidates with maintainer evidence: {report['candidates_with_maintainer_evidence']}",
        f"- Candidates with merged PR evidence: {report['candidates_with_merged_pr_evidence']}",
        "- Candidates with release/changelog/docs evidence: "
        f"{report['candidates_with_release_changelog_docs_evidence']} ({report['release_changelog_docs_ratio']:.1%})",
        "- Closed issue or merged PR candidates: "
        f"{report['closed_issue_or_merged_pr_candidates']} ({report['closed_issue_or_merged_pr_ratio']:.1%})",
        f"- Open issue candidates: {report['open_issue_candidates']} ({report['open_issue_ratio']:.1%})",
        f"- Open-only candidates: {report['open_only_candidates']}",
        f"- Duplicate source count: {report['duplicate_source_count']}",
        f"- API failure / rate-limit counts: {json.dumps(report['api_failure_counts'], sort_keys=True)}",
        f"- Stop reason: {sanitize_text(report.get('stop_reason'), max_chars=400)}",
        "",
        "## Gate Checks",
        "",
    ]
    for key, ok in gate.items():
        lines.append(f"- {key}: {'PASS' if ok else 'FAIL'}")
    lines.extend(["", "## Per-Repo Distribution", ""])
    for key, count in report["per_repo"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Per-Query-Group Distribution", ""])
    for key, count in report["per_query_group"].items():
        lines.append(f"- {key}: {count}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mining-summary", type=Path, default=Path("datasets/v1.4_real_seeded/raw/mining_report.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.raw)
    summary = _load_json(args.mining_summary)
    report = mining_quality(rows, summary)
    text = markdown(report)
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
