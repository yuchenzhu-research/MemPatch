#!/usr/bin/env python
"""Filter provisional GitHub candidates into high-quality real-seeded cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.real_seeded.common import (
    AUTHORITATIVE_EVENT_TYPES,
    PipelineError,
    has_sensitive_content,
    infer_failure_modes,
    infer_operations,
    public_github_url,
    read_jsonl,
    sanitize_obj,
    sensitive_findings,
    unique_preserve_order,
    utc_now,
    write_jsonl,
)

FEATURE_LABELS = {"enhancement", "feature", "feature request", "proposal", "question"}
SUBJECTIVE_TERMS = ("should we", "what do you think", "proposal", "brainstorm", "maybe")
REPRO_TERMS = ("steps to reproduce", "minimal reproducer", "reproduction", "stack trace")


def event_priority(event: dict[str, Any]) -> tuple[int, str]:
    source_type = str(event.get("source_type") or "")
    order = {
        "issue_body": 0,
        "pr_body": 0,
        "maintainer_comment": 1,
        "pr_merged": 2,
        "release_note": 3,
        "changelog": 4,
        "docs": 4,
        "issue_comment": 5,
    }
    return (order.get(source_type, 9), str(event.get("timestamp") or ""))


def compress_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(events) <= 8:
        return events
    first = [event for event in events if event.get("source_type") in {"issue_body", "pr_body"}][:1]
    authoritative = [
        event for event in events if event.get("source_type") in AUTHORITATIVE_EVENT_TYPES
    ][:5]
    remainder = [event for event in events if event not in first and event not in authoritative]
    selected = unique_preserve_order(first + authoritative + remainder)[:8]
    return sorted(selected, key=lambda ev: str(ev.get("timestamp") or "9999"))


def rejection_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    events = candidate.get("raw_events") or []
    if len(events) < 2:
        reasons.append("only one issue/body event exists")
    if len(events) < 3:
        reasons.append("fewer than three public evidence events")
    authoritative = [event for event in events if event.get("source_type") in AUTHORITATIVE_EVENT_TYPES]
    if not authoritative:
        reasons.append("no maintainer, merged PR, release note, changelog, or docs evidence")
    if candidate.get("state") == "open" and not any(
        event.get("source_type") in {"pr_merged", "release_note", "changelog", "docs"} for event in events
    ):
        reasons.append("open issue/PR with unclear final state")
    labels = {str(label).lower() for label in candidate.get("labels") or []}
    text = " ".join([str(candidate.get("title") or ""), *(str(ev.get("text") or "") for ev in events)]).lower()
    if labels & FEATURE_LABELS and not any(term in text for term in ("deprecated", "fixed in", "removed", "reverted")):
        reasons.append("feature request without final memory transition")
    if not candidate.get("candidate_failure_modes"):
        inferred = infer_failure_modes([str((candidate.get("retrieval_query_terms") or {}).get("group") or "")], text)
        if not inferred:
            reasons.append("does not map to a MemPatch failure mode")
    if not candidate.get("candidate_memory_operations"):
        inferred_ops = infer_operations([str((candidate.get("retrieval_query_terms") or {}).get("group") or "")], text)
        if not inferred_ops:
            reasons.append("does not map to a memory operation")
    if any(term in text for term in SUBJECTIVE_TERMS):
        reasons.append("needs subjective interpretation")
    if any(term in text for term in REPRO_TERMS) and not authoritative:
        reasons.append("requires reproducing/debugging a bug rather than public state revision")
    if has_sensitive_content(candidate):
        kinds = sorted({finding["kind"] for finding in sensitive_findings(candidate)})
        reasons.append("contains sensitive content: " + ", ".join(kinds))
    source_urls = candidate.get("source_urls") or []
    if not source_urls or any(not public_github_url(url) for url in source_urls):
        reasons.append("missing public GitHub source URLs")
    if len(compress_events(events)) < 3:
        reasons.append("cannot compress into at least three public evidence events")
    if len(compress_events(events)) > 8:
        reasons.append("cannot compress into eight or fewer evidence events")
    return sorted(dict.fromkeys(reasons))


def accepted_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    events = compress_events(candidate.get("raw_events") or [])
    text = " ".join(str(ev.get("text") or "") for ev in events)
    group = str((candidate.get("retrieval_query_terms") or {}).get("group") or "")
    failure_modes = list(candidate.get("candidate_failure_modes") or infer_failure_modes([group], text))[:2]
    operations = list(candidate.get("candidate_memory_operations") or infer_operations([group], text))[:2]
    return sanitize_obj(
        {
            **candidate,
            "raw_events": events,
            "candidate_failure_modes": failure_modes,
            "candidate_memory_operations": operations,
            "accepted_at": utc_now(),
            "adjudication_notes": [
                "Passed automated structural screening; manual acceptance review is required before release.",
                "Structural screening covers required fields, enums, public URLs, duplicates, and sensitive-content patterns.",
            ],
        }
    )


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str]] = set()
    for row in rows:
        source_urls = row.get("source_urls") or []
        source_key = (str(row.get("source_repo")), str(source_urls[0] if source_urls else row.get("candidate_id")))
        if source_key in seen_sources:
            rejected.append({**sanitize_obj(row), "rejection_reasons": ["duplicate source candidate"]})
            continue
        seen_sources.add(source_key)
        reasons = rejection_reasons(row)
        if reasons:
            rejected.append({**sanitize_obj(row), "rejection_reasons": reasons})
            continue
        accepted.append(accepted_candidate(row))
    return accepted, rejected


def write_summary(accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    reason_counter: Counter[str] = Counter()
    for row in rejected:
        reason_counter.update(row.get("rejection_reasons") or [])
    summary = {
        "accepted_candidates": len(accepted),
        "rejected_candidates": len(rejected),
        "top_rejection_reasons": dict(reason_counter.most_common(10)),
        "accepted_failure_modes": Counter(
            mode for row in accepted for mode in row.get("candidate_failure_modes") or []
        ),
        "accepted_memory_operations": Counter(
            op for row in accepted for op in row.get("candidate_memory_operations") or []
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_jsonl(args.input)
        accepted, rejected = filter_rows(rows)
        write_jsonl(args.accepted, accepted)
        write_jsonl(args.rejected, rejected)
        write_summary(accepted, rejected)
    except PipelineError as exc:
        print(f"Filtering failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
