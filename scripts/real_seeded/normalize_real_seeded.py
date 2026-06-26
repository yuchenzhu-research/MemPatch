#!/usr/bin/env python
"""Normalize accepted real-seeded GitHub candidates to MemPatch v1.4 JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.real_seeded.common import (
    PipelineError,
    read_jsonl,
    sanitize_text,
    sha256_text,
    stable_hash,
    utc_now,
    write_jsonl,
)

SCHEMA_VERSION = "mempatch_bench_v1.4"
SPLIT = "real_seeded_challenge"


def operation_to_status(operation: str) -> str:
    return {
        "PRESERVE": "current",
        "REVISE": "outdated",
        "RESTRICT_SCOPE": "out_of_scope",
        "BLOCK": "blocked",
        "MARK_UNRESOLVED": "unresolved",
        "DELETE_OR_FORGET": "deleted",
        "RESTORE_OR_RELEASE": "restored",
        "REJECT_NEW_MEMORY": "should_not_store",
        "NO_WRITE": "should_not_store",
        "ESCALATE": "unresolved",
    }.get(operation, "unresolved")


def operation_to_decision(operation: str, failure_mode: str) -> str:
    if operation == "BLOCK" or failure_mode == "policy_violation":
        return "refuse_due_to_policy"
    if operation in {"MARK_UNRESOLVED", "ESCALATE"}:
        return "mark_unresolved"
    if operation in {"RESTRICT_SCOPE", "REJECT_NEW_MEMORY", "NO_WRITE"}:
        return "ask_clarification" if operation == "RESTRICT_SCOPE" else "mark_unresolved"
    return "use_current_memory"


def scenario_id(candidate: dict[str, Any]) -> str:
    return f"real_seeded_{stable_hash(str(candidate.get('candidate_id')), length=16)}"


def event_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(candidate.get("raw_events") or [], 1):
        rows.append(
            {
                "event_id": f"ev_{index:03d}",
                "timestamp_order": index,
                "timestamp": event.get("timestamp"),
                "source": event.get("url"),
                "actor_role": actor_role(event.get("source_type")),
                "trust_level": trust_level(event.get("source_type")),
                "visibility_scope": "public_github",
                "event_type": event.get("source_type"),
                "content": sanitize_text(event.get("text"), max_chars=1400),
                "related_memory_ids": ["mem_target"],
            }
        )
    return rows


def actor_role(source_type: Any) -> str:
    source = str(source_type or "")
    if source == "maintainer_comment":
        return "maintainer"
    if source in {"pr_merged", "release_note", "changelog", "docs"}:
        return "release_or_docs"
    if source == "pr_body":
        return "contributor"
    return "reporter"


def trust_level(source_type: Any) -> str:
    return "verified" if source_type in {"maintainer_comment", "pr_merged", "release_note", "changelog", "docs"} else "trusted"


def initial_claim(candidate: dict[str, Any]) -> str:
    title = sanitize_text(candidate.get("title"), max_chars=220)
    repo = candidate.get("source_repo")
    return (
        f"For {repo}, retain the initial public GitHub claim until later maintainer, "
        f"merged PR, release, changelog, or docs evidence revises it: {title}"
    )


def public_row(candidate: dict[str, Any]) -> dict[str, Any]:
    sid = scenario_id(candidate)
    repo = candidate.get("source_repo")
    events = event_rows(candidate)
    query = (
        f"Using only the public evidence events, decide how the durable memory for {repo} "
        "should be updated for future answers."
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": sid,
        "split": SPLIT,
        "domain": "software_release",
        "workflow_context": (
            "A persistent software-maintenance agent tracks public GitHub project state. "
            "Earlier issue reports may be superseded by maintainer comments, merged PRs, "
            "release notes, changelogs, docs, or scope constraints."
        ),
        "public_input": {
            "initial_memories": [
                {
                    "memory_id": "mem_target",
                    "content": initial_claim(candidate),
                    "scope": f"public_repo:{repo}",
                    "source_event_ids": [events[0]["event_id"]] if events else [],
                    "memory_type": "project_state",
                    "category": "software_release",
                    "tags": ["real_seeded", "public_github"],
                },
                {
                    "memory_id": "mem_source_policy",
                    "content": (
                        "Treat maintainer comments, merged PRs, release notes, changelogs, and docs "
                        "as higher-authority public evidence than an initial issue or PR body."
                    ),
                    "scope": "benchmark_rule",
                    "memory_type": "source_policy",
                    "category": "software_release",
                    "tags": ["public_evidence"],
                },
            ],
            "events": events,
            "query": query,
        },
        "tasks": {
            "black_box_task": {"prompt": query},
            "memory_state_task": {"prompt": "Return a status for every visible memory."},
            "evidence_retrieval_task": {"prompt": "Return minimal event IDs supporting the memory operation."},
            "diagnostic_task": {"prompt": "Name the primary memory failure mode being tested."},
            "followup_task": {
                "prompt": f"Later, what should the agent remember about this public state for {repo}?"
            },
        },
        "output_contract": {
            "format": "json",
            "required_fields": [
                "answer",
                "decision",
                "memory_operation",
                "memory_state",
                "evidence_event_ids",
                "failure_diagnosis",
                "followup_answer",
            ],
        },
    }


def label_row(candidate: dict[str, Any]) -> dict[str, Any]:
    sid = scenario_id(candidate)
    events = event_rows(candidate)
    failure_mode = (candidate.get("candidate_failure_modes") or ["stale_memory_reuse"])[0]
    operation = (candidate.get("candidate_memory_operations") or ["REVISE"])[0]
    decision = operation_to_decision(operation, failure_mode)
    status = operation_to_status(operation)
    evidence_ids = [
        event["event_id"]
        for event in events
        if event.get("trust_level") == "verified" or event.get("timestamp_order") == 1
    ][:8]
    if len(evidence_ids) < min(3, len(events)):
        evidence_ids = [event["event_id"] for event in events[: min(3, len(events))]]
    source_pointers = [
        {
            "kind": "public_github",
            "url_or_id": url,
            "license_or_terms_note": candidate.get("license_provenance_note"),
        }
        for url in candidate.get("source_urls") or []
    ]
    evidence_hashes = [
        {
            "event_id": event["event_id"],
            "sha256": sha256_text(str(event.get("content") or "")),
        }
        for event in events
    ]
    repo = candidate.get("source_repo")
    answer = (
        f"Update the {repo} memory according to the latest public maintainer, merged PR, "
        "release, changelog, or docs evidence; do not rely only on the initial issue claim."
    )
    followup = f"For {repo}, use the post-evidence memory state and cite the public evidence if asked."
    return {
        "scenario_id": sid,
        "split": SPLIT,
        "difficulty": "challenge",
        "domain": "software_release",
        "failure_mode": failure_mode,
        "pattern": "public_evidence_supersession",
        "expected_decision": decision,
        "expected_memory_operation": operation,
        "expected_memory_states": [
            {"memory_id": "mem_target", "status": status},
            {"memory_id": "mem_source_policy", "status": "current"},
        ],
        "expected_evidence_event_ids": evidence_ids,
        "counterevidence_event_ids": [],
        "expected_failure_diagnosis": failure_mode,
        "expected_answer": answer,
        "expected_followup_answer": followup,
        "expected_followup_answer_key_facts": [str(repo), operation],
        "expected_answer_key_facts": [str(repo), operation],
        "stale_or_wrong_answers": ["Rely only on the initial issue or PR body without later public evidence."],
        "unsafe_reuse_patterns": ["initial issue claim only"],
        "source_pointers": source_pointers,
        "evidence_span_hashes": evidence_hashes,
        "provenance_license_notes": candidate.get("license_provenance_note"),
        "adjudication_notes": candidate.get("adjudication_notes")
        or ["Heuristic local label; requires human audit before treating as final truth."],
        "original_candidate_id": candidate.get("candidate_id"),
        "source_repo": repo,
        "source_urls": candidate.get("source_urls") or [],
        "release_urls": candidate.get("release_urls") or [],
        "doc_urls": candidate.get("doc_urls") or [],
        "normalized_at": utc_now(),
    }


def normalize_rows(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(candidate.get("raw_events") or []) < 3:
            raise PipelineError(f"{candidate.get('candidate_id')}: accepted candidate has fewer than 3 events")
        public_rows.append(public_row(candidate))
        label_rows.append(label_row(candidate))
    return public_rows, label_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        candidates = read_jsonl(args.accepted)
        public_rows, label_rows = normalize_rows(candidates)
        write_jsonl(args.public, public_rows)
        write_jsonl(args.labels, label_rows)
        print(f"Normalized {len(public_rows)} real-seeded cases.")
    except PipelineError as exc:
        print(f"Normalization failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
