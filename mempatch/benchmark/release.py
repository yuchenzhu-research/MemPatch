"""Export raw internal scenarios to the v1.4 public/label release layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mempatch.benchmark.contracts import state_map_to_list
from mempatch.benchmark.leakage import audit_public_rows, sanitize_public_value

TASK_KEYS = (
    "black_box_task",
    "memory_state_task",
    "evidence_retrieval_task",
    "diagnostic_task",
)

INTERNAL_KEYS = {
    "canonical_failure_mode",
    "decision_triggers",
    "difficulty",
    "difficulty_level",
    "failure_mode",
    "hidden_gold",
    "is_distractor",
    "pattern",
    "pattern_trap_type",
    "primary_failure_mode",
    "resolver_trace",
    "source_pointers",
    "template_family_id",
    "template_instance_id",
    "validation_notes",
}

PUBLIC_MEMORY_KEYS = (
    "memory_id",
    "content",
    "scope",
    "source_event_ids",
    "memory_type",
    "user_id",
    "session_id",
    "owner_id",
    "created_at",
    "category",
    "tags",
)

PUBLIC_EVENT_KEYS = (
    "event_id",
    "content",
    "timestamp_order",
    "timestamp",
    "source",
    "actor_role",
    "trust_level",
    "visibility_scope",
    "event_type",
    "related_memory_ids",
    "user_id",
    "session_id",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def strip_internal(value: Any) -> Any:
    return sanitize_public_value(value)


def compact(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None and item != [] and item != {}}


def public_memory(item: dict[str, Any]) -> dict[str, Any]:
    row = {
        "memory_id": str(item.get("memory_id") or item.get("id") or ""),
        "content": str(item.get("content") or item.get("text") or ""),
        "scope": item.get("scope"),
        "source_event_ids": [str(x) for x in item.get("source_event_ids") or []],
        "memory_type": item.get("memory_type") or item.get("type"),
        "user_id": item.get("user_id"),
        "session_id": item.get("session_id"),
        "owner_id": item.get("owner_id"),
        "created_at": item.get("created_at"),
        "category": item.get("category"),
        "tags": strip_internal(item.get("tags") or []),
    }
    cleaned = compact(row)
    return {key: cleaned[key] for key in PUBLIC_MEMORY_KEYS if key in cleaned}


def public_event(item: dict[str, Any]) -> dict[str, Any]:
    row = {
        "event_id": str(item.get("event_id") or item.get("id") or ""),
        "content": str(item.get("content") or item.get("text") or ""),
        "timestamp_order": item.get("timestamp_order"),
        "timestamp": item.get("timestamp"),
        "source": item.get("source") or item.get("actor_role"),
        "actor_role": item.get("actor_role"),
        "trust_level": item.get("trust_level"),
        "visibility_scope": item.get("visibility_scope"),
        "event_type": item.get("event_type"),
        "related_memory_ids": [str(x) for x in item.get("related_memory_ids") or []],
        "user_id": item.get("user_id"),
        "session_id": item.get("session_id"),
    }
    cleaned = compact(row)
    return {key: cleaned[key] for key in PUBLIC_EVENT_KEYS if key in cleaned}


def public_row(raw: dict[str, Any]) -> dict[str, Any]:
    public_input = raw.get("public_input") or {}
    memories = public_input.get("initial_memories", public_input.get("initial_memory", [])) or []
    events = public_input.get("events", public_input.get("event_trace", [])) or []
    tasks = {key: strip_internal(raw[key]) for key in TASK_KEYS if key in raw}
    return {
        "schema_version": "mempatch_bench_v1.4",
        "scenario_id": str(raw["scenario_id"]),
        "split": raw.get("public_split_name") or raw.get("split"),
        "domain": raw.get("domain"),
        "workflow_context": strip_internal(raw.get("workflow_context", "")),
        "public_input": {
            "initial_memories": [
                public_memory(strip_internal(item))
                for item in memories
                if isinstance(item, dict)
            ],
            "events": [
                public_event(strip_internal(item))
                for item in events
                if isinstance(item, dict)
            ],
        },
        "tasks": tasks,
        "output_contract": {
            "format": "json",
            "required_fields": [
                "answer",
                "decision",
                "memory_state",
                "evidence_event_ids",
                "failure_diagnosis",
            ],
        },
    }


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def split_stats(public_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts: dict[str, int] = {}
    memory_counts: dict[str, int] = {}
    for row in public_rows:
        public_input = row.get("public_input") or {}
        event_count = str(len(public_input.get("events") or []))
        memory_count = str(len(public_input.get("initial_memories") or []))
        event_counts[event_count] = event_counts.get(event_count, 0) + 1
        memory_counts[memory_count] = memory_counts.get(memory_count, 0) + 1
    return {
        "domains": _count(label_rows, "domain"),
        "difficulties": _count(label_rows, "difficulty"),
        "failure_modes": _count(label_rows, "failure_mode"),
        "patterns": _count(label_rows, "pattern"),
        "event_count_histogram": dict(sorted(event_counts.items())),
        "memory_count_histogram": dict(sorted(memory_counts.items())),
    }


def label_row(raw: dict[str, Any], split: str) -> dict[str, Any]:
    gold = raw.get("hidden_gold") or {}
    metadata = raw.get("metadata") or {}
    rubric = gold.get("rubric") or {}
    return {
        "scenario_id": str(raw["scenario_id"]),
        "split": split,
        "difficulty": raw.get("difficulty") or raw.get("difficulty_level"),
        "domain": raw.get("domain"),
        "failure_mode": raw.get("primary_failure_mode") or raw.get("failure_mode") or gold.get("expected_failure_diagnosis"),
        "pattern": raw.get("pattern") or metadata.get("pattern"),
        "expected_decision": gold.get("expected_decision"),
        "expected_memory_states": state_map_to_list(gold.get("expected_memory_state") or gold.get("expected_memory_states")),
        "expected_evidence_event_ids": [str(x) for x in gold.get("expected_evidence_event_ids") or []],
        "counterevidence_event_ids": [str(x) for x in gold.get("counterevidence_event_ids") or []],
        "expected_failure_diagnosis": gold.get("expected_failure_diagnosis"),
        "expected_answer": gold.get("expected_answer"),
        "expected_answer_key_facts": [str(x) for x in rubric.get("must_include") or []],
        "stale_or_wrong_answers": [str(x) for x in gold.get("stale_or_wrong_answers") or []],
        "rubric": rubric,
        "resolver_trace": metadata.get("resolver_trace"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_release(split_paths: dict[str, Path], output_dir: Path, version: str = "v1.4.0-dev") -> dict[str, Any]:
    public_dir = output_dir / "public"
    labels_dir = output_dir / "labels"
    manifest_dir = output_dir / "manifests"
    split_manifest: dict[str, Any] = {}
    all_public: list[dict[str, Any]] = []
    split_outputs: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for split, path in split_paths.items():
        raw_rows = read_jsonl(path)
        public_rows = [public_row(row) for row in raw_rows]
        label_rows = [label_row(row, split) for row in raw_rows]
        split_outputs[split] = (public_rows, label_rows)
        all_public.extend(public_rows)
        split_manifest[split] = {
            "num_examples": len(raw_rows),
            "public_path": f"public/{split}.jsonl",
            "labels_path": f"labels/{split}.labels.jsonl",
            "stats": split_stats(public_rows, label_rows),
        }
    audit = {"public_forbidden_field_violations": audit_public_rows(all_public)}
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "audit_report.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit["public_forbidden_field_violations"]:
        raise ValueError(
            "public release leakage audit failed: "
            f"{audit['public_forbidden_field_violations'][:5]}"
        )
    for split, (public_rows, label_rows) in split_outputs.items():
        write_jsonl(public_dir / f"{split}.jsonl", public_rows)
        write_jsonl(labels_dir / f"{split}.labels.jsonl", label_rows)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "dataset_name": "MemPatch-Bench",
        "release_version": version,
        "schema_version": "mempatch_bench_v1.4",
        "splits": split_manifest,
        "audit_summary": {
            "public_forbidden_field_violation_count": len(audit["public_forbidden_field_violations"]),
        },
        "checksums": {path.relative_to(output_dir).as_posix(): sha256_file(path) for path in files},
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
