#!/usr/bin/env python3
"""Analyze de-identified template signatures for ReTrace data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ID_PATTERNS = (
    (re.compile(r"\brt-(?:train|dev|test|templateheldout)-[a-z0-9-]*\d+\b", re.I), "<SCENARIO_ID>"),
    (re.compile(r"\brb-(?:hard-)?en-\d+\b", re.I), "<SCENARIO_ID>"),
    (re.compile(r"\b(?:C|CASE|TICKET|ORDER|PR|INC|REQ|TASK)-?\d+[A-Z]?\b", re.I), "<CASE_ID>"),
    (re.compile(r"\bPROJ-[A-Z]\d+\b", re.I), "<PROJECT_ID>"),
    (re.compile(r"\b(?:EMP|REF|USR|USER|PERSON|CUST)-\d+\b", re.I), "<PERSON_ID>"),
    (re.compile(r"\bworkspace-[A-Z0-9-]+\b", re.I), "<WORKSPACE_ID>"),
    (re.compile(r"\bm-[a-z0-9-]+(?:-\d+)?\b", re.I), "<MEMORY_ID>"),
    (re.compile(r"\be-[a-z0-9-]+(?:-\d+)?\b", re.I), "<EVENT_ID>"),
    (re.compile(r"\bt-\w[\w-]*\b", re.I), "<TASK_ID>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\b"), "<TIMESTAMP>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<DATE>"),
    (re.compile(r"\b\d{1,2}:\d{2}\b"), "<TIME>"),
    (re.compile(r"\b\d+\b"), "<NUM>"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    for pattern, replacement in ID_PATTERNS:
        text = pattern.sub(replacement.lower(), text)
    text = re.sub(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", "<ordinal>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def event_text_templates(scenario: dict[str, Any]) -> list[str]:
    return [normalize_text(event.get("text", "")) for event in scenario.get("public_input", {}).get("event_trace", [])]


def workflow_context_template(scenario: dict[str, Any]) -> str:
    return normalize_text(scenario.get("workflow_context", ""))


def scenario_signature_payload(scenario: dict[str, Any]) -> dict[str, Any]:
    public = scenario.get("public_input", {})
    return {
        "workflow_context": workflow_context_template(scenario),
        "event_trace": [
            {
                "source": normalize_text(event.get("source", "")),
                "actor": normalize_text(event.get("actor", "")),
                "event_type": normalize_text(event.get("event_type", "")),
                "trust_level": normalize_text(event.get("trust_level", "")),
                "visibility_scope": normalize_text(event.get("visibility_scope", "")),
                "text": normalize_text(event.get("text", "")),
                "related_count": len(event.get("related_memory_ids", [])),
            }
            for event in public.get("event_trace", [])
        ],
        "initial_memory": [
            {
                "text": normalize_text(memory.get("text", "")),
                "scope": normalize_text(memory.get("visibility_scope", "")),
                "is_distractor": bool(memory.get("is_distractor")),
            }
            for memory in public.get("initial_memory", [])
        ],
        "task_prompts": [normalize_text(task.get("prompt", "")) for task in scenario.get("tasks", [])],
    }


def scenario_signature(scenario: dict[str, Any]) -> str:
    payload = json.dumps(scenario_signature_payload(scenario), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def signature_examples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for row in rows:
        sig = scenario_signature(row)
        examples.setdefault(
            sig,
            {
                "scenario_id": row.get("scenario_id"),
                "domain": row.get("domain"),
                "primary_failure_mode": row.get("primary_failure_mode"),
                "expected_decision": row.get("hidden_gold", {}).get("expected_decision"),
                "signature_payload": scenario_signature_payload(row),
            },
        )
    return examples


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": len(rows),
        "event_text_templates": len({template for row in rows for template in event_text_templates(row)}),
        "workflow_context_templates": len({workflow_context_template(row) for row in rows}),
        "scenario_signatures": len({scenario_signature(row) for row in rows}),
    }


def _load_optional(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if p.is_dir():
        p = p / "scenarios.jsonl"
    return read_jsonl(p)


def _pct(numer: int, denom: int) -> float:
    return 0.0 if denom == 0 else numer / denom * 100.0


def render_report(
    *,
    train_path: str,
    dev_path: str | None,
    test_path: str,
    prototype_test_path: str | None,
    out_path: Path,
) -> None:
    train = _load_optional(train_path)
    dev = _load_optional(dev_path)
    test = _load_optional(test_path)
    prototype = _load_optional(prototype_test_path)

    train_sigs = {scenario_signature(row) for row in train}
    dev_sigs = {scenario_signature(row) for row in dev}
    test_sigs = {scenario_signature(row) for row in test}
    prototype_sigs = {scenario_signature(row) for row in prototype}

    train_test = train_sigs & test_sigs
    dev_test = dev_sigs & test_sigs
    train_prototype = train_sigs & prototype_sigs
    dev_prototype = dev_sigs & prototype_sigs

    test_examples = signature_examples(test)
    proto_examples = signature_examples(prototype)

    lines = [
        "# ReTrace-Bench Template Signature Report",
        "",
        "This diagnostic de-identifies scenario text before comparing templates. It normalizes scenario IDs, case/project/person/workspace IDs, memory/event IDs, timestamps, numeric counters, and split prefixes.",
        "",
        "The existing `test_800_en` is treated as prototype/diagnostic. The new `test_800_templateheldout_en` is the candidate paper-facing held-out split.",
        "",
        "## Split Summary",
        "",
        "| split | scenarios | event-text templates | workflow-context templates | scenario signatures |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, rows in (("train", train), ("dev", dev), ("test", test), ("prototype_test", prototype)):
        if not rows:
            continue
        stats = split_stats(rows)
        lines.append(
            f"| {name} | {stats['scenarios']} | {stats['event_text_templates']} | "
            f"{stats['workflow_context_templates']} | {stats['scenario_signatures']} |"
        )

    lines.extend(
        [
            "",
            "## Signature Overlap",
            "",
            "| comparison | overlap count | percent of test signatures |",
            "| --- | ---: | ---: |",
            f"| train∩test | {len(train_test)} | {_pct(len(train_test), len(test_sigs)):.2f}% |",
            f"| dev∩test | {len(dev_test)} | {_pct(len(dev_test), len(test_sigs)):.2f}% |",
        ]
    )
    if prototype:
        lines.extend(
            [
                f"| train∩prototype_test | {len(train_prototype)} | {_pct(len(train_prototype), len(prototype_sigs)):.2f}% |",
                f"| dev∩prototype_test | {len(dev_prototype)} | {_pct(len(dev_prototype), len(prototype_sigs)):.2f}% |",
            ]
        )

    def add_examples(title: str, overlap: set[str], examples: dict[str, dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", ""])
        if not overlap:
            lines.append("No overlapping scenario signatures found.")
            return
        for sig in sorted(overlap)[:10]:
            ex = examples.get(sig, {})
            lines.append(
                f"- `{sig}`: `{ex.get('scenario_id')}` / `{ex.get('domain')}` / "
                f"`{ex.get('primary_failure_mode')}` / `{ex.get('expected_decision')}`"
            )

    add_examples("Examples: train∩test", train_test, test_examples)
    add_examples("Examples: dev∩test", dev_test, test_examples)
    if prototype:
        add_examples("Examples: train∩prototype_test", train_prototype, proto_examples)
        add_examples("Examples: dev∩prototype_test", dev_prototype, proto_examples)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev")
    parser.add_argument("--test", required=True)
    parser.add_argument("--prototype-test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    render_report(
        train_path=args.train,
        dev_path=args.dev,
        test_path=args.test,
        prototype_test_path=args.prototype_test,
        out_path=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
