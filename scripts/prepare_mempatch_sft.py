#!/usr/bin/env python3
"""Prepare MemPatch SFT chat JSONL from main/hard scenario splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.mempatch_bench.general_taxonomy import canonical_hidden_gold_fields
from benchmark.mempatch_bench.model_runner import build_prompt
from benchmark.mempatch_bench.public_view import public_scenario_view

SYSTEM_PROMPT = "You are MemPatch Revision Policy. Return only strict JSON."

SYSTEM_PROMPT_EVIDENCE_COMPACT = """You are MemPatch Revision Policy.
Return only one strict JSON object.
evidence_event_ids must be a JSON array of exact event_id strings from event_trace.
If supporting evidence exists, evidence_event_ids must not be empty.
Do not include counterevidence unless it supports the final decision.
answer must be one short sentence.
Do not explain."""

LEAKAGE_MARKERS = (
    "hidden_gold",
    "expected_decision",
    "expected_answer",
    "expected_memory_state",
    "expected_failure_diagnosis",
    "expected_evidence_event_ids",
    "counterevidence_event_ids",
    "validation_notes",
    "source_pointers",
    "primary_failure_mode",
    "pattern_trap_type",
    "canonical_failure_mode",
)

TARGET_STYLES = ("default", "evidence_compact")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def compact_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    match = re.search(r"[.!?](?:\s|$)", text)
    if match:
        return text[: match.end()].strip()
    return text


def gold_to_response(scenario: dict[str, Any], *, target_style: str = "default") -> dict[str, Any]:
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    answer = gold["expected_answer"] or ""
    if target_style == "evidence_compact":
        answer = compact_answer(answer)
    response = {
        "decision": gold["expected_decision"],
        "memory_state": gold["expected_memory_state"],
        "evidence_event_ids": gold["expected_evidence_event_ids"],
        "failure_diagnosis": gold["expected_failure_diagnosis"],
        "answer": answer,
    }
    if target_style == "evidence_compact":
        return {
            "decision": response["decision"],
            "memory_state": response["memory_state"],
            "evidence_event_ids": response["evidence_event_ids"],
            "failure_diagnosis": response["failure_diagnosis"],
            "answer": response["answer"],
        }
    return response


def system_prompt_for_style(target_style: str, evidence_event_ids: list[str]) -> str:
    if target_style != "evidence_compact":
        return SYSTEM_PROMPT
    prompt = SYSTEM_PROMPT_EVIDENCE_COMPACT
    if evidence_event_ids:
        prompt += "\nevidence_event_ids must not be empty for this scenario."
    return prompt


def assistant_content(response: dict[str, Any]) -> str:
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def assert_no_leakage(user_content: str, *, scenario_id: str) -> None:
    lowered = user_content.lower()
    for marker in LEAKAGE_MARKERS:
        if marker in lowered:
            raise ValueError(f"{scenario_id}: user content leaks {marker!r}")


def sft_example(scenario: dict[str, Any], *, target_style: str = "default") -> dict[str, Any]:
    view = public_scenario_view(scenario)
    user_content = build_prompt(view)
    assert_no_leakage(user_content, scenario_id=str(scenario["scenario_id"]))
    response = gold_to_response(scenario, target_style=target_style)
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt_for_style(
                    target_style,
                    response["evidence_event_ids"],
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content(response)},
        ]
    }


def hard_probe_row(scenario: dict[str, Any]) -> dict[str, Any]:
    view = public_scenario_view(scenario)
    return {
        "scenario_id": scenario["scenario_id"],
        "public_input": view["public_input"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MemPatch SFT JSONL for smoke training.")
    parser.add_argument(
        "--main",
        type=Path,
        default=Path("hf_release/mempatch_v1_1/main/scenarios.jsonl"),
        help="main split scenarios.jsonl",
    )
    parser.add_argument(
        "--hard",
        type=Path,
        default=Path("hf_release/mempatch_v1_1/hard/scenarios.jsonl"),
        help="hard split scenarios.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("local/train_data/mempatch_qwen14b_smoke"),
        help="Output directory for train/valid/hard_probe JSONL",
    )
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--valid-size", type=int, default=64)
    parser.add_argument("--hard-probe-size", type=int, default=50)
    parser.add_argument(
        "--target-style",
        choices=TARGET_STYLES,
        default="default",
        help="SFT target style; evidence_compact emphasizes evidence_event_ids ordering",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.main.is_file():
        print(f"error: main scenarios not found: {args.main}", file=sys.stderr)
        return 1
    if not args.hard.is_file():
        print(f"error: hard scenarios not found: {args.hard}", file=sys.stderr)
        return 1

    main_rows = read_jsonl(args.main)
    hard_rows = read_jsonl(args.hard)

    need_main = args.train_size + args.valid_size
    if len(main_rows) < need_main:
        print(
            f"error: main split has {len(main_rows)} rows, need at least {need_main}",
            file=sys.stderr,
        )
        return 1
    if len(hard_rows) < args.hard_probe_size:
        print(
            f"error: hard split has {len(hard_rows)} rows, need at least {args.hard_probe_size}",
            file=sys.stderr,
        )
        return 1

    train_rows = [
        sft_example(s, target_style=args.target_style) for s in main_rows[: args.train_size]
    ]
    valid_rows = [
        sft_example(s, target_style=args.target_style)
        for s in main_rows[args.train_size : need_main]
    ]
    probe_rows = [hard_probe_row(s) for s in hard_rows[: args.hard_probe_size]]

    out_dir = args.out_dir
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "hard_probe.jsonl", probe_rows)

    print(
        f"Wrote {len(train_rows)} train, {len(valid_rows)} valid, "
        f"{len(probe_rows)} hard_probe rows to {out_dir} "
        f"(target_style={args.target_style})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
