#!/usr/bin/env python3
"""Prepare MemPatch SFT chat JSONL from main/hard scenario splits."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.mempatch_bench.general_taxonomy import DECISIONS, canonical_hidden_gold_fields
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

SYSTEM_PROMPT_DECISION_BALANCED = """You are MemPatch Revision Policy.
Return only one strict JSON object.
Choose decision carefully; do not default to use_current_memory.
Use ask_clarification when evidence is insufficient or ambiguous.
Use mark_unresolved when the memory cannot be safely updated.
Use escalate when policy or authority requires human review.
Use refuse_due_to_policy only for policy-blocked cases.
evidence_event_ids must be exact event_id strings from event_trace.
answer must be one short sentence.
Do not explain."""

DEFAULT_DECISION_QUOTAS: dict[str, int] = {
    "use_current_memory": 400,
    "mark_unresolved": 400,
    "refuse_due_to_policy": 300,
    "ask_clarification": 300,
    "escalate": 300,
}

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

TARGET_STYLES = ("default", "evidence_compact", "decision_balanced")
ORDERED_RESPONSE_STYLES = frozenset({"evidence_compact", "decision_balanced"})


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


def scenario_decision(scenario: dict[str, Any]) -> str | None:
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    return gold.get("expected_decision")


def index_main_by_decision(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {decision: [] for decision in DECISIONS}
    for row in rows:
        decision = scenario_decision(row)
        if decision in buckets:
            buckets[decision].append(row)
    return buckets


def validate_decision_quotas(
    buckets: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    *,
    split_name: str,
) -> list[str]:
    errors: list[str] = []
    for decision, need in quotas.items():
        if decision not in DECISIONS:
            errors.append(f"unknown decision quota {decision!r}")
            continue
        have = len(buckets.get(decision, []))
        if need > have:
            errors.append(
                f"{split_name}: {decision} quota {need} exceeds available {have}"
            )
    return errors


def sample_by_decision_quotas(
    buckets: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for decision in DECISIONS:
        need = quotas.get(decision, 0)
        if need <= 0:
            continue
        pool = list(buckets[decision])
        rng.shuffle(pool)
        if len(pool) < need:
            raise ValueError(f"{decision}: need {need}, have {len(pool)}")
        sampled.extend(pool[:need])
    rng.shuffle(sampled)
    return sampled


def parse_decision_quotas(raw: str | None) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_DECISION_QUOTAS)
    quotas = dict(DEFAULT_DECISION_QUOTAS)
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        decision, count = part.split("=", 1)
        quotas[decision.strip()] = int(count.strip())
    return quotas


def gold_to_response(scenario: dict[str, Any], *, target_style: str = "default") -> dict[str, Any]:
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    answer = gold["expected_answer"] or ""
    if target_style in ORDERED_RESPONSE_STYLES:
        answer = compact_answer(answer)
    response = {
        "decision": gold["expected_decision"],
        "memory_state": gold["expected_memory_state"],
        "evidence_event_ids": gold["expected_evidence_event_ids"],
        "failure_diagnosis": gold["expected_failure_diagnosis"],
        "answer": answer,
    }
    if target_style in ORDERED_RESPONSE_STYLES:
        return {
            "decision": response["decision"],
            "memory_state": response["memory_state"],
            "evidence_event_ids": response["evidence_event_ids"],
            "failure_diagnosis": response["failure_diagnosis"],
            "answer": response["answer"],
        }
    return response


def system_prompt_for_style(target_style: str, evidence_event_ids: list[str]) -> str:
    if target_style == "decision_balanced":
        return SYSTEM_PROMPT_DECISION_BALANCED
    if target_style == "evidence_compact":
        prompt = SYSTEM_PROMPT_EVIDENCE_COMPACT
        if evidence_event_ids:
            prompt += "\nevidence_event_ids must not be empty for this scenario."
        return prompt
    return SYSTEM_PROMPT


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


def decision_distribution_from_sft_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        assistant = next(
            (message["content"] for message in row["messages"] if message["role"] == "assistant"),
            "{}",
        )
        payload = json.loads(assistant)
        counts[payload.get("decision", "<missing>")] += 1
    return counts


def print_decision_distribution(label: str, rows: list[dict[str, Any]]) -> None:
    counts = decision_distribution_from_sft_rows(rows)
    print(f"{label} decision distribution (n={len(rows)}):")
    for decision in DECISIONS:
        if counts.get(decision, 0):
            print(f"  {decision}: {counts[decision]}")
    for decision, count in sorted(counts.items()):
        if decision not in DECISIONS:
            print(f"  {decision}: {count}")


def print_main_inventory(buckets: dict[str, list[dict[str, Any]]]) -> None:
    print("main split hidden_gold.expected_decision inventory:")
    for decision in DECISIONS:
        print(f"  {decision}: {len(buckets[decision])}")
    renderer = Counter(
        row.get("metadata", {}).get("renderer")
        for rows in buckets.values()
        for row in rows
    )
    if renderer:
        print(f"  renderers: {dict(renderer)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MemPatch SFT JSONL for smoke training.")
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="optional v1.2 train split scenarios.jsonl (preferred over --main for SFT)",
    )
    parser.add_argument(
        "--main",
        type=Path,
        default=Path("hf_release/mempatch_v1_1/main/scenarios.jsonl"),
        help="main split scenarios.jsonl (used when --train is absent)",
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
        help="SFT target style; decision_balanced samples main by decision quotas",
    )
    parser.add_argument(
        "--decision-quotas",
        default=None,
        help=(
            "Comma-separated decision=count quotas for decision_balanced, e.g. "
            "'use_current_memory=400,mark_unresolved=400'"
        ),
    )
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_path = args.train
    main_path = args.main
    if train_path is not None and train_path.is_file():
        source_path = train_path
        source_label = "train"
    elif main_path.is_file():
        source_path = main_path
        source_label = "main"
    else:
        if train_path is not None:
            print(f"error: train scenarios not found: {train_path}", file=sys.stderr)
        print(f"error: main scenarios not found: {main_path}", file=sys.stderr)
        return 1
    if not args.hard.is_file():
        print(f"error: hard scenarios not found: {args.hard}", file=sys.stderr)
        return 1

    main_rows = read_jsonl(source_path)
    hard_rows = read_jsonl(args.hard)
    if source_label == "train":
        print(f"SFT source: train split ({len(main_rows)} rows) from {source_path}")

    if args.hard_probe_size > len(hard_rows):
        print(
            f"error: hard split has {len(hard_rows)} rows, need at least {args.hard_probe_size}",
            file=sys.stderr,
        )
        return 1

    probe_rows = [hard_probe_row(s) for s in hard_rows[: args.hard_probe_size]]

    if args.target_style == "decision_balanced":
        quotas = parse_decision_quotas(args.decision_quotas)
        buckets = index_main_by_decision(main_rows)
        print_main_inventory(buckets)
        quota_errors = validate_decision_quotas(buckets, quotas, split_name="main")
        if quota_errors:
            print(
                "error: decision_balanced quotas cannot be met from main split alone",
                file=sys.stderr,
            )
            print(
                "note: main_final_renderer does not emit ask_clarification or escalate; "
                "those labels appear only in hard_final_renderer (hard split).",
                file=sys.stderr,
            )
            print(
                "note: hard split hidden_gold must not be copied into train data.",
                file=sys.stderr,
            )
            for err in quota_errors:
                print(f"  - {err}", file=sys.stderr)
            return 1

        total_needed = sum(quotas.values())
        if args.valid_size >= total_needed:
            print(
                f"error: valid-size {args.valid_size} must be smaller than "
                f"sampled total {total_needed}",
                file=sys.stderr,
            )
            return 1

        sampled = sample_by_decision_quotas(buckets, quotas, seed=args.seed)
        valid_rows_raw = sampled[: args.valid_size]
        train_rows_raw = sampled[args.valid_size :]
        train_rows = [sft_example(s, target_style="decision_balanced") for s in train_rows_raw]
        valid_rows = [sft_example(s, target_style="decision_balanced") for s in valid_rows_raw]
    else:
        need_main = args.train_size + args.valid_size
        if len(main_rows) < need_main:
            print(
                f"error: main split has {len(main_rows)} rows, need at least {need_main}",
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

    out_dir = args.out_dir
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "hard_probe.jsonl", probe_rows)

    print(
        f"Wrote {len(train_rows)} train, {len(valid_rows)} valid, "
        f"{len(probe_rows)} hard_probe rows to {out_dir} "
        f"(target_style={args.target_style})"
    )
    print_decision_distribution("train", train_rows)
    print_decision_distribution("valid", valid_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
