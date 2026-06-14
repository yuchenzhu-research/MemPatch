#!/usr/bin/env python3
"""Prepare MemPatch v1.3 multitask SFT JSONL (Path B response + Path A typed actions).

Default (smoke): quota-sampled train/val from the train split plus a balanced sample from test.
``--full-train``: fixed stratified 80/20 partition within train3500 (Linux paper protocol).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import REPO_ROOT, bootstrap_from

bootstrap_from(__file__)

from benchmark.general_taxonomy import (
    DECISIONS,
    PRIMARY_FAILURE_MODES,
    PRIMARY_MEMORY_STATUSES,
    canonical_hidden_gold_fields,
)
from benchmark.model_runner import build_prompt
from benchmark.public_view import public_scenario_view
from mempatch.revision.runtime.learned_proposer import actions_to_json, build_proposer_prompt
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view
from mempatch.revision.schemas import RevisionAction

SYSTEM_PROMPT_V13_SMOKE = """You are MemPatch Revision Policy.
Return only one strict JSON object.
Apply this decision tree in order (first match wins):
1) refuse_due_to_policy — the requested write/action is explicitly forbidden by policy events.
2) escalate — human review, compliance, or authority gate blocks automatic revision (not mere ambiguity).
3) ask_clarification — required target/scope/version/workspace/slot is missing from the user request.
4) mark_unresolved — verified in-scope evidence conflicts and cannot be merged without new facts.
5) use_current_memory — sufficient verified evidence supports answering from current memory.
Do not default to use_current_memory when steps 1–4 apply.
evidence_event_ids must be exact event_id strings from event_trace; cite only the minimal set that supports the chosen decision and memory_state. Do not cite distractor, background, or out-of-scope events.
failure_diagnosis must be exactly one enum string from required_output_schema; never use empty string, empty list, "none", or null.
answer must be one short sentence.
Do not explain."""

TRAIN_QUOTAS: dict[str, int] = {
    "use_current_memory": 100,
    "mark_unresolved": 100,
    "ask_clarification": 100,
    "escalate": 100,
    "refuse_due_to_policy": 100,
}

TEST_BALANCED_QUOTAS: dict[str, int] = {d: 10 for d in DECISIONS}
DEFAULT_SPLIT_PARTS = 5

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

def fixed_stratified_split(
    rows: list[dict[str, Any]],
    *,
    split_index: int,
    split_parts: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic 80/20-style partition within one split (not cross-validation).

    Each decision bucket is shuffled with ``seed``, then every ``split_parts``-th
    row (starting at ``split_index``) is held out for val loss only.
    """
    if not 0 <= split_index < split_parts:
        raise ValueError(f"split_index must be in [0, {split_parts}), got {split_index}")
    rng = random.Random(seed)
    train_part: list[dict[str, Any]] = []
    valid_part: list[dict[str, Any]] = []
    buckets = index_by_decision(rows)
    for decision in DECISIONS:
        pool = list(buckets.get(decision, []))
        rng.shuffle(pool)
        for i, row in enumerate(pool):
            if i % split_parts == split_index:
                valid_part.append(row)
            else:
                train_part.append(row)
    rng.shuffle(train_part)
    rng.shuffle(valid_part)
    return train_part, valid_part


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


def scenario_decision(scenario: dict[str, Any]) -> str | None:
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    return gold.get("expected_decision")


def index_by_decision(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {decision: [] for decision in DECISIONS}
    for row in rows:
        decision = scenario_decision(row)
        if decision in buckets:
            buckets[decision].append(row)
    return buckets


def sample_quotas(
    buckets: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    *,
    seed: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    actual: dict[str, int] = {}
    for decision in DECISIONS:
        need = quotas.get(decision, 0)
        if need <= 0:
            continue
        pool = list(buckets.get(decision, []))
        rng.shuffle(pool)
        take = min(need, len(pool))
        actual[decision] = take
        if take < need:
            print(
                f"warning: {split_name} {decision} requested {need}, only {take} available",
                file=sys.stderr,
            )
        sampled.extend(pool[:take])
    rng.shuffle(sampled)
    return sampled, actual


def compact_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    match = re.search(r"[.!?](?:\s|$)", text)
    if match:
        return text[: match.end()].strip()
    return text


def gold_to_response(scenario: dict[str, Any]) -> dict[str, Any]:
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    return {
        "decision": gold["expected_decision"],
        "memory_state": gold["expected_memory_state"],
        "evidence_event_ids": gold["expected_evidence_event_ids"],
        "failure_diagnosis": gold["expected_failure_diagnosis"],
        "answer": compact_answer(gold["expected_answer"] or ""),
    }


def assert_no_leakage(user_content: str, *, scenario_id: str) -> None:
    lowered = user_content.lower()
    for marker in LEAKAGE_MARKERS:
        if marker in lowered:
            raise ValueError(f"{scenario_id}: user content leaks {marker!r}")


def sft_example(scenario: dict[str, Any]) -> dict[str, Any]:
    view = public_scenario_view(scenario)
    user_content = build_prompt(view)
    assert_no_leakage(user_content, scenario_id=str(scenario["scenario_id"]))
    response = gold_to_response(scenario)
    return {
        "task_type": "FINAL_STATE",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_V13_SMOKE},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def gold_to_revision_actions(scenario: dict[str, Any]) -> list[RevisionAction]:
    """Derive typed-action supervision from scenario hidden gold."""
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    view = build_scenario_revision_view(scenario)
    evidence_ids = tuple(str(value) for value in gold["expected_evidence_event_ids"])
    if not evidence_ids:
        evidence_ids = (view.new_evidence.evidence_id,)

    conditions_by_belief = dict(view.candidate_conditions_by_belief)
    public_memories = {
        str(memory["memory_id"]): memory
        for memory in (scenario.get("public_input") or {}).get("initial_memory") or []
        if memory.get("memory_id")
    }
    actions: list[RevisionAction] = []
    for memory_id, status in gold["expected_memory_state"].items():
        memory_id = str(memory_id)
        memory = public_memories.get(memory_id) or {}
        text = str(memory.get("text") or "")
        if status == "blocked":
            conditions = conditions_by_belief.get(memory_id) or ()
            if not conditions:
                raise ValueError(
                    f"{scenario['scenario_id']}: blocked belief {memory_id!r} has no public condition"
                )
            actions.append(
                RevisionAction(
                    action_type="BLOCKS",
                    target_condition_id=conditions[0].condition_id,
                    evidence_ids=evidence_ids,
                    rationale="The cited public evidence activates the required condition block.",
                )
            )
        elif status == "unresolved":
            actions.append(
                RevisionAction(
                    action_type="UNCERTAIN",
                    target_belief_id=memory_id,
                    evidence_ids=evidence_ids,
                    rationale="The cited public evidence leaves this belief unresolved.",
                )
            )
        elif (
            status == "current"
            and gold["expected_decision"] == "use_current_memory"
            and not text.startswith("Condition rule:")
            and not text.startswith("Distractor info:")
        ):
            actions.append(
                RevisionAction(
                    action_type="REAFFIRMS",
                    target_belief_id=memory_id,
                    evidence_ids=evidence_ids,
                    rationale="The cited verified evidence reaffirms the current belief.",
                )
            )

    if not actions:
        actions.append(
            RevisionAction(
                action_type="NO_REVISION",
                evidence_ids=evidence_ids,
                rationale="No DPA core transition is required for this policy outcome.",
            )
        )
    for action in actions:
        action.validate()
    return actions


def typed_action_sft_example(scenario: dict[str, Any]) -> dict[str, Any]:
    view = build_scenario_revision_view(scenario)
    user_content = build_proposer_prompt(view)
    assert_no_leakage(user_content, scenario_id=str(scenario["scenario_id"]))
    return {
        "task_type": "PATCH_ACTION",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the typed-action proposer for MemPatch Path A. "
                    "Return only the requested JSON array and copy every ID exactly."
                ),
            },
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": actions_to_json(gold_to_revision_actions(scenario)),
            },
        ],
    }


def multitask_sft_examples(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [sft_example(scenario), typed_action_sft_example(scenario)]


def test_balanced_row(scenario: dict[str, Any]) -> dict[str, Any]:
    view = public_scenario_view(scenario)
    return {
        "scenario_id": scenario["scenario_id"],
        "public_input": view["public_input"],
    }


def decision_distribution_sft(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("task_type") != "FINAL_STATE":
            continue
        assistant = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        counts[json.loads(assistant).get("decision", "<missing>")] += 1
    return counts


def action_distribution_sft(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("task_type") != "PATCH_ACTION":
            continue
        assistant = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        for action in json.loads(assistant):
            counts[str(action.get("action_type", "<missing>"))] += 1
    return counts


def decision_distribution_scenarios(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        decision = scenario_decision(row) or "<missing>"
        counts[decision] += 1
    return counts


def assert_label_coverage(rows: list[dict[str, Any]], *, split_name: str) -> None:
    """Fail before training if a fixed partition drops a required target label."""
    decisions: set[str] = set()
    diagnoses: set[str] = set()
    memory_statuses: set[str] = set()
    for row in rows:
        gold = canonical_hidden_gold_fields(row.get("hidden_gold") or {})
        decision = gold.get("expected_decision")
        diagnosis = gold.get("expected_failure_diagnosis")
        if isinstance(decision, str):
            decisions.add(decision)
        if isinstance(diagnosis, str):
            diagnoses.add(diagnosis)
        memory_statuses.update(
            status
            for status in gold.get("expected_memory_state", {}).values()
            if isinstance(status, str)
        )

    missing = {
        "decision": sorted(set(DECISIONS) - decisions),
        "failure_diagnosis": sorted(set(PRIMARY_FAILURE_MODES) - diagnoses),
        "memory_status": sorted(set(PRIMARY_MEMORY_STATUSES) - memory_statuses),
    }
    missing = {name: labels for name, labels in missing.items() if labels}
    if missing:
        details = ", ".join(f"{name}={labels}" for name, labels in missing.items())
        raise ValueError(f"{split_name} partition is missing required labels: {details}")


def print_distribution(label: str, counts: Counter[str]) -> None:
    print(f"\n== {label} decision distribution ==")
    for decision in DECISIONS:
        if counts.get(decision, 0):
            print(f"  {decision}: {counts[decision]}")
    for decision, count in sorted(counts.items()):
        if decision not in DECISIONS:
            print(f"  {decision}: {count}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = REPO_ROOT
    parser = argparse.ArgumentParser(description="Prepare MemPatch v1.3 multitask SFT JSONL.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=root / "local/data/mempatch/train/scenarios.jsonl",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=root / "local/data/mempatch/test/scenarios.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "local/train_data/mempatch_v13_smoke",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Use the full train split instead of the smoke TRAIN_QUOTAS sample.",
    )
    parser.add_argument(
        "--split-parts",
        type=int,
        default=DEFAULT_SPLIT_PARTS,
        help="Fixed partition count (5 → 80/20 train/val within the train split).",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="Which 1/N partition is held out for val loss (default 0).",
    )
    parser.add_argument("--seed", type=int, default=20270607)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for path, name in ((args.train_data, "train"), (args.test_data, "test")):
        if not path.is_file():
            print(f"error: {name} scenarios not found: {path}", file=sys.stderr)
            return 1

    train_rows = read_jsonl(args.train_data)
    test_source = read_jsonl(args.test_data)

    if args.full_train:
        pool = list(train_rows)
        random.Random(args.seed).shuffle(pool)
        train_actual = dict(decision_distribution_scenarios(pool))
    else:
        pool, train_actual = sample_quotas(
            index_by_decision(train_rows),
            TRAIN_QUOTAS,
            seed=args.seed,
            split_name="train",
        )
    train_sampled, valid_sampled = fixed_stratified_split(
        pool,
        split_index=args.split_index,
        split_parts=args.split_parts,
        seed=args.seed + 1,
    )
    assert_label_coverage(train_sampled, split_name="train")
    assert_label_coverage(valid_sampled, split_name="val")
    valid_actual = dict(decision_distribution_scenarios(valid_sampled))
    test_sampled, test_actual = sample_quotas(
        index_by_decision(test_source),
        TEST_BALANCED_QUOTAS,
        seed=args.seed + 2,
        split_name="test",
    )

    train_sft = [example for scenario in train_sampled for example in multitask_sft_examples(scenario)]
    valid_sft = [example for scenario in valid_sampled for example in multitask_sft_examples(scenario)]
    test_balanced = [test_balanced_row(s) for s in test_sampled]
    test_balanced_sft = [sft_example(s) for s in test_sampled]

    out_dir = args.out_dir
    write_jsonl(out_dir / "train.jsonl", train_sft)
    write_jsonl(out_dir / "valid.jsonl", valid_sft)
    write_jsonl(out_dir / "test_balanced50.jsonl", test_balanced)
    write_jsonl(out_dir / "test_balanced50_sft.jsonl", test_balanced_sft)

    if args.full_train:
        print(
            f"Wrote fixed split {args.split_index}/{args.split_parts} "
            f"({len(train_sft)} train, {len(valid_sft)} val) -> {out_dir}"
        )
    else:
        print(
            f"Wrote {len(train_sft)} train, {len(valid_sft)} valid "
            f"(split {args.split_index}/{args.split_parts}), "
            f"{len(test_balanced)} test_balanced50, {len(test_balanced_sft)} test_balanced50_sft -> {out_dir}"
        )
        print(f"val partition counts: {valid_actual}")
        if test_actual != TEST_BALANCED_QUOTAS:
            print(f"test_balanced50 actual sample counts: {test_actual}")
        print_distribution("train", decision_distribution_sft(train_sft))
        print_distribution("valid", decision_distribution_sft(valid_sft))
        print(f"train typed-action distribution: {dict(action_distribution_sft(train_sft))}")
        print(f"valid typed-action distribution: {dict(action_distribution_sft(valid_sft))}")
        print_distribution("test_balanced50 (gold in source)", decision_distribution_scenarios(test_sampled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
