#!/usr/bin/env python3
"""Prepare MemPatch v1.3 SFT data and MLX LoRA config.

Profiles:
  smoke — 500 train / 100 valid, rank-8 LoRA on q/v/o, 64 iters
  bench — quick dev train, rank-8, 32 iters
  paper — paper default, rank-16 attn+mlp, 256 iters
  paper_lite — memory-safe: seq 1024, rank-8 q/v/o, 256 iters
  heavy — full 2700 train / 200 valid, rank-16 attn+mlp, 1024 iters (legacy)

Use --full-train with --profile paper or heavy for the full train split.
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

from benchmark.general_taxonomy import DECISIONS, canonical_hidden_gold_fields
from benchmark.model_runner import build_prompt
from benchmark.public_view import public_scenario_view

SYSTEM_PROMPT_V13_SMOKE = """You are MemPatch Revision Policy.
Return only one strict JSON object.
Choose decision carefully; do not default to use_current_memory.
Use ask_clarification when user intent, target scope, version, workspace, or candidate memory is ambiguous or missing.
Use mark_unresolved when evidence is conflicting or insufficient and no user clarification or policy escalation path applies.
Use escalate when policy, authority, compliance, security, or human review blocks automatic memory revision.
Use refuse_due_to_policy only when the requested memory write or action is explicitly forbidden.
evidence_event_ids must be exact event_id strings from event_trace; cite only the minimal set that supports the chosen decision and memory_state. Do not cite distractor, background, or out-of-scope events.
failure_diagnosis must be exactly one enum from required_output_schema; never use empty string, empty list, "none", or null.
For use_current_memory: pick diagnosis from the evidence pattern (under_update if prior memory is outdated, scope_leakage if version/workspace scope is wrong, wrong_source_attribution if source is misattributed); do not default to stale_memory_reuse.
answer must be one short sentence.
Do not explain."""

TRAIN_QUOTAS: dict[str, int] = {
    "use_current_memory": 100,
    "mark_unresolved": 100,
    "ask_clarification": 100,
    "escalate": 100,
    "refuse_due_to_policy": 100,
}

VALID_QUOTAS: dict[str, int] = {d: 20 for d in DECISIONS}
VALID_HEAVY_QUOTAS: dict[str, int] = {d: 40 for d in DECISIONS}
HARD_QUOTAS: dict[str, int] = {d: 10 for d in DECISIONS}

HEAVY_LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

SMOKE_LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]

MLX_PROFILES: dict[str, dict[str, Any]] = {
    "bench": {
        "batch_size": 1,
        "iters": 32,
        "learning_rate": 1.0e-5,
        "max_seq_length": 2048,
        "grad_accumulation_steps": 4,
        "save_every": 16,
        "steps_per_eval": 16,
        "val_batches": 16,
        "lora_keys": SMOKE_LORA_KEYS,
        "lora_rank": 8,
        "lora_scale": 16.0,
        "lora_dropout": 0.05,
    },
    "smoke": {
        "batch_size": 1,
        "iters": 64,
        "learning_rate": 1.0e-5,
        "max_seq_length": 2048,
        "grad_accumulation_steps": 8,
        "save_every": 32,
        "steps_per_eval": 32,
        "val_batches": 32,
        "lora_keys": SMOKE_LORA_KEYS,
        "lora_rank": 8,
        "lora_scale": 16.0,
        "lora_dropout": 0.05,
    },
    "heavy": {
        "batch_size": 2,
        "iters": 1024,
        "learning_rate": 1.0e-5,
        "max_seq_length": 2048,
        "grad_accumulation_steps": 4,
        "save_every": 128,
        "steps_per_eval": 128,
        "val_batches": 64,
        "lora_keys": HEAVY_LORA_KEYS,
        "lora_rank": 16,
        "lora_scale": 32.0,
        "lora_dropout": 0.05,
    },
    "paper": {
        "batch_size": 1,
        "iters": 256,
        "learning_rate": 1.0e-5,
        "max_seq_length": 2048,
        "grad_accumulation_steps": 4,
        "save_every": 64,
        "steps_per_eval": 64,
        "val_batches": 32,
        "lora_keys": HEAVY_LORA_KEYS,
        "lora_rank": 16,
        "lora_scale": 32.0,
        "lora_dropout": 0.05,
    },
    "paper_lite": {
        "batch_size": 1,
        "iters": 256,
        "learning_rate": 1.0e-5,
        "max_seq_length": 1024,
        "grad_accumulation_steps": 8,
        "save_every": 32,
        "steps_per_eval": 32,
        "val_batches": 32,
        "lora_keys": SMOKE_LORA_KEYS,
        "lora_rank": 8,
        "lora_scale": 16.0,
        "lora_dropout": 0.05,
    },
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

def mlx_mask_prompt_for_model(model_dir: Path) -> bool:
    """Mistral chat templates can make prompt offsets exceed the full sequence under mask_prompt."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return True
    model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
    return model_type not in {"mistral"}


def mlx_lora_yaml(
    *,
    root: Path,
    data_dir: Path,
    adapter_dir: Path,
    profile: str = "smoke",
    model_dir: Path | None = None,
) -> str:
    if profile not in MLX_PROFILES:
        raise ValueError(f"unknown MLX profile {profile!r}; expected one of {sorted(MLX_PROFILES)}")
    cfg = MLX_PROFILES[profile]
    if model_dir is None:
        model_dir = root / "local/models/Qwen3-14B-MLX-4bit"
    mask_prompt = mlx_mask_prompt_for_model(model_dir)
    keys_yaml = json.dumps(cfg["lora_keys"])
    return f"""model: "{model_dir.resolve()}"
train: true
fine_tune_type: lora
optimizer: adamw
data: "{data_dir.resolve()}"
seed: 2027
batch_size: {cfg["batch_size"]}
iters: {cfg["iters"]}
learning_rate: {cfg["learning_rate"]}
max_seq_length: {cfg["max_seq_length"]}
grad_accumulation_steps: {cfg["grad_accumulation_steps"]}
grad_checkpoint: true
mask_prompt: {"true" if mask_prompt else "false"}
adapter_path: "{adapter_dir.resolve()}"
save_every: {cfg["save_every"]}
steps_per_eval: {cfg["steps_per_eval"]}
val_batches: {cfg["val_batches"]}
lora_parameters:
  keys: {keys_yaml}
  rank: {cfg["lora_rank"]}
  scale: {cfg["lora_scale"]}
  dropout: {cfg["lora_dropout"]}
"""


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
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_V13_SMOKE},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def hard_balanced_row(scenario: dict[str, Any]) -> dict[str, Any]:
    view = public_scenario_view(scenario)
    return {
        "scenario_id": scenario["scenario_id"],
        "public_input": view["public_input"],
    }


def decision_distribution_sft(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        assistant = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        counts[json.loads(assistant).get("decision", "<missing>")] += 1
    return counts


def decision_distribution_scenarios(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        decision = scenario_decision(row) or "<missing>"
        counts[decision] += 1
    return counts


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
    parser = argparse.ArgumentParser(description="Prepare MemPatch v1.3 SFT bundle and MLX LoRA config.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=root / "hf_release/mempatch/train/scenarios.jsonl",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=root / "hf_release/mempatch/validation/scenarios.jsonl",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=root / "hf_release/mempatch/test/scenarios.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "local/train_data/mempatch_v13_smoke",
    )
    parser.add_argument(
        "--mlx-config",
        type=Path,
        default=root / "local/logs/qwen3_14b_mempatch_v13_smoke/mlx_lora.yaml",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=root / "local/adapters/qwen3_14b_mempatch_v13_smoke",
        help="LoRA adapter output directory written into the MLX config.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "local/models/Qwen3-14B-MLX-4bit",
        help="Base MLX model directory for the LoRA config.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only write mlx_lora.yaml (reuse existing SFT JSONL in --out-dir).",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(MLX_PROFILES),
        default="smoke",
        help="MLX LoRA training profile written into the config.",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Use the full train split instead of the smoke TRAIN_QUOTAS sample.",
    )
    parser.add_argument("--seed", type=int, default=20270607)
    return parser.parse_args(argv)


def write_mlx_config(root: Path, args: argparse.Namespace) -> None:
    args.mlx_config.parent.mkdir(parents=True, exist_ok=True)
    args.mlx_config.write_text(
        mlx_lora_yaml(
            root=root,
            data_dir=args.out_dir,
            adapter_dir=args.adapter_dir,
            profile=args.profile,
            model_dir=args.model_dir,
        ),
        encoding="utf-8",
    )
    print(f"Wrote MLX config ({args.profile}) -> {args.mlx_config}")


def main(argv: list[str] | None = None) -> int:
    root = REPO_ROOT
    args = parse_args(argv)
    if args.config_only:
        if not (args.out_dir / "train.jsonl").is_file():
            print(f"error: --config-only requires {args.out_dir}/train.jsonl", file=sys.stderr)
            return 1
        write_mlx_config(root, args)
        return 0

    for path, name in (
        (args.train_data, "train"),
        (args.validation_data, "validation"),
        (args.test_data, "test"),
    ):
        if not path.is_file():
            print(f"error: {name} scenarios not found: {path}", file=sys.stderr)
            return 1

    train_rows = read_jsonl(args.train_data)
    valid_source = read_jsonl(args.validation_data)
    test_source = read_jsonl(args.test_data)

    valid_quotas = (
        VALID_HEAVY_QUOTAS
        if args.profile in ("heavy", "paper", "paper_lite")
        else VALID_QUOTAS
    )

    if args.full_train:
        train_sampled = list(train_rows)
        random.Random(args.seed).shuffle(train_sampled)
        train_actual = dict(decision_distribution_scenarios(train_sampled))
    else:
        train_sampled, train_actual = sample_quotas(
            index_by_decision(train_rows),
            TRAIN_QUOTAS,
            seed=args.seed,
            split_name="train",
        )
    valid_sampled, valid_actual = sample_quotas(
        index_by_decision(valid_source),
        valid_quotas,
        seed=args.seed + 1,
        split_name="validation",
    )
    hard_sampled, hard_actual = sample_quotas(
        index_by_decision(test_source),
        HARD_QUOTAS,
        seed=args.seed + 2,
        split_name="test",
    )

    train_sft = [sft_example(s) for s in train_sampled]
    valid_sft = [sft_example(s) for s in valid_sampled]
    hard50 = [hard_balanced_row(s) for s in hard_sampled]
    hard50_sft = [sft_example(s) for s in hard_sampled]

    out_dir = args.out_dir
    write_jsonl(out_dir / "train.jsonl", train_sft)
    write_jsonl(out_dir / "valid.jsonl", valid_sft)
    write_jsonl(out_dir / "hard_balanced50.jsonl", hard50)
    write_jsonl(out_dir / "hard_balanced50_sft.jsonl", hard50_sft)

    write_mlx_config(root, args)

    print(
        f"Wrote {len(train_sft)} train, {len(valid_sft)} valid, "
        f"{len(hard50)} hard_balanced50, {len(hard50_sft)} hard_balanced50_sft -> {out_dir}"
    )
    if valid_actual != valid_quotas:
        print(f"validation actual sample counts: {valid_actual}")
    if hard_actual != HARD_QUOTAS:
        print(f"test balanced50 actual sample counts: {hard_actual}")

    print_distribution("train", decision_distribution_sft(train_sft))
    print_distribution("valid", decision_distribution_sft(valid_sft))
    print_distribution("hard_balanced50 (gold in source)", decision_distribution_scenarios(hard_sampled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
