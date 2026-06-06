#!/usr/bin/env python3
"""Generate MemPatch v1.2 scenario JSONL from v1.1 pools with unified decision coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.mempatch_bench.general_taxonomy import DECISIONS, canonical_hidden_gold_fields
from scripts.validate_mempatch_bench_dataset import validate_one

RENDERER = "unified_renderer_v12"
SCENARIO_JSONL = "scenarios.jsonl"

SPLIT_RANGES: dict[str, tuple[int, int]] = {
    "train": (100_001, 102_700),
    "main": (1, 800),
    "hard": (3_001, 3_500),
}

DEFAULT_QUOTAS: dict[str, dict[str, int]] = {
    "train": {
        "use_current_memory": 600,
        "mark_unresolved": 600,
        "ask_clarification": 600,
        "escalate": 600,
        "refuse_due_to_policy": 300,
    },
    "main": {
        "use_current_memory": 400,
        "mark_unresolved": 150,
        "ask_clarification": 100,
        "escalate": 75,
        "refuse_due_to_policy": 75,
    },
    "hard": {
        "use_current_memory": 150,
        "mark_unresolved": 100,
        "ask_clarification": 100,
        "escalate": 75,
        "refuse_due_to_policy": 75,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def scenario_num(scenario_id: str) -> int:
    return int(scenario_id.split("-", 1)[1])


def case_token(scenario_num_value: int) -> str:
    return f"CASE-{scenario_num_value - 1}"


def v12_seed(split: str, new_num: int, template_seed: int, variant: int) -> int:
    payload = f"mempatch_v12:{split}:{new_num}:{template_seed}:{variant}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return 1_000_000 + (int(digest[:8], 16) % 8_000_000)


def remap_scenario(
    template: dict[str, Any],
    *,
    new_scenario_id: str,
    split: str,
    variant: int,
) -> dict[str, Any]:
    old_scenario_id = template["scenario_id"]
    old_num = scenario_num(old_scenario_id)
    new_num = scenario_num(new_scenario_id)
    old_case = case_token(old_num)
    new_case = case_token(new_num)

    blob = json.dumps(template, ensure_ascii=False)
    replacements = [
        (old_scenario_id, new_scenario_id),
        (f"case-{old_num:06d}", f"case-{new_num:06d}"),
        (old_case, new_case),
    ]
    for old, new in replacements:
        blob = blob.replace(old, new)

    scenario = json.loads(blob)
    template_seed = int(template.get("metadata", {}).get("seed") or 0)
    scenario["scenario_id"] = new_scenario_id
    scenario["public_split_name"] = split
    scenario["benchmark_version"] = "v1.2"
    metadata = dict(scenario.get("metadata") or {})
    metadata["split"] = split
    metadata["renderer"] = RENDERER
    metadata["seed"] = v12_seed(split, new_num, template_seed, variant)
    metadata["v12_source_scenario_id"] = old_scenario_id
    metadata["v12_variant"] = variant
    scenario["metadata"] = metadata

    pointers = scenario.get("source_pointers") or []
    if pointers:
        updated: list[dict[str, Any]] = []
        for ptr in pointers:
            ptr = dict(ptr)
            url_or_id = str(ptr.get("url_or_id") or "")
            if url_or_id:
                ptr["url_or_id"] = re.sub(
                    r"-\d+$",
                    f"-v12-{split}-{new_num}",
                    url_or_id,
                )
            updated.append(ptr)
        scenario["source_pointers"] = updated
    return scenario


def index_by_decision(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {d: [] for d in DECISIONS}
    for row in rows:
        decision = canonical_hidden_gold_fields(row.get("hidden_gold") or {}).get("expected_decision")
        if decision in buckets:
            buckets[decision].append(row)
    return buckets


def load_quotas(manifest_path: Path | None) -> dict[str, dict[str, int]]:
    if manifest_path is None or not manifest_path.is_file():
        return deepcopy(DEFAULT_QUOTAS)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quotas = manifest.get("decision_quotas")
    if not isinstance(quotas, dict):
        return deepcopy(DEFAULT_QUOTAS)
    out: dict[str, dict[str, int]] = {}
    for split, split_quotas in quotas.items():
        if isinstance(split_quotas, dict):
            out[split] = {str(k): int(v) for k, v in split_quotas.items()}
    return out or deepcopy(DEFAULT_QUOTAS)


def passes_hard_packaging(template: dict[str, Any]) -> bool:
    probe = remap_scenario(
        template,
        new_scenario_id="case-999999",
        split="hard",
        variant=0,
    )
    errors, _ = validate_one(
        probe,
        data_path=Path("hard/scenarios.jsonl"),
        packaging_final=True,
    )
    return not errors


def draw_templates(
    pool: list[dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
    used_source_ids: set[str],
    allow_reuse: bool,
) -> list[tuple[dict[str, Any], int]]:
    if not pool:
        raise RuntimeError("empty template pool")
    available = [row for row in pool if row["scenario_id"] not in used_source_ids]
    if len(available) < count and not allow_reuse:
        raise RuntimeError(f"need {count} unique templates, only {len(available)} unused remain")
    picks: list[tuple[dict[str, Any], int]] = []
    for idx in range(count):
        if available:
            template = available.pop(rng.randrange(len(available)))
            used_source_ids.add(template["scenario_id"])
        else:
            template = pool[idx % len(pool)]
        picks.append((template, idx))
    return picks


def generate_split(
    split: str,
    quotas: dict[str, int],
    buckets: dict[str, list[dict[str, Any]]],
    *,
    rng: random.Random,
    used_source_ids: set[str],
    template_filter: Any | None = None,
) -> list[dict[str, Any]]:
    start, end = SPLIT_RANGES[split]
    next_num = start
    rows: list[dict[str, Any]] = []

    for decision in DECISIONS:
        need = int(quotas.get(decision, 0))
        if need <= 0:
            continue
        pool = list(buckets.get(decision, []))
        if template_filter is not None:
            pool = [row for row in pool if template_filter(row)]
        if not pool:
            raise RuntimeError(f"no v1.1 templates for decision={decision} (split={split})")
        rng.shuffle(pool)
        templates = draw_templates(
            pool,
            need,
            rng=rng,
            used_source_ids=used_source_ids,
            allow_reuse=True,
        )
        for template, variant in templates:
            if next_num > end:
                raise RuntimeError(f"{split} exceeded id range ({start}..{end})")
            new_id = f"case-{next_num:06d}"
            rows.append(
                remap_scenario(
                    template,
                    new_scenario_id=new_id,
                    split=split,
                    variant=variant,
                )
            )
            next_num += 1

    expected_total = sum(int(quotas.get(d, 0)) for d in DECISIONS)
    if len(rows) != expected_total:
        raise RuntimeError(f"{split}: wrote {len(rows)} rows, expected {expected_total}")
    if next_num - start != expected_total:
        raise RuntimeError(f"{split}: id allocation mismatch")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MemPatch v1.2 JSONL splits from v1.1 pools.")
    parser.add_argument(
        "--v11-main",
        type=Path,
        default=Path("local/MemPatch/main/scenarios.jsonl"),
    )
    parser.add_argument(
        "--v11-hard",
        type=Path,
        default=Path("local/MemPatch/hard/scenarios.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("hf_release/mempatch_v1_2/manifest.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("local/mempatch_v12_export"),
    )
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.v11_main.is_file() or not args.v11_hard.is_file():
        print("error: v1.1 main/hard scenarios.jsonl required", file=sys.stderr)
        return 1

    quotas = load_quotas(args.manifest)
    v11_rows = read_jsonl(args.v11_main) + read_jsonl(args.v11_hard)
    buckets = index_by_decision(v11_rows)
    print("v1.1 pool by decision:")
    for decision in DECISIONS:
        print(f"  {decision}: {len(buckets[decision])}")

    rng = random.Random(args.seed)
    used_source_ids: set[str] = set()
    out_dir = args.out_dir

    for split in ("train", "main", "hard"):
        split_quotas = quotas[split]
        template_filter = passes_hard_packaging if split == "hard" else None
        rows = generate_split(
            split,
            split_quotas,
            buckets,
            rng=rng,
            used_source_ids=used_source_ids,
            template_filter=template_filter,
        )
        dest = out_dir / split / SCENARIO_JSONL
        write_jsonl(dest, rows)
        print(f"Wrote {len(rows)} rows -> {dest}")

    print(f"Done. Export in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
