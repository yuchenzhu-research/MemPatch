#!/usr/bin/env python3
"""Export human-annotation packets for ReTrace-Bench (internal "v1.1").

This builds two stratified packets from the canonical public splits
(``main`` / ``hard`` / ``realistic`` / ``calibration`` — the private hidden
split is *never* sampled into an annotation packet):

* ``quick_audit_50``      — 50-example internal author audit (Level 1).
* ``paper_validation_200`` — 200-example paper-grade validation pool (Level 2).

For each packet it writes a gold-free ``*_public.jsonl`` (what annotators see),
a ``*_gold.jsonl`` (scenario_id + hidden_gold, for scoring only), and for the
paper packet also a ready-to-fill ``*_sheet.csv`` and a ``*_readme.md``.

The public packet is produced through ``benchmark.retrace_bench.public_view`` so
it can never leak ``hidden_gold``, ``metadata``, ``primary_failure_mode``, etc.
The script self-checks this invariant before writing.

No model APIs are called and no gold is fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.retrace_bench.general_taxonomy import (
    DECISIONS,
    FAILURE_MODES,
    MEMORY_STATUSES,
)
from benchmark.retrace_bench.generation.pattern_spec import infer_pattern
from benchmark.retrace_bench.public_view import (
    INTERNAL_ONLY_FIELDS,
    public_scenario_view,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = REPO / "data" / "retrace_bench_v1_1"
DEFAULT_OUT = REPO / "annotation_packets" / "retrace_bench_v1_1"

# Public splits eligible for annotation (private_hidden is intentionally excluded).
PACKET_SPLITS = (
    ("main_3000_en", "main"),
    ("hard_500_en", "hard"),
    ("realistic_200_en", "realistic"),
    ("calibration_80_en", "calibration"),
)

ALLOWED_LABELS = {
    "decision_label": list(DECISIONS),
    "memory_status": list(MEMORY_STATUSES),
    "failure_diagnosis": list(FAILURE_MODES),
    "solvable_from_visible_evidence": ["yes", "no", "uncertain"],
    "topic_domain_consistent": ["yes", "no", "uncertain"],
    "ambiguous_or_multiple_valid_answers": ["yes", "no", "uncertain"],
    "filler_heavy": ["yes", "no", "uncertain"],
    "confidence": [1, 2, 3, 4, 5],
}

# Annotation fields written as empty columns in the CSV entry sheet.
SHEET_FIELDS = [
    "annotator_id",
    "scenario_id",
    "solvable_from_visible_evidence",
    "topic_domain_consistent",
    "ambiguous_or_multiple_valid_answers",
    "filler_heavy",
    "decision_label",
    "answer_short_free_text",
    "memory_state_json",
    "evidence_event_ids",
    "failure_diagnosis",
    "confidence",
    "notes",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _decision(sc: dict[str, Any]) -> str:
    return sc.get("hidden_gold", {}).get("expected_decision", "") or ""


def _is_non_answer(sc: dict[str, Any]) -> bool:
    """A 'non-answer' gold decision is anything other than answering from memory."""
    return _decision(sc) not in ("", "use_current_memory")


def _difficulty(sc: dict[str, Any]) -> str:
    return sc.get("difficulty") or sc.get("difficulty_level") or ""


def load_pool(bench: Path) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for dir_name, split in PACKET_SPLITS:
        path = bench / dir_name / "scenarios.jsonl"
        if not path.exists():
            continue
        for sc in _read_jsonl(path):
            sc["_split"] = split
            sc["_pattern"] = infer_pattern(sc)
            pool.append(sc)
    return pool


def _greedy_select(
    pool: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    *,
    min_hard: int,
    min_non_answer: int,
    split_targets: dict[str, int],
) -> list[dict[str, Any]]:
    """Stratified, coverage-first selection.

    Guarantees (when the pool allows): all five decision labels, all fifteen
    patterns, both L3 and L4, >= ``min_hard`` hard rows, and >= ``min_non_answer``
    non-answer gold decisions, then fills the remainder toward ``split_targets``
    (a desired per-split row count) while keeping the pattern spread balanced.
    """
    chosen: dict[str, dict[str, Any]] = {}

    def take(sc: dict[str, Any]) -> None:
        chosen[sc["scenario_id"]] = sc

    def pick(predicate, prefer_splits=("hard", "main", "realistic", "calibration"), count=1):
        for split in prefer_splits:
            cands = [
                sc for sc in pool
                if sc["scenario_id"] not in chosen and sc["_split"] == split and predicate(sc)
            ]
            rng.shuffle(cands)
            for sc in cands:
                if count <= 0:
                    return
                take(sc)
                count -= 1
            if count <= 0:
                return

    # 1. Guarantee every decision label (rare ones come from hard).
    for decision in DECISIONS:
        pick(lambda sc, d=decision: _decision(sc) == d, count=max(1, n // 25))
    # 2. Guarantee every workflow pattern.
    for pattern in {sc["_pattern"] for sc in pool if sc["_pattern"]}:
        if not any(sc["_pattern"] == pattern for sc in chosen.values()):
            pick(lambda sc, p=pattern: sc["_pattern"] == p, count=1)
    # 3. Guarantee both hard difficulties.
    for diff in ("L3", "L4"):
        if not any(_difficulty(sc) == diff and sc["_split"] == "hard" for sc in chosen.values()):
            pick(lambda sc, dd=diff: _difficulty(sc) == dd, prefer_splits=("hard",), count=1)
    # 4. Hit the hard-row floor.
    need_hard = min_hard - sum(1 for sc in chosen.values() if sc["_split"] == "hard")
    if need_hard > 0:
        pick(lambda sc: True, prefer_splits=("hard",), count=need_hard)
    # 5. Hit the non-answer floor.
    need_na = min_non_answer - sum(1 for sc in chosen.values() if _is_non_answer(sc))
    if need_na > 0:
        pick(_is_non_answer, count=need_na)

    # 6. Fill the remainder toward the per-split targets, keeping pattern spread.
    pat_count: Counter[str] = Counter(sc["_pattern"] for sc in chosen.values())
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sc in pool:
        if sc["scenario_id"] not in chosen:
            by_split[sc["_split"]].append(sc)
    for cands in by_split.values():
        # Prefer under-represented patterns first, then deterministic shuffle.
        rng.shuffle(cands)
        cands.sort(key=lambda sc: (pat_count[sc["_pattern"]], rng.random()))

    while len(chosen) < n:
        cur = Counter(sc["_split"] for sc in chosen.values())
        # Pick the split furthest below its target that still has candidates.
        deficits = sorted(
            ((cur.get(s, 0) - split_targets.get(s, 0), s) for s in split_targets),
            key=lambda x: (x[0], x[1]),
        )
        progressed = False
        for _deficit, split in deficits:
            if by_split.get(split):
                take(by_split[split].pop(0))
                progressed = True
                break
        if not progressed:
            # Targets exhausted; fall back to anything still available.
            leftovers = [sc for lst in by_split.values() for sc in lst]
            if not leftovers:
                break
            take(leftovers[0])
            for lst in by_split.values():
                if lst and lst[0] is leftovers[0]:
                    lst.pop(0)
                    break

    selected = list(chosen.values())[:n]
    # Stable, reproducible ordering in the packet.
    selected.sort(key=lambda sc: sc["scenario_id"])
    return selected


def _public_row(sc: dict[str, Any]) -> dict[str, Any]:
    view = public_scenario_view(sc)
    view["split"] = sc["_split"]
    view["allowed_labels"] = ALLOWED_LABELS
    return view


def _assert_gold_free(rows: list[dict[str, Any]]) -> None:
    blob = json.dumps(rows)
    for field in sorted(INTERNAL_ONLY_FIELDS | {"expected_decision", "expected_answer",
                                                "expected_memory_state", "expected_evidence_event_ids",
                                                "expected_failure_diagnosis"}):
        assert field not in blob, f"public packet leaks forbidden field: {field}"


def _gold_row(sc: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": sc["scenario_id"],
        "split": sc["_split"],
        "pattern": sc["_pattern"],
        "difficulty": _difficulty(sc),
        "hidden_gold": sc.get("hidden_gold", {}),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_sheet(path: Path, public_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for row in public_rows:
            writer.writerow({"scenario_id": row["scenario_id"]})


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "splits": dict(Counter(sc["_split"] for sc in rows)),
        "decisions": dict(Counter(_decision(sc) for sc in rows)),
        "patterns": len({sc["_pattern"] for sc in rows}),
        "difficulties": dict(Counter(_difficulty(sc) for sc in rows)),
        "domains": len({sc.get("domain") for sc in rows}),
        "hard_rows": sum(1 for sc in rows if sc["_split"] == "hard"),
        "non_answer_rows": sum(1 for sc in rows if _is_non_answer(sc)),
    }


def export_packet(
    name: str, pool: list[dict[str, Any]], n: int, seed: int, out_dir: Path, *,
    min_hard: int, min_non_answer: int, write_sheet: bool,
    split_targets: dict[str, int],
) -> dict[str, Any]:
    rng = random.Random(seed)
    selected = _greedy_select(
        pool, n, rng, min_hard=min_hard, min_non_answer=min_non_answer,
        split_targets=split_targets,
    )
    public_rows = [_public_row(sc) for sc in selected]
    _assert_gold_free(public_rows)
    gold_rows = [_gold_row(sc) for sc in selected]

    _write_jsonl(out_dir / f"{name}_public.jsonl", public_rows)
    _write_jsonl(out_dir / f"{name}_gold.jsonl", gold_rows)
    if write_sheet:
        _write_sheet(out_dir / f"{name}_sheet.csv", public_rows)
    return _coverage(selected)


def _write_paper_readme(path: Path, cov: dict[str, Any]) -> None:
    path.write_text(
        "# ReTrace-Bench paper-grade human validation packet (200)\n\n"
        "**Do not open the `_gold.jsonl` file before annotating.** Annotators work\n"
        "only from `paper_validation_200_public.jsonl` (gold-free) and record answers\n"
        "in `paper_validation_200_sheet.csv` (one row per `scenario_id`).\n\n"
        "## Files\n"
        "- `paper_validation_200_public.jsonl` — gold-free scenarios (annotator input).\n"
        "- `paper_validation_200_sheet.csv` — empty entry sheet, one row per scenario.\n"
        "- `paper_validation_200_gold.jsonl` — hidden gold, **scoring lead only**.\n\n"
        "## Procedure\n"
        "1. Read `docs/retrace_bench/human_annotation_codebook.md` first.\n"
        "2. At least two independent human annotators each fill their own copy of the sheet.\n"
        "3. Set a unique `annotator_id` per person (LLMs may NOT be annotators).\n"
        "4. Adjudicate disagreements, then run `scripts/score_human_annotations.py`.\n\n"
        "## Coverage of this packet\n"
        f"- scenarios: {cov['n']}\n"
        f"- splits: {cov['splits']}\n"
        f"- decision labels: {cov['decisions']}\n"
        f"- patterns covered: {cov['patterns']}/15\n"
        f"- difficulties: {cov['difficulties']}\n"
        f"- distinct domains: {cov['domains']}\n"
        f"- hard rows: {cov['hard_rows']} (target >= 50)\n"
        f"- non-answer gold rows: {cov['non_answer_rows']} (target >= 40)\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args(argv)

    pool = load_pool(args.bench)
    if not pool:
        raise SystemExit(f"no scenarios found under {args.bench}")

    quick = export_packet(
        "quick_audit_50", pool, 50, args.seed, args.out,
        min_hard=15, min_non_answer=15, write_sheet=False,
        split_targets={"hard": 20, "main": 15, "realistic": 10, "calibration": 5},
    )
    paper = export_packet(
        "paper_validation_200", pool, 200, args.seed + 1, args.out,
        min_hard=50, min_non_answer=40, write_sheet=True,
        split_targets={"hard": 90, "main": 80, "realistic": 20, "calibration": 10},
    )
    _write_paper_readme(args.out / "paper_validation_200_readme.md", paper)

    print("quick_audit_50 coverage:", json.dumps(quick))
    print("paper_validation_200 coverage:", json.dumps(paper))
    print(f"packets written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
