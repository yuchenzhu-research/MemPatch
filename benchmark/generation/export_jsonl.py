"""Export MemPatch v1.3 scenarios to split JSONL files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.generation.blueprints import RENDERER
from benchmark.general_taxonomy import DECISIONS, canonical_hidden_gold_fields


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def export_splits(
    scenarios_by_split: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> dict[str, Any]:
    """Write train/main/hard/scenarios.jsonl and return export manifest summary."""
    manifest: dict[str, Any] = {
        "renderer": RENDERER,
        "benchmark_version": "v1.3",
        "splits": {},
    }
    for split, rows in scenarios_by_split.items():
        path = out_dir / split / "scenarios.jsonl"
        write_jsonl(path, rows)
        decisions = Counter()
        patterns: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            gold = canonical_hidden_gold_fields(row.get("hidden_gold") or {})
            decision = gold.get("expected_decision") or "<missing>"
            decisions[decision] += 1
            pattern = str(row.get("pattern") or row.get("metadata", {}).get("pattern") or "<missing>")
            patterns[pattern][decision] += 1
        manifest["splits"][split] = {
            "path": str(path),
            "count": len(rows),
            "decision_counts": {d: decisions.get(d, 0) for d in DECISIONS},
            "pattern_decision_matrix": {p: dict(c) for p, c in sorted(patterns.items())},
        }
    return manifest
