"""Export MemPatch v1.3 scenarios to split JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.generation.blueprints import RENDERER


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def export_splits(
    scenarios_by_split: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> dict[str, Any]:
    """Write train/test scenarios.jsonl files and return manifest summary."""
    manifest: dict[str, Any] = {
        "renderer": RENDERER,
        "benchmark_version": "v1.3",
        "splits": {},
    }
    for split, rows in scenarios_by_split.items():
        path = out_dir / split / "scenarios.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][split] = {"path": str(path)}
    return manifest
