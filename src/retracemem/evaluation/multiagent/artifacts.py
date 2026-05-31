"""Structured run-output writing for the multi-agent evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_outputs(
    output_dir: str,
    stage_a_raw_rows: list[dict[str, Any]],
    stage_b_raw_rows: list[dict[str, Any]],
    stage_a_parsed_rows: list[dict[str, Any]],
    stage_b_parsed_rows: list[dict[str, Any]],
    dpa_trace_rows: list[dict[str, Any]],
    global_metrics: dict[str, Any],
    manifest: dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Writes all raw files, metrics, and manifest to disk."""
    if dry_run:
        return
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def save_jsonl(p: Path, data: list):
        with open(p, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

    save_jsonl(output_path / "stage_a_raw.jsonl", stage_a_raw_rows)
    save_jsonl(output_path / "stage_b_raw.jsonl", stage_b_raw_rows)
    save_jsonl(output_path / "stage_a_parsed.jsonl", stage_a_parsed_rows)
    save_jsonl(output_path / "stage_b_parsed.jsonl", stage_b_parsed_rows)
    save_jsonl(output_path / "dpa_traces.jsonl", dpa_trace_rows)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)

    with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("✓ Wrote all raw, parsed, and trace files to output folder.")

