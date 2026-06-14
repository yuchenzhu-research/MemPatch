#!/usr/bin/env python3
"""Rescore existing prediction JSONL files without rerunning model inference."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from benchmark.api import evaluate_predictions, load_predictions, load_scenarios


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def rescore_file(
    *,
    scenarios: list[dict[str, Any]],
    prediction_path: Path,
    source_root: Path,
    out_root: Path,
) -> dict[str, Any]:
    relative = prediction_path.relative_to(source_root)
    slug = relative.parent
    tag = prediction_path.name.removesuffix("_predictions.jsonl")
    output_dir = out_root / slug
    predictions = load_predictions(prediction_path)
    scenario_ids = {row.get("scenario_id") for row in predictions}
    selected = [row for row in scenarios if row.get("scenario_id") in scenario_ids]
    result = evaluate_predictions(selected, predictions, strict=False, allow_missing=True)

    source_manifest = _load_json(prediction_path.with_name(f"{tag}_manifest.json"))
    source_metrics = _load_json(prediction_path.with_name(f"{tag}_metrics.json"))
    payload = {
        "run_tag": tag,
        "count": result.get("count"),
        "headline_metrics": result.get("headline_metrics"),
        "auxiliary_metrics": result.get("auxiliary_metrics"),
        "all_metrics": result.get("all_metrics"),
        "warnings": result.get("warnings"),
        "errors": result.get("errors"),
        "missing_prediction_count": result.get("missing_prediction_count"),
        "source_predictions": str(prediction_path.resolve()),
        "source_manifest": source_manifest,
        "source_recorded_at": source_metrics.get("recorded_at"),
        "rescore_only": True,
    }
    _write_json(output_dir / f"{tag}_metrics.json", payload)
    _write_json(
        output_dir / f"{tag}_manifest.json",
        {
            "run_tag": tag,
            "headline_metrics": result.get("headline_metrics"),
            "run_meta": {
                **(source_manifest.get("run_meta") or {}),
                "rescore_only": True,
                "source_predictions": str(prediction_path.resolve()),
            },
            "source_manifest": source_manifest,
        },
    )
    _write_jsonl(output_dir / f"{tag}_scored.jsonl", result.get("scored_predictions") or [])
    _write_jsonl(
        output_dir / f"{tag}_validation_errors.jsonl",
        [
            {
                "scenario_id": row.get("scenario_id"),
                "validation_errors": row.get("validation_errors") or [],
                "metrics": row.get("metrics"),
                "response": row.get("response"),
            }
            for row in result.get("scored_predictions") or []
            if row.get("validation_errors")
        ],
    )
    return {
        "slug": str(slug),
        "tag": tag,
        "count": len(predictions),
        **(result.get("headline_metrics") or {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--out-results", type=Path, required=True)
    parser.add_argument("--slugs", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_results.resolve()
    output = args.out_results.resolve()
    if source == output:
        raise SystemExit("error: --out-results must differ from --source-results")
    scenarios = load_scenarios(args.data)
    prediction_paths = sorted(source.glob("*/*_predictions.jsonl"))
    if args.slugs:
        allowed = set(args.slugs)
        prediction_paths = [path for path in prediction_paths if path.parent.name in allowed]
    if not prediction_paths:
        print(f"No prediction files found under {source}")
        return 0

    if output.exists():
        shutil.rmtree(output)
    rows = [
        rescore_file(
            scenarios=scenarios,
            prediction_path=path,
            source_root=source,
            out_root=output,
        )
        for path in prediction_paths
    ]
    _write_json(output / "rescore_summary.json", {"source_results": str(source), "runs": rows})
    print(f"Rescored {len(rows)} prediction files -> {output}")
    for row in rows:
        print(
            f"{row['slug']}/{row['tag']}: n={row['count']} "
            f"Joint={100 * row.get('joint_revision_success', 0):.1f} "
            f"MemState={100 * row.get('memory_state_accuracy', 0):.1f} "
            f"EvidenceF1={100 * row.get('evidence_f1', 0):.1f} "
            f"DecisionF1={100 * row.get('decision_macro_f1', 0):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
