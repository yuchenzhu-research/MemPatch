#!/usr/bin/env python3
"""Write rich evaluation artifacts for downstream paper analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_eval_bundle(
    *,
    result_dir: Path,
    run_tag: str,
    eval_result: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> dict[str, Path]:
    """Persist metrics, scored rows, per-case errors, and raw predictions."""
    result_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    metrics_path = result_dir / f"{run_tag}_metrics.json"
    metrics_payload = {
        "run_tag": run_tag,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "count": eval_result.get("count"),
        "headline_metrics": eval_result.get("headline_metrics"),
        "auxiliary_metrics": eval_result.get("auxiliary_metrics"),
        "all_metrics": eval_result.get("all_metrics"),
        "warnings": eval_result.get("warnings"),
        "errors": eval_result.get("errors"),
        "missing_prediction_count": eval_result.get("missing_prediction_count"),
        "validation_error_count": len(eval_result.get("errors") or []),
        "run_meta": run_meta,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["metrics"] = metrics_path

    pred_path = result_dir / f"{run_tag}_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["predictions"] = pred_path

    scored_path = result_dir / f"{run_tag}_scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as handle:
        for row in eval_result.get("scored_predictions") or []:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["scored"] = scored_path

    error_rows: list[dict[str, Any]] = []
    for row in eval_result.get("scored_predictions") or []:
        validation_errors = row.get("validation_errors") or []
        if not validation_errors:
            continue
        error_rows.append(
            {
                "scenario_id": row.get("scenario_id"),
                "validation_errors": validation_errors,
                "metrics": row.get("metrics"),
                "response": row.get("response"),
            }
        )
    errors_path = result_dir / f"{run_tag}_validation_errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["validation_errors"] = errors_path

    manifest_path = result_dir / f"{run_tag}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "recorded_at": metrics_payload["recorded_at"],
                "artifacts": {key: str(path) for key, path in paths.items()},
                "run_meta": run_meta,
                "headline_metrics": eval_result.get("headline_metrics"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths
