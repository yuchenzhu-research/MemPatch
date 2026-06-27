#!/usr/bin/env python3
"""Build final MemPatch-Bench v1.4 aggregate CSVs from completed score outputs.

This script never runs model inference.  It reads score JSONL files, optionally
joins prediction metadata for token/latency fields, normalizes method aliases,
and writes the canonical reporting CSV package.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mempatch.benchmark.method_names import (  # noqa: E402
    FINAL_METHODS,
    FINAL_MODELS,
    FINAL_SPLITS,
    HEADLINE_SPLITS,
    normalize_method_name,
)
from mempatch.benchmark.reporting_taxonomy import (  # noqa: E402
    MEMORY_CAPABILITIES,
    METHOD_BASELINE_FAMILIES,
    baseline_family_for_method,
    capability_for_score,
)


OUTPUT_FILENAMES = (
    "per_model_method_split.csv",
    "main_results.csv",
    "challenge_results.csv",
    "per_failure_mode.csv",
    "per_capability.csv",
    "per_baseline_family.csv",
    "per_domain.csv",
    "per_operation.csv",
    "per_difficulty.csv",
    "cost_latency.csv",
    "parse_schema_failures.csv",
    "aggregate_status.json",
)

PER_MODEL_COLUMNS = (
    "split",
    "model",
    "method",
    "n",
    "status",
    "schema_valid_rate",
    "parse_failure_rate",
    "exact_state_map",
    "contract_valid_state_success",
    "decision_accuracy",
    "decision_macro_f1",
    "answer_key_fact_accuracy",
    "memory_state_accuracy",
    "evidence_precision",
    "evidence_recall",
    "evidence_f1",
    "evidence_coverage",
    "diagnosis_accuracy",
    "strict_joint",
    "task_success_rate",
    "unsafe_reuse_rate",
    "downstream_contamination_rate",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_sec",
    "throughput_cases_per_min",
    "retrieved_event_count",
    "memory_size",
    "overcitation_rate",
    "unsupported_or_hallucinated_evidence_rate",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if bool_value(row.get(key))) / len(rows)


def optional_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows or not any(key in row for row in rows):
        return None
    return rate(rows, key)


def macro_correct(rows: list[dict[str, Any]], class_key: str, correct_key: str) -> float:
    by_class: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        cls = str(row.get(class_key) or row.get("decision_f1_class") or "")
        by_class[cls].append(bool_value(row.get(correct_key)))
    if not by_class:
        return 0.0
    return sum(sum(values) / len(values) for values in by_class.values() if values) / len(by_class)


def score_files(scores_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in scores_roots:
        if root.is_file() and root.name.endswith(".jsonl"):
            files.append(root)
        elif root.exists():
            files.extend(sorted(root.rglob("*.scores.jsonl")))
    return sorted(set(files))


def prediction_files(prediction_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in prediction_roots:
        if root.is_file() and root.name.endswith(".jsonl"):
            files.append(root)
        elif root.exists():
            files.extend(sorted(root.rglob("*.predictions.jsonl")))
    return sorted(set(files))


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    method = normalize_method_name(row.get("method"))
    return (
        str(row.get("scenario_id")),
        str(row.get("split") or ""),
        str(row.get("model") or ""),
        method,
    )


def token_latency_from_prediction(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_generation") or {}
    input_tokens = row.get("input_tokens")
    output_tokens = row.get("output_tokens")
    latency = row.get("latency_sec", row.get("latency_seconds"))

    if input_tokens is None:
        input_tokens = raw.get("input_tokens")
    if output_tokens is None:
        output_tokens = raw.get("output_tokens")
    if latency is None:
        latency = raw.get("latency_seconds")

    # MemPatch local-smoke rows have response/action token fields.
    if input_tokens is None:
        parts = [
            raw.get("response_input_tokens"),
            raw.get("action_input_tokens"),
        ]
        if any(part is not None for part in parts):
            input_tokens = sum(int(part or 0) for part in parts)
    if output_tokens is None:
        parts = [
            raw.get("response_output_tokens"),
            raw.get("action_output_tokens"),
        ]
        if any(part is not None for part in parts):
            output_tokens = sum(int(part or 0) for part in parts)
    if latency is None:
        parts = [
            raw.get("response_latency_seconds"),
            raw.get("action_latency_seconds"),
        ]
        if any(part is not None for part in parts):
            latency = sum(float(part or 0.0) for part in parts)

    total = None
    if input_tokens is not None or output_tokens is not None:
        total = int(input_tokens or 0) + int(output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "latency_sec": latency,
        "retrieved_event_count": row.get("retrieved_event_count"),
    }


def load_prediction_meta(prediction_roots: list[Path]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in prediction_files(prediction_roots):
        for row in read_jsonl(path):
            try:
                meta[prediction_key(row)] = token_latency_from_prediction(row)
            except ValueError:
                continue
    return meta


def normalize_score_row(row: dict[str, Any], prediction_meta: dict[tuple[str, str, str, str], dict[str, Any]]) -> dict[str, Any]:
    method = normalize_method_name(row.get("method"))
    normalized = dict(row)
    normalized["method"] = method
    normalized["parse_failed"] = bool_value(row.get("parse_failed", row.get("parse_failure", row.get("parse_error"))))
    normalized["contract_valid_state_success"] = bool_value(row.get("schema_valid")) and bool_value(row.get("exact_state_map"))
    normalized["decision_macro_f1_class"] = row.get("decision_macro_f1_class") or row.get("decision_f1_class")
    normalized["operation"] = row.get("memory_operation_f1_class") or row.get("expected_memory_operation") or row.get("memory_operation")
    normalized["capability"] = capability_for_score(normalized)
    normalized["baseline_family"] = baseline_family_for_method(method)
    key = (
        str(normalized.get("scenario_id")),
        str(normalized.get("split") or ""),
        str(normalized.get("model") or ""),
        method,
    )
    for field_name, value in prediction_meta.get(key, {}).items():
        if normalized.get(field_name) is None:
            normalized[field_name] = value
    if normalized.get("total_tokens") is None:
        input_tokens = number(normalized.get("input_tokens"))
        output_tokens = number(normalized.get("output_tokens"))
        if input_tokens is not None or output_tokens is not None:
            normalized["total_tokens"] = int(input_tokens or 0) + int(output_tokens or 0)
    return normalized


def load_score_rows(scores_roots: list[Path], prediction_roots: list[Path]) -> list[dict[str, Any]]:
    prediction_meta = load_prediction_meta(prediction_roots)
    rows: list[dict[str, Any]] = []
    for path in score_files(scores_roots):
        for row in read_jsonl(path):
            rows.append(normalize_score_row(row, prediction_meta))
    return rows


def aggregate_bucket(rows: list[dict[str, Any]], *, status: str = "completed") -> dict[str, Any]:
    n = len(rows)
    latency_sum = sum(number(row.get("latency_sec")) or 0.0 for row in rows)
    throughput = (n / latency_sum * 60.0) if latency_sum > 0 else None
    schema_valid_rate = rate(rows, "schema_valid")
    exact_state_map = rate(rows, "exact_state_map")
    evidence_precision = mean([number(row.get("evidence_precision")) for row in rows])
    evidence_recall = mean([number(row.get("evidence_recall")) for row in rows])
    strict_joint = rate(rows, "strict_joint")
    unsupported_rate = mean([number(row.get("unsupported_or_hallucinated_evidence_rate")) for row in rows])
    if unsupported_rate is None and evidence_precision is not None:
        unsupported_rate = max(0.0, min(1.0, 1.0 - evidence_precision))
    overcitation_rate = mean([number(row.get("overcitation_rate")) for row in rows])
    if overcitation_rate is None:
        overcitation_rate = optional_rate(rows, "overcitation")
    return {
        "n": n,
        "status": status,
        "schema_valid_rate": schema_valid_rate,
        "parse_failure_rate": rate(rows, "parse_failed"),
        "exact_state_map": exact_state_map,
        "contract_valid_state_success": sum(
            1.0
            for row in rows
            if bool_value(row.get("schema_valid")) and bool_value(row.get("exact_state_map"))
        )
        / n
        if n
        else 0.0,
        "decision_accuracy": rate(rows, "decision_correct"),
        "decision_macro_f1": macro_correct(rows, "decision_macro_f1_class", "decision_correct"),
        "answer_key_fact_accuracy": mean([number(row.get("answer_key_fact_correct")) for row in rows]),
        "memory_state_accuracy": mean([number(row.get("memory_state_accuracy")) for row in rows]) or 0.0,
        "evidence_precision": evidence_precision or 0.0,
        "evidence_recall": evidence_recall or 0.0,
        "evidence_f1": mean([number(row.get("evidence_f1")) for row in rows]) or 0.0,
        "evidence_coverage": evidence_recall or 0.0,
        "diagnosis_accuracy": rate(rows, "diagnosis_correct"),
        "strict_joint": strict_joint,
        "task_success_rate": mean([number(row.get("task_success")) for row in rows]) or strict_joint,
        "unsafe_reuse_rate": rate(rows, "unsafe_reuse"),
        "downstream_contamination_rate": rate(rows, "downstream_contamination"),
        "input_tokens": mean([number(row.get("input_tokens")) for row in rows]),
        "output_tokens": mean([number(row.get("output_tokens")) for row in rows]),
        "total_tokens": mean([number(row.get("total_tokens")) for row in rows]),
        "latency_sec": mean([number(row.get("latency_sec")) for row in rows]),
        "throughput_cases_per_min": throughput,
        "retrieved_event_count": mean([number(row.get("retrieved_event_count")) for row in rows]),
        "memory_size": mean([number(row.get("memory_size")) for row in rows]),
        "overcitation_rate": overcitation_rate,
        "unsupported_or_hallucinated_evidence_rate": unsupported_rate,
    }


def grouped(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field_name) for field_name in group_keys)
        buckets[key].append(row)
    out: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = {field_name: value for field_name, value in zip(group_keys, key)}
        record.update(aggregate_bucket(bucket))
        out.append(record)
    return out


def empty_rows_for_expected(expected_cells: set[tuple[str, str, str]], observed: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows = []
    for model, method, split in sorted(expected_cells - observed):
        rows.append(
            {
                "split": split,
                "model": model,
                "method": method,
                **aggregate_bucket([], status="pending"),
            }
        )
    return rows


def parse_cell(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected MODEL:METHOD:SPLIT")
    model, method, split = parts
    return model, normalize_method_name(method), split


def expected_from_args(args: argparse.Namespace) -> set[tuple[str, str, str]]:
    models = tuple(args.expected_model or FINAL_MODELS)
    methods = tuple(normalize_method_name(method) for method in (args.expected_method or FINAL_METHODS))
    splits = tuple(args.expected_split or FINAL_SPLITS)
    return {(model, method, split) for model in models for method in methods for split in splits}


def ensure_can_write(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "refusing to overwrite aggregate outputs without --overwrite: "
            + ", ".join(path.name for path in existing)
        )


def validate_required(rows: list[dict[str, Any]], *, allow_partial: bool) -> list[str]:
    required = (
        "scenario_id",
        "split",
        "model",
        "method",
        "schema_valid",
        "decision_correct",
        "exact_state_map",
        "memory_state_accuracy",
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
        "diagnosis_correct",
        "strict_joint",
        "unsafe_reuse",
    )
    missing: list[str] = []
    for idx, row in enumerate(rows):
        absent = [field for field in required if field not in row]
        if absent:
            missing.append(f"row {idx} missing {absent}")
            if not allow_partial and len(missing) >= 5:
                break
    return missing


def build(args: argparse.Namespace) -> dict[str, Any]:
    ensure_can_write(args.output_dir, args.overwrite)
    rows = load_score_rows(args.scores_root, args.predictions_root)
    if not rows and not args.allow_empty:
        raise RuntimeError("no score rows found; pass --allow-empty for a pending aggregate package")
    missing_columns = validate_required(rows, allow_partial=args.allow_partial)
    if missing_columns and not args.allow_partial:
        raise RuntimeError("missing required score fields: " + "; ".join(missing_columns[:5]))

    expected = expected_from_args(args)
    skipped = set(args.skip_cell or [])
    completed = {
        (str(row.get("model")), str(row.get("method")), str(row.get("split")))
        for row in rows
        if row.get("model") and row.get("method") and row.get("split")
    }
    missing_cells = sorted(expected - completed - skipped)
    unexpected_cells = sorted(completed - expected)
    if missing_cells and not args.allow_partial:
        raise RuntimeError(
            "missing expected model/method/split cells; pass --allow-partial or --skip-cell: "
            + ", ".join(f"{m}:{me}:{s}" for m, me, s in missing_cells[:10])
        )

    per_model = grouped(rows, ("split", "model", "method")) if rows else []
    per_model.extend(empty_rows_for_expected(expected - skipped, completed) if args.include_pending_cells else [])
    per_model.sort(
        key=lambda row: (
            str(row.get("split")),
            str(row.get("model")),
            FINAL_METHODS.index(row["method"]) if row.get("method") in FINAL_METHODS else 999,
        )
    )

    main_rows = [row for row in per_model if row.get("split") == "main_test_synthetic"]
    challenge_rows = [row for row in per_model if row.get("split") == "challenge_test_hard"]
    headline_bad = [row for row in main_rows + challenge_rows if row.get("split") == "dev_calibration"]
    if headline_bad:
        raise RuntimeError("headline outputs include dev_calibration")

    subgroup_specs = {
        "per_failure_mode.csv": ("split", "model", "method", "failure_mode"),
        "per_capability.csv": ("split", "model", "method", "capability"),
        "per_baseline_family.csv": ("split", "model", "baseline_family"),
        "per_domain.csv": ("split", "model", "method", "domain"),
        "per_operation.csv": ("split", "model", "method", "operation"),
        "per_difficulty.csv": ("split", "model", "method", "difficulty"),
    }
    subgroup_rows = {
        filename: grouped(rows, keys) if rows else []
        for filename, keys in subgroup_specs.items()
    }
    parse_schema = grouped(rows, ("split", "model", "method")) if rows else []
    parse_schema = [
        {
            "split": row["split"],
            "model": row["model"],
            "method": row["method"],
            "n": row["n"],
            "schema_valid_rate": row["schema_valid_rate"],
            "parse_failure_rate": row["parse_failure_rate"],
        }
        for row in parse_schema
    ]
    cost_latency = [
        {
            "split": row["split"],
            "model": row["model"],
            "method": row["method"],
            "n": row["n"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "latency_sec": row["latency_sec"],
            "throughput_cases_per_min": row["throughput_cases_per_min"],
            "retrieved_event_count": row["retrieved_event_count"],
            "memory_size": row["memory_size"],
            "overcitation_rate": row["overcitation_rate"],
            "unsupported_or_hallucinated_evidence_rate": row["unsupported_or_hallucinated_evidence_rate"],
        }
        for row in per_model
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_model_method_split.csv", per_model, PER_MODEL_COLUMNS)
    write_csv(args.output_dir / "main_results.csv", main_rows, PER_MODEL_COLUMNS)
    write_csv(args.output_dir / "challenge_results.csv", challenge_rows, PER_MODEL_COLUMNS)
    for filename, keys in subgroup_specs.items():
        columns = list(keys) + [column for column in PER_MODEL_COLUMNS if column not in {"split", "model", "method"}]
        write_csv(args.output_dir / filename, subgroup_rows[filename], columns)
    write_csv(
        args.output_dir / "cost_latency.csv",
        cost_latency,
        (
            "split",
            "model",
            "method",
            "n",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "latency_sec",
            "throughput_cases_per_min",
            "retrieved_event_count",
            "memory_size",
            "overcitation_rate",
            "unsupported_or_hallucinated_evidence_rate",
        ),
    )
    write_csv(
        args.output_dir / "parse_schema_failures.csv",
        parse_schema,
        ("split", "model", "method", "n", "schema_valid_rate", "parse_failure_rate"),
    )
    status = {
        "status": "complete" if not missing_cells and rows else ("pending" if not rows else "partial"),
        "score_rows": len(rows),
        "completed_cells": len(completed),
        "expected_cells": len(expected),
        "skipped_cells": [f"{m}:{me}:{s}" for m, me, s in sorted(skipped)],
        "missing_cells": [f"{m}:{me}:{s}" for m, me, s in missing_cells],
        "unexpected_cells": [f"{m}:{me}:{s}" for m, me, s in unexpected_cells],
        "missing_required_fields": missing_columns,
        "headline_splits": list(HEADLINE_SPLITS),
        "headline_excludes_dev_calibration": True,
        "aggregate_exports": [name for name in OUTPUT_FILENAMES if name != "aggregate_status.json"],
        "memory_capabilities": list(MEMORY_CAPABILITIES),
        "baseline_families": sorted(set(METHOD_BASELINE_FAMILIES.values())),
    }
    write_json(args.output_dir / "aggregate_status.json", status)
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-root",
        action="append",
        type=Path,
        default=[],
        help="Directory or JSONL score file. Defaults to local smoke scores.",
    )
    parser.add_argument(
        "--predictions-root",
        action="append",
        type=Path,
        default=[],
        help="Directory or JSONL prediction file used for token/latency joins.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v1.4/final_synthetic/aggregates"))
    parser.add_argument("--expected-model", action="append", default=[])
    parser.add_argument("--expected-method", action="append", default=[])
    parser.add_argument("--expected-split", action="append", default=[])
    parser.add_argument("--skip-cell", action="append", type=parse_cell, default=[])
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--include-pending-cells", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.scores_root:
        args.scores_root = [Path("runs/v1.4/local_ollama_smoke/scores")]
    if not args.predictions_root:
        args.predictions_root = [Path("runs/v1.4/local_ollama_smoke/predictions")]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        status = build(args)
    except Exception as exc:
        print(f"aggregate build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
