#!/usr/bin/env python3
"""Rebuild the displayed Table 3 values from frozen rows and predictions.

This script performs no model inference.  It verifies the frozen main-split
identity, deterministically reconstructs the selective MemPatch response path,
scores all frozen paper cells plus a deterministic scope-only baseline, and
checks the one-decimal values printed in the paper.  The archived
``mempatch_typed_projection`` rows are the typed fallback, not the final
paper-facing MemPatch rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from mempatch.benchmark.api import (  # noqa: E402
    _scenario_event_ids,
    _scenario_memory_ids,
    _validate_response,
)
from mempatch.benchmark.contracts import validate_prediction  # noqa: E402
from mempatch.benchmark.score import score_row  # noqa: E402
from mempatch.benchmark.scorers_general import score_prediction  # noqa: E402


FROZEN_HASHES = {
    "raw": "64fbd89520f6e847c041efba10bf8ee22c3bfd56bca90a7874657ac9a2b04a1e",
    "labels": "e0bffbe9860973b9b1a7b9ba3b5138f96189f0a59484262ac7533764632fc750",
    "public": "97564e9f71203f348b8571497b4d72d58b5ba3f675ef0e9aa521c3622baca2f1",
}

MODELS = (
    "deepseek_r1_qwen_14b",
    "glm4_9b",
    "mistral_nemo_12b",
    "phi4_14b",
    "qwen3_14b",
)

PREDICTION_METHODS = (
    "direct_json",
    "full_context_json",
    "summary_memory_json",
    "bm25_rag_json",
    "dense_rag_json",
    "time_aware_rag_json",
    "mempatch",
)
METHODS = PREDICTION_METHODS + ("scope_only",)

DISPLAY_NAMES = {
    "direct_json": "Direct JSON",
    "full_context_json": "Full context",
    "summary_memory_json": "Summary",
    "bm25_rag_json": "Lexical RAG",
    "dense_rag_json": "Hash-vector RAG",
    "time_aware_rag_json": "Time-aware RAG",
    "mempatch": "MemPatch",
    "scope_only": "Scope-only",
}

PAPER_TABLE3 = {
    "direct_json": (20.9, 7.2, 11.9, 47.7, 48.6),
    "full_context_json": (20.9, 7.5, 12.0, 47.9, 48.5),
    "summary_memory_json": (15.0, 3.7, 15.6, 51.0, 29.7),
    "bm25_rag_json": (19.9, 9.5, 9.8, 46.6, 34.9),
    "dense_rag_json": (20.3, 8.6, 11.0, 47.4, 47.5),
    "time_aware_rag_json": (18.6, 8.6, 10.2, 46.8, 33.5),
    "mempatch": (24.1, 7.2, 12.3, 56.4, 47.4),
    "scope_only": (17.6, 4.8, 36.4, 85.7, 0.0),
}

EXPECTED_TRANSITION_JOINT_HITS = {
    "direct_json": 0,
    "full_context_json": 0,
    "summary_memory_json": 0,
    "bm25_rag_json": 5,
    "dense_rag_json": 25,
    "time_aware_rag_json": 10,
    "mempatch": 0,
    "scope_only": 0,
}

ROW_METRICS = (
    "schema",
    "decision_accuracy",
    "operation_accuracy",
    "exact",
    "state",
    "evidence_f1",
    "diagnosis",
    "transition_joint",
    "contamination",
)

PRIMARY_TABLE_METRICS = (
    "decision_macro_f1",
    "operation_macro_f1",
    "exact",
    "state",
    "evidence_f1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def native_response_valid(scenario: dict[str, Any], response: Any) -> bool:
    errors: list[str] = []
    warnings: list[str] = []
    _validate_response(
        str(scenario["scenario_id"]),
        response,
        _scenario_event_ids(scenario),
        _scenario_memory_ids(scenario),
        errors,
        warnings,
    )
    return not errors


def partial_response_valid(response: Any) -> bool:
    return isinstance(response, dict) and not validate_prediction({"parsed": response})


def reconstruct_mempatch(
    direct_rows: list[dict[str, Any]],
    typed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    direct_by_id = {str(row["scenario_id"]): row for row in direct_rows}
    out: list[dict[str, Any]] = []
    counts = {"direct_json_valid": 0, "typed_projection_fallback": 0}
    for typed in typed_rows:
        scenario_id = str(typed["scenario_id"])
        direct = direct_by_id[scenario_id]
        use_direct = partial_response_valid(direct.get("response"))
        source = "direct_json_valid" if use_direct else "typed_projection_fallback"
        response = direct["response"] if use_direct else typed["response"]
        rebuilt = {
            **typed,
            "method": "mempatch",
            "response": response,
            "repair_source": source,
        }
        counts[source] += 1
        out.append(rebuilt)
    return out, counts


def scope_only_predictions(
    scenarios: list[dict[str, Any]],
    *,
    model: str,
) -> list[dict[str, Any]]:
    """Build the deterministic public-scope baseline without model inference."""
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        public_input = scenario.get("public_input", {}) or {}
        memories = (
            public_input.get("initial_memory")
            or public_input.get("initial_memories")
            or []
        )
        memory_state = {
            str(memory["memory_id"]): (
                "out_of_scope"
                if str(memory.get("scope", "")).lower().endswith("-side")
                else "current"
            )
            for memory in memories
            if memory.get("memory_id") is not None
        }
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "method": "scope_only",
                "model": model,
                "response": {
                    "answer": "",
                    "decision": "use_current_memory",
                    "memory_operation": "REVISE",
                    "memory_state": memory_state,
                    "evidence_event_ids": [],
                    "failure_diagnosis": "stale_memory_reuse",
                    "followup_answer": "",
                },
            }
        )
    return rows


def macro_f1(expected: list[Any], predicted: list[Any]) -> float:
    """Return standard macro-F1 over classes observed in frozen gold labels."""
    if len(expected) != len(predicted):
        raise ValueError("macro-F1 inputs must have the same length")
    classes = sorted(set(expected), key=str)
    if not classes:
        return 0.0
    class_f1: list[float] = []
    for cls in classes:
        tp = sum(p == cls and y == cls for y, p in zip(expected, predicted))
        fp = sum(p == cls and y != cls for y, p in zip(expected, predicted))
        fn = sum(p != cls and y == cls for y, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_f1.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(class_f1) / len(class_f1)


def row_metrics(
    scenario: dict[str, Any],
    label: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, float]:
    label_metrics = score_row(label, prediction)
    deployment_metrics = score_prediction(scenario, prediction)
    response = prediction.get("response")
    return {
        # This is deliberately the same scenario-aware diagnostic for every
        # method.  The frozen partial validator remains only in branch
        # reconstruction above because it defines the archived MemPatch policy.
        "schema": float(native_response_valid(scenario, response)),
        "decision_accuracy": float(label_metrics["decision_correct"]),
        "operation_accuracy": float(label_metrics["memory_operation_correct"]),
        "exact": float(label_metrics["exact_state_map"]),
        "state": float(label_metrics["memory_state_accuracy"]),
        "evidence_f1": float(label_metrics["evidence_f1"]),
        "diagnosis": float(label_metrics["diagnosis_correct"]),
        # Same-case exact control transition. This is narrower than the legacy
        # strict_joint diagnostic, which also requires schema, diagnosis,
        # follow-up correctness, and two unsafe-reuse checks.
        "transition_joint": float(
            label_metrics["decision_correct"]
            and label_metrics["memory_operation_correct"]
            and label_metrics["exact_state_map"]
            and label_metrics["evidence_f1"] == 1.0
        ),
        "contamination": float(deployment_metrics["downstream_contamination_rate"]),
    }


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def build(artifact_root: Path, materialize: bool) -> dict[str, Any]:
    raw_path = artifact_root / "data/synthetic/frozen_run_input/main_test_synthetic.jsonl"
    labels_path = artifact_root / "data/synthetic/release/labels/main_test_synthetic.labels.jsonl"
    public_path = artifact_root / "data/synthetic/release/public/main_test_synthetic.jsonl"
    predictions_root = artifact_root / "results/frozen_main_predictions"

    observed_hashes = {
        "raw": sha256(raw_path),
        "labels": sha256(labels_path),
        "public": sha256(public_path),
    }
    if observed_hashes != FROZEN_HASHES:
        raise RuntimeError(f"frozen data identity mismatch: {observed_hashes}")

    scenarios = read_jsonl(raw_path)
    labels = read_jsonl(labels_path)
    scenario_by_id = {str(row["scenario_id"]): row for row in scenarios}
    label_by_id = {str(row["scenario_id"]): row for row in labels}
    expected_ids = [str(row["scenario_id"]) for row in labels]
    if len(expected_ids) != 3000 or len(set(expected_ids)) != 3000:
        raise RuntimeError("frozen main split must contain 3,000 unique labels")
    decision_classes = sorted({row.get("expected_decision") for row in labels})
    operation_classes = sorted(
        {row.get("expected_memory_operation") for row in labels}
    )
    if len(decision_classes) != 4 or None in decision_classes:
        raise RuntimeError(
            f"frozen main split must contain four decision classes: {decision_classes}"
        )
    if len(operation_classes) != 9 or None in operation_classes:
        raise RuntimeError(
            f"frozen main split must contain nine operation classes: {operation_classes}"
        )

    per_model: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    transition_joint_hits: dict[str, dict[str, int]] = defaultdict(dict)
    repair_counts: dict[str, dict[str, int]] = {}
    prediction_hashes: dict[str, str] = {}

    for model in MODELS:
        model_root = predictions_root / model
        direct_rows = read_jsonl(model_root / "direct_json.predictions.jsonl")
        typed_path = model_root / "mempatch_typed_projection.predictions.jsonl"
        typed_rows = read_jsonl(typed_path)
        effective_rows, counts = reconstruct_mempatch(direct_rows, typed_rows)
        repair_counts[model] = counts
        effective_path = model_root / "mempatch.predictions.jsonl"
        if materialize:
            write_jsonl(effective_path, effective_rows)
        elif effective_path.exists():
            provided = read_jsonl(effective_path)
            if provided != effective_rows:
                raise RuntimeError(f"{model}: materialized MemPatch rows do not match reconstruction")

        rows_by_method: dict[str, list[dict[str, Any]]] = {}
        for method in PREDICTION_METHODS:
            rows = (
                effective_rows
                if method == "mempatch"
                else read_jsonl(model_root / f"{method}.predictions.jsonl")
            )
            ids = [str(row.get("scenario_id")) for row in rows]
            if ids != expected_ids:
                raise RuntimeError(f"{model}/{method}: prediction IDs or order differ from frozen labels")
            rows_by_method[method] = rows
            source_path = effective_path if method == "mempatch" else model_root / f"{method}.predictions.jsonl"
            if source_path.exists():
                prediction_hashes[f"{model}/{method}"] = sha256(source_path)
        rows_by_method["scope_only"] = scope_only_predictions(scenarios, model=model)

        for method, rows in rows_by_method.items():
            sums = defaultdict(float)
            predicted_decisions: list[Any] = []
            expected_decisions: list[Any] = []
            predicted_operations: list[Any] = []
            expected_operations: list[Any] = []
            for prediction in rows:
                scenario_id = str(prediction["scenario_id"])
                label = label_by_id[scenario_id]
                scored = row_metrics(
                    scenario_by_id[scenario_id],
                    label,
                    prediction,
                )
                for metric, value in scored.items():
                    sums[metric] += value
                response = prediction.get("response") or {}
                predicted_decisions.append(response.get("decision"))
                expected_decisions.append(label.get("expected_decision"))
                predicted_operations.append(response.get("memory_operation"))
                expected_operations.append(label.get("expected_memory_operation"))
            per_model[model][method] = {
                metric: sums[metric] / len(rows) for metric in ROW_METRICS
            }
            transition_joint_hits[model][method] = int(sums["transition_joint"])
            per_model[model][method].update(
                {
                    "decision_macro_f1": macro_f1(
                        expected_decisions, predicted_decisions
                    ),
                    "operation_macro_f1": macro_f1(
                        expected_operations, predicted_operations
                    ),
                }
            )

    aggregate_path = artifact_root / "results/aggregates/main_results.csv"
    with aggregate_path.open(newline="", encoding="utf-8") as handle:
        aggregate_rows = {
            (str(row["model"]), str(row["method"])): row
            for row in csv.DictReader(handle)
        }
    aggregate_columns = {
        # Schema validity is intentionally excluded: the archived CSV used the
        # asymmetric partial MemPatch validator.  ``decision_macro_f1`` is also
        # excluded because the historical aggregate mislabeled macro recall
        # (balanced accuracy) as macro-F1.
        "decision_accuracy": "decision_accuracy",
        "exact": "exact_state_map",
        "state": "memory_state_accuracy",
        "evidence_f1": "evidence_f1",
        "diagnosis": "diagnosis_accuracy",
        "contamination": "downstream_contamination_rate",
    }
    for model in MODELS:
        for method in PREDICTION_METHODS:
            row = aggregate_rows.get((model, method))
            if row is None:
                raise RuntimeError(f"aggregate CSV missing {model}/{method}")
            for metric, column in aggregate_columns.items():
                observed = float(row[column])
                expected = per_model[model][method][metric]
                if abs(observed - expected) > 1e-12:
                    raise RuntimeError(
                        f"aggregate CSV mismatch for {model}/{method}/{column}: "
                        f"{observed} != {expected}"
                    )

    macro: dict[str, dict[str, float]] = {}
    displayed: dict[str, list[float]] = {}
    for method in METHODS:
        macro[method] = {
            metric: mean(per_model[model][method][metric] for model in MODELS)
            for metric in (*ROW_METRICS, "decision_macro_f1", "operation_macro_f1")
        }
        displayed[method] = [
            round(100.0 * macro[method][metric], 1)
            for metric in PRIMARY_TABLE_METRICS
        ]
        if tuple(displayed[method]) != PAPER_TABLE3[method]:
            raise RuntimeError(
                f"{method}: rebuilt display {displayed[method]} != paper {PAPER_TABLE3[method]}"
            )

    transition_joint_summary: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        hits = sum(transition_joint_hits[model][method] for model in MODELS)
        expected_hits = EXPECTED_TRANSITION_JOINT_HITS[method]
        if hits != expected_hits:
            raise RuntimeError(
                f"{method}: Transition-Joint hits {hits} != expected {expected_hits}"
            )
        transition_joint_summary[method] = {
            "hits": hits,
            "total": len(MODELS) * len(labels),
            "percent": round(100.0 * macro[method]["transition_joint"], 3),
            "per_model_hits": {
                model: transition_joint_hits[model][method] for model in MODELS
            },
        }

    return {
        "status": "PASS",
        "scope": "controlled main split; five-model unweighted macro",
        "frozen_hashes": observed_hashes,
        "models": list(MODELS),
        "methods": list(METHODS),
        "classification_labels": {
            "decision": decision_classes,
            "operation": operation_classes,
        },
        "metric_order": list(PRIMARY_TABLE_METRICS),
        "auxiliary_diagnosis_percent": {
            DISPLAY_NAMES[method]: round(100.0 * macro[method]["diagnosis"], 1)
            for method in METHODS
        },
        "diagnostic_metric_order": list(ROW_METRICS),
        "repair_counts": repair_counts,
        "prediction_sha256": prediction_hashes,
        "aggregate_csv_verified": {
            "path": "results/aggregates/main_results.csv",
            "cells": len(MODELS) * len(PREDICTION_METHODS),
            "metric_columns": aggregate_columns,
            "ignored_legacy_columns": {
                "schema_valid_rate": "used asymmetric validators for MemPatch and read paths",
                "decision_macro_f1": "historically contains macro recall / balanced accuracy",
            },
        },
        "scope_only_policy": {
            "main_scope_state": "current",
            "side_scope_state": "out_of_scope",
            "decision": "use_current_memory",
            "memory_operation": "REVISE",
            "evidence_event_ids": [],
            "failure_diagnosis": "stale_memory_reuse",
        },
        "transition_joint": {
            "definition": "decision exact AND operation exact AND complete state map exact AND evidence-id set exact",
            "note": "supplementary sparse diagnostic; distinct from legacy strict_joint",
            "by_method": {
                DISPLAY_NAMES[method]: transition_joint_summary[method]
                for method in METHODS
            },
        },
        "per_model": per_model,
        "macro": macro,
        "displayed_percent": {
            DISPLAY_NAMES[method]: displayed[method] for method in METHODS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_artifact = CODE_ROOT.parent
    parser.add_argument("--artifact-root", type=Path, default=default_artifact)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--materialize-effective",
        action="store_true",
        help="write canonical MemPatch rows with per-row repair_source",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(args.artifact_root.resolve(), args.materialize_effective)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
