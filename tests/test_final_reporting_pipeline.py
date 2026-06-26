from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts.reporting.build_final_aggregate import build


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _score_row(*, scenario_id: str, split: str, method: str, parse_failure: bool = False) -> dict:
    return {
        "scenario_id": scenario_id,
        "split": split,
        "domain": "software_release",
        "difficulty": "medium",
        "failure_mode": "stale_memory_reuse",
        "method": method,
        "model": "m",
        "schema_valid": not parse_failure,
        "decision_correct": not parse_failure,
        "decision_f1_class": "use_current_memory",
        "memory_operation_f1_class": "REVISE",
        "exact_state_map": not parse_failure,
        "memory_state_accuracy": 0.0 if parse_failure else 1.0,
        "evidence_precision": 0.0 if parse_failure else 1.0,
        "evidence_recall": 0.0 if parse_failure else 1.0,
        "evidence_f1": 0.0 if parse_failure else 1.0,
        "diagnosis_correct": not parse_failure,
        "strict_joint": not parse_failure,
        "unsafe_reuse": False,
        "downstream_contamination": False,
        "parse_failure": parse_failure,
    }


def _prediction_row(*, scenario_id: str, split: str, method: str, input_tokens: int = 10) -> dict:
    return {
        "scenario_id": scenario_id,
        "split": split,
        "model": "m",
        "method": method,
        "response": {},
        "raw_generation": {
            "input_tokens": input_tokens,
            "output_tokens": 5,
            "latency_seconds": 2.0,
        },
        "retrieved_event_count": 3,
    }


def _args(tmp_path: Path, *, allow_empty: bool = False, allow_partial: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        scores_root=[tmp_path / "scores"],
        predictions_root=[tmp_path / "predictions"],
        output_dir=tmp_path / "aggregates",
        expected_model=["m"],
        expected_method=["frozen_direct"],
        expected_split=["main_test_synthetic"],
        skip_cell=[],
        allow_empty=allow_empty,
        allow_partial=allow_partial,
        include_pending_cells=False,
        overwrite=False,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_aggregate_tiny_fixture_normalizes_alias_and_computes_tokens(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "scores" / "m.frozen_direct.main.scores.jsonl",
        [
            _score_row(scenario_id="a", split="main_test_synthetic", method="frozen_direct"),
            _score_row(scenario_id="b", split="main_test_synthetic", method="frozen_direct", parse_failure=True),
        ],
    )
    _write_jsonl(
        tmp_path / "predictions" / "m.frozen_direct.main.predictions.jsonl",
        [
            _prediction_row(scenario_id="a", split="main_test_synthetic", method="frozen_direct", input_tokens=10),
            _prediction_row(scenario_id="b", split="main_test_synthetic", method="frozen_direct", input_tokens=20),
        ],
    )

    status = build(_args(tmp_path))
    assert status["status"] == "complete"
    rows = _read_csv(tmp_path / "aggregates" / "per_model_method_split.csv")
    assert rows[0]["method"] == "direct_json"
    assert rows[0]["n"] == "2"
    assert float(rows[0]["parse_failure_rate"]) == 0.5
    assert float(rows[0]["total_tokens"]) == 20.0
    assert float(rows[0]["unsupported_or_hallucinated_evidence_rate"]) == 0.5
    cost = _read_csv(tmp_path / "aggregates" / "cost_latency.csv")
    assert "memory_size" in cost[0]
    assert "unsupported_or_hallucinated_evidence_rate" in cost[0]
    capabilities = _read_csv(tmp_path / "aggregates" / "per_capability.csv")
    assert capabilities[0]["capability"] == "update_handling"
    families = _read_csv(tmp_path / "aggregates" / "per_baseline_family.csv")
    assert families[0]["baseline_family"] == "no_memory_vanilla"


def test_empty_input_with_allow_empty_writes_pending_status(tmp_path: Path) -> None:
    args = _args(tmp_path, allow_empty=True, allow_partial=True)
    status = build(args)
    assert status["status"] == "pending"
    assert (tmp_path / "aggregates" / "aggregate_status.json").exists()


def test_missing_required_columns_fails_unless_allow_partial(tmp_path: Path) -> None:
    row = _score_row(scenario_id="a", split="main_test_synthetic", method="direct_json")
    row.pop("evidence_recall")
    _write_jsonl(tmp_path / "scores" / "bad.scores.jsonl", [row])
    with pytest.raises(RuntimeError, match="missing required score fields"):
        build(_args(tmp_path))
    status = build(_args(tmp_path, allow_partial=True))
    assert status["missing_required_fields"]


def test_dev_calibration_excluded_from_headline_outputs(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "scores" / "rows.scores.jsonl",
        [
            _score_row(scenario_id="a", split="main_test_synthetic", method="direct_json"),
            _score_row(scenario_id="b", split="dev_calibration", method="direct_json"),
        ],
    )
    args = _args(tmp_path, allow_partial=True)
    args.expected_split = ["main_test_synthetic", "dev_calibration"]
    status = build(args)
    assert status["status"] in {"complete", "partial"}
    main = _read_csv(tmp_path / "aggregates" / "main_results.csv")
    assert main
    assert all(row["split"] == "main_test_synthetic" for row in main)
    challenge = _read_csv(tmp_path / "aggregates" / "challenge_results.csv")
    assert challenge == []
