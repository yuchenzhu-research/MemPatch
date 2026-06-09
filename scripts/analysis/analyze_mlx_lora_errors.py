#!/usr/bin/env python3
"""Compare base vs LoRA MLX predictions with per-field error bucketing.

Scores two prediction files against the same scenario split, then reports:
- aggregate metric deltas
- per-field error counts (decision, memory_state, evidence, failure_diagnosis)
- transition buckets (base wrong -> LoRA fixed, base correct -> LoRA regressed)
- per expected_decision and primary_failure_mode breakdowns
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import REPO_ROOT, bootstrap_from

bootstrap_from(__file__)

from benchmark.api import evaluate_predictions, load_predictions, load_scenarios  # noqa: E402
from benchmark.general_taxonomy import canonical_hidden_gold_fields  # noqa: E402
from benchmark.scorers_general import decision_matches, normalize_failure_mode  # noqa: E402


FIELD_KEYS = ("decision", "memory_state", "evidence", "failure_diagnosis", "joint")


def _resolve_scenarios_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "scenarios.jsonl"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"no scenarios.jsonl in directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"scenarios file not found: {path}")
    return path


def _field_errors(scenario: dict[str, Any], response: dict[str, Any]) -> dict[str, bool]:
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    rubric = gold.get("rubric") or {}
    aliases = (
        gold.get("decision_aliases")
        or rubric.get("decision_aliases")
        or scenario.get("decision_aliases")
    )

    expected_state = gold["expected_memory_state"] or {}
    predicted_state = response.get("memory_state") or {}
    if not isinstance(predicted_state, dict):
        predicted_state = {}

    if expected_state:
        state_total = len(expected_state)
        state_correct = sum(
            1 for mid, status in expected_state.items() if predicted_state.get(mid) == status
        )
        memory_ok = state_correct == state_total
    else:
        memory_ok = True

    pred_ev = set(response.get("evidence_event_ids") or [])
    gold_ev = set(gold["expected_evidence_event_ids"])
    if not gold_ev and not pred_ev:
        evidence_ok = True
    elif not gold_ev:
        evidence_ok = len(pred_ev) == 0
    else:
        tp = len(pred_ev & gold_ev)
        if tp == 0:
            evidence_ok = False
        else:
            precision = tp / len(pred_ev) if pred_ev else 0.0
            recall = tp / len(gold_ev)
            f1 = 2 * precision * recall / (precision + recall)
            evidence_ok = f1 >= 1.0

    decision_ok = decision_matches(response.get("decision"), gold["expected_decision"], aliases)
    predicted_diag = normalize_failure_mode(response.get("failure_diagnosis"))
    diagnosis_ok = gold["expected_failure_diagnosis"] == predicted_diag

    return {
        "decision": decision_ok,
        "memory_state": memory_ok,
        "evidence": evidence_ok,
        "failure_diagnosis": diagnosis_ok,
        "joint": decision_ok and memory_ok and evidence_ok,
    }


def _memory_state_mismatches(
    scenario: dict[str, Any], response: dict[str, Any]
) -> list[dict[str, str]]:
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    expected_state = gold["expected_memory_state"] or {}
    predicted_state = response.get("memory_state") or {}
    if not isinstance(predicted_state, dict):
        predicted_state = {}
    mismatches: list[dict[str, str]] = []
    for mid, expected in expected_state.items():
        predicted = predicted_state.get(mid)
        if predicted != expected:
            mismatches.append(
                {
                    "memory_id": mid,
                    "expected": expected,
                    "predicted": str(predicted),
                }
            )
    return mismatches


def _case_record(
    scenario: dict[str, Any],
    base_response: dict[str, Any] | None,
    lora_response: dict[str, Any] | None,
    *,
    base_metrics: dict[str, float] | None,
    lora_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    gold = canonical_hidden_gold_fields(scenario.get("hidden_gold") or {})
    base_fields = _field_errors(scenario, base_response or {}) if base_response else {}
    lora_fields = _field_errors(scenario, lora_response or {}) if lora_response else {}

    def _failed(fields: dict[str, bool], key: str) -> bool:
        return bool(fields) and not fields.get(key, True)

    record: dict[str, Any] = {
        "scenario_id": scenario.get("scenario_id"),
        "expected_decision": gold["expected_decision"],
        "expected_failure_diagnosis": gold["expected_failure_diagnosis"],
        "primary_failure_mode": scenario.get("primary_failure_mode"),
        "pattern": scenario.get("pattern"),
        "base": {
            "response": base_response,
            "field_ok": base_fields,
            "metrics": base_metrics,
        },
        "lora": {
            "response": lora_response,
            "field_ok": lora_fields,
            "metrics": lora_metrics,
        },
        "transitions": {},
    }

    for key in FIELD_KEYS:
        base_ok = base_fields.get(key, False)
        lora_ok = lora_fields.get(key, False)
        if not base_ok and lora_ok:
            record["transitions"][key] = "fixed"
        elif base_ok and not lora_ok:
            record["transitions"][key] = "regressed"
        elif not base_ok and not lora_ok:
            record["transitions"][key] = "still_wrong"
        else:
            record["transitions"][key] = "still_correct"

    if lora_response and _failed(lora_fields, "memory_state"):
        record["lora"]["memory_state_mismatches"] = _memory_state_mismatches(
            scenario, lora_response
        )
    if base_response and _failed(base_fields, "memory_state"):
        record["base"]["memory_state_mismatches"] = _memory_state_mismatches(
            scenario, base_response
        )
    return record


def _index_scored(scored_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["scenario_id"]: row for row in scored_rows}


def _counter_for_field(
    cases: list[dict[str, Any]], side: str, field: str, *, ok: bool
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in cases:
        field_ok = case[side]["field_ok"].get(field, True)
        if field_ok != ok:
            continue
        bucket = case.get("expected_decision") or "<missing>"
        counts[bucket] += 1
    return counts


def _transition_counter(cases: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(case["transitions"].get(field, "unknown") for case in cases)


def _headline_delta(base_result: dict[str, Any], lora_result: dict[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, base_val in base_result["headline_metrics"].items():
        lora_val = lora_result["headline_metrics"].get(key)
        if isinstance(base_val, (int, float)) and isinstance(lora_val, (int, float)):
            delta[key] = float(lora_val) - float(base_val)
    return delta


def analyze(
    scenarios: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    lora_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    base_ids = {p.get("scenario_id") for p in base_predictions if p.get("scenario_id")}
    lora_ids = {p.get("scenario_id") for p in lora_predictions if p.get("scenario_id")}
    common_ids = base_ids & lora_ids
    if not common_ids:
        raise ValueError("base and LoRA predictions share no scenario_id values")

    scenario_by_id = {s["scenario_id"]: s for s in scenarios}
    missing = sorted(sid for sid in common_ids if sid not in scenario_by_id)
    if missing:
        raise ValueError(
            f"{len(missing)} prediction scenario_id(s) missing from scenarios file "
            f"(first: {missing[0]})"
        )

    subset = [scenario_by_id[sid] for sid in sorted(common_ids)]
    base_subset = [p for p in base_predictions if p.get("scenario_id") in common_ids]
    lora_subset = [p for p in lora_predictions if p.get("scenario_id") in common_ids]

    base_result = evaluate_predictions(
        subset,
        base_subset,
        strict=False,
        allow_missing=False,
    )
    lora_result = evaluate_predictions(
        subset,
        lora_subset,
        strict=False,
        allow_missing=False,
    )

    base_by_id = _index_scored(base_result["scored_predictions"])
    lora_by_id = _index_scored(lora_result["scored_predictions"])

    common_ids_sorted = sorted(common_ids)
    cases: list[dict[str, Any]] = []
    for sid in common_ids_sorted:
        scenario = scenario_by_id[sid]
        base_row = base_by_id[sid]
        lora_row = lora_by_id[sid]
        cases.append(
            _case_record(
                scenario,
                base_row.get("response"),
                lora_row.get("response"),
                base_metrics=base_row.get("metrics"),
                lora_metrics=lora_row.get("metrics"),
            )
        )

    by_decision: dict[str, dict[str, Any]] = {}
    for decision in sorted({c["expected_decision"] for c in cases}):
        subset = [c for c in cases if c["expected_decision"] == decision]
        by_decision[decision] = {
            "count": len(subset),
            "transitions": {field: dict(_transition_counter(subset, field)) for field in FIELD_KEYS},
            "base_field_errors": {
                field: len(subset) - sum(1 for c in subset if c["base"]["field_ok"].get(field, False))
                for field in FIELD_KEYS
            },
            "lora_field_errors": {
                field: len(subset) - sum(1 for c in subset if c["lora"]["field_ok"].get(field, False))
                for field in FIELD_KEYS
            },
        }

    by_failure_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({c["primary_failure_mode"] for c in cases}):
        subset = [c for c in cases if c["primary_failure_mode"] == mode]
        by_failure_mode[str(mode)] = {
            "count": len(subset),
            "base_diagnosis_errors": sum(
                1 for c in subset if not c["base"]["field_ok"].get("failure_diagnosis", False)
            ),
            "lora_diagnosis_errors": sum(
                1 for c in subset if not c["lora"]["field_ok"].get("failure_diagnosis", False)
            ),
            "transitions": dict(_transition_counter(subset, "failure_diagnosis")),
        }

    still_wrong_joint = [
        c
        for c in cases
        if c["transitions"].get("joint") == "still_wrong"
    ]
    fixed_joint = [c for c in cases if c["transitions"].get("joint") == "fixed"]
    regressed_joint = [c for c in cases if c["transitions"].get("joint") == "regressed"]

    return {
        "count": len(cases),
        "headline_delta": _headline_delta(base_result, lora_result),
        "base_headline_metrics": base_result["headline_metrics"],
        "lora_headline_metrics": lora_result["headline_metrics"],
        "field_transitions": {
            field: dict(_transition_counter(cases, field)) for field in FIELD_KEYS
        },
        "base_field_errors_by_decision": {
            field: dict(_counter_for_field(cases, "base", field, ok=False))
            for field in FIELD_KEYS
        },
        "lora_field_errors_by_decision": {
            field: dict(_counter_for_field(cases, "lora", field, ok=False))
            for field in FIELD_KEYS
        },
        "by_expected_decision": by_decision,
        "by_primary_failure_mode": by_failure_mode,
        "joint_fixed_cases": [c["scenario_id"] for c in fixed_joint],
        "joint_regressed_cases": [c["scenario_id"] for c in regressed_joint],
        "joint_still_wrong_cases": [c["scenario_id"] for c in still_wrong_joint],
        "cases": cases,
        "base_eval_errors": base_result["errors"],
        "lora_eval_errors": lora_result["errors"],
    }


def _print_section(title: str) -> None:
    print(f"\n== {title} ==")


def _print_counter(label: str, counter: dict[str, int]) -> None:
    print(f"{label}:")
    for key, value in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {key}: {value}")


def print_report(report: dict[str, Any], *, show_cases: int = 0) -> None:
    _print_section(f"Compared {report['count']} cases")
    print("Headline metric delta (LoRA - base):")
    for key, delta in sorted(report["headline_delta"].items()):
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<32} {sign}{delta:.3f}")

    _print_section("Field transitions (base -> LoRA)")
    for field, counter in report["field_transitions"].items():
        _print_counter(field, counter)

    _print_section("LoRA field errors by expected_decision")
    for field, counter in report["lora_field_errors_by_decision"].items():
        if any(counter.values()):
            _print_counter(field, counter)

    _print_section("By primary_failure_mode (diagnosis focus)")
    for mode, payload in report["by_primary_failure_mode"].items():
        print(
            f"{mode}: n={payload['count']} "
            f"base_diag_err={payload['base_diagnosis_errors']} "
            f"lora_diag_err={payload['lora_diagnosis_errors']} "
            f"transitions={payload['transitions']}"
        )

    _print_section("Joint revision transitions")
    print(f"fixed: {len(report['joint_fixed_cases'])}")
    print(f"regressed: {len(report['joint_regressed_cases'])}")
    print(f"still_wrong: {len(report['joint_still_wrong_cases'])}")

    weak_decisions = sorted(
        report["by_expected_decision"].items(),
        key=lambda item: item[1]["lora_field_errors"].get("joint", 0),
        reverse=True,
    )
    _print_section("Weakest expected_decision buckets (LoRA joint errors)")
    for decision, payload in weak_decisions[:5]:
        print(
            f"{decision}: joint_errors={payload['lora_field_errors']['joint']} "
            f"decision_errors={payload['lora_field_errors']['decision']} "
            f"diag_errors={payload['lora_field_errors']['failure_diagnosis']}"
        )

    if report["lora_eval_errors"]:
        _print_section("LoRA validation errors (invalid labels)")
        for err in report["lora_eval_errors"][:20]:
            print(f"  - {err}")
        if len(report["lora_eval_errors"]) > 20:
            print(f"  ... and {len(report['lora_eval_errors']) - 20} more")

    if show_cases > 0:
        _print_section(f"Sample still-wrong joint cases (first {show_cases})")
        still_wrong = [
            c for c in report["cases"] if c["transitions"].get("joint") == "still_wrong"
        ]
        for case in still_wrong[:show_cases]:
            print(f"- {case['scenario_id']} expected={case['expected_decision']} "
                  f"pattern={case['pattern']}")
            print(
                f"  base decision={case['base']['response'].get('decision')} "
                f"diag={case['base']['response'].get('failure_diagnosis')}"
            )
            print(
                f"  lora decision={case['lora']['response'].get('decision')} "
                f"diag={case['lora']['response'].get('failure_diagnosis')}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "hf_release/mempatch/validation/scenarios.jsonl",
        help="Scenario JSONL or directory containing scenarios.jsonl.",
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=root / "local/results/qwen3_14b_base_mempatch_v13_smoke_valid_predictions.jsonl",
    )
    parser.add_argument(
        "--lora-predictions",
        type=Path,
        default=root / "local/results/qwen3_14b_mempatch_v13_smoke_valid_predictions.jsonl",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=root / "local/results/qwen3_14b_mempatch_v13_smoke_valid_error_analysis.json",
    )
    parser.add_argument(
        "--show-cases",
        type=int,
        default=5,
        help="Print this many still-wrong joint examples (0 to skip).",
    )
    parser.add_argument(
        "--include-case-details",
        action="store_true",
        help="Keep full per-case payloads in the JSON output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios_path = _resolve_scenarios_path(args.data)
    scenarios = load_scenarios(scenarios_path)
    base_predictions = load_predictions(args.base_predictions)
    lora_predictions = load_predictions(args.lora_predictions)

    report = analyze(scenarios, base_predictions, lora_predictions)
    print_report(report, show_cases=args.show_cases)

    payload = dict(report)
    if not args.include_case_details:
        payload.pop("cases", None)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote analysis -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
