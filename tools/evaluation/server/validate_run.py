"""Fail-fast validation for smoke and completed model runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools._root import bootstrap_from

bootstrap_from(__file__)

from mempatch.benchmark.api import evaluate_predictions, load_predictions, load_scenarios
from mempatch.benchmark.contracts import validate_prediction
from mempatch.benchmark.public_view import public_scenario_view

try:
    from .run_core import ALL_METHODS, BASELINE_METHODS
    from .methods import build_method_view
except ImportError:
    from run_core import ALL_METHODS, BASELINE_METHODS
    from methods import build_method_view

from mempatch.revision.runtime.dpa_runtime import run_from_text, parse_actions
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view
from mempatch.revision.runtime.benchmark_projection import project_to_benchmark_response
from mempatch.revision.runtime.ablation_projection import project_actions_without_dpa


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _classify_error(err: str) -> str:
    err_lower = err.lower()
    if "missing response field" in err_lower or "missing or empty response" in err_lower:
        return "Missing Field"
    if "invalid decision label" in err_lower or "invalid memory_state labels" in err_lower or "invalid failure_diagnosis label" in err_lower:
        return "Invalid Label"
    if "evidence_event_ids reference ids not in event_trace" in err_lower or "must be a list of event ids" in err_lower:
        return "Evidence Error"
    return "Other"


def _valid_response(response: dict[str, Any] | None) -> bool:
    return response is not None and not validate_prediction({"parsed": response})


def _repair_response(direct_response: dict[str, Any], projected_response: dict[str, Any]) -> dict[str, Any]:
    return direct_response if _valid_response(direct_response) else projected_response


def _prediction(row: dict[str, Any], final_name: str, legacy_name: str | None = None) -> dict[str, Any]:
    predictions = row.get("predictions", {})
    if final_name in predictions:
        return predictions[final_name]
    if legacy_name and legacy_name in predictions:
        return predictions[legacy_name]
    raise KeyError(final_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--allow-validation-errors", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_path = run_dir / "raw_cases.jsonl"
    if not raw_path.exists():
        print(f"Error: raw_cases.jsonl not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    raw_rows = _read_jsonl(raw_path)
    ids = [str(row["scenario_id"]) for row in raw_rows]
    if len(ids) != len(set(ids)):
        print("Error: raw_cases.jsonl contains duplicate scenario IDs", file=sys.stderr)
        sys.exit(1)
    if args.expected_cases is not None and len(raw_rows) != args.expected_cases:
        print(f"Error: expected {args.expected_cases} cases, found {len(raw_rows)}", file=sys.stderr)
        sys.exit(1)

    scenarios = load_scenarios(args.data)
    selected_ids = set(ids)
    selected = [row for row in scenarios if str(row["scenario_id"]) in selected_ids]
    if len(selected) != len(raw_rows):
        print("Error: raw outputs contain IDs absent from the dataset", file=sys.stderr)
        sys.exit(1)
        
    scenario_by_id = {str(s["scenario_id"]): s for s in selected}

    # Check pairing constraints and count generation parse errors.
    parse_errors_count = Counter()
    has_pairing_error = False

    for row in raw_rows:
        sid = str(row["scenario_id"])
        missing = set(ALL_METHODS) - set(row.get("predictions", {}))
        if missing:
            print(f"Error: {sid} missing methods {sorted(missing)} in raw predictions", file=sys.stderr)
            sys.exit(1)

        generations = row.get("generations", {})
        for method in BASELINE_METHODS:
            if generations.get(method, {}).get("parse_error") is not None:
                parse_errors_count[method] += 1
        
        shared_actions = generations.get("mempatch_shared_actions", {})
        if shared_actions.get("parse_error") is not None:
            parse_errors_count["mempatch"] += 1
            parse_errors_count["mempatch_noguard"] += 1

        scenario = scenario_by_id[sid]
        view = build_scenario_revision_view(scenario)
        public_view = public_scenario_view(scenario)

        guarded = row["predictions"]["mempatch"]
        unguarded = _prediction(row, "mempatch_noguard", "mempatch_no_guard")
        direct_response = _prediction(row, "direct_json", "frozen_direct")["response"]

        # MemPatch preserves valid direct_json responses and otherwise uses typed projection.
        # mempatch and mempatch_noguard share exactly the same proposed actions.
        actions_text = shared_actions.get("actions_text", "")
        parse_result = parse_actions(actions_text)
        runtime_result = run_from_text(view, actions_text)

        expected_mempatch_response = project_to_benchmark_response(
            runtime_result=runtime_result,
            raw_response=direct_response,
            scenario_public_view=public_view,
            fallback_answer=""
        )
        expected_mempatch_response = _repair_response(direct_response, expected_mempatch_response)
        if guarded.get("response") != expected_mempatch_response:
            print(f"Pairing Error: {sid} - MemPatch response does not match conservative repair rule", file=sys.stderr)
            has_pairing_error = True

        expected_no_guard_response = project_actions_without_dpa(
            view=view,
            parse_result=parse_result,
            raw_response=direct_response,
            scenario_public_view=public_view,
        )
        expected_no_guard_response = _repair_response(direct_response, expected_no_guard_response)
        if unguarded.get("response") != expected_no_guard_response:
            print(f"Pairing Error: {sid} - mempatch_noguard response does not match conservative repair rule", file=sys.stderr)
            has_pairing_error = True

    if has_pairing_error:
        print("Error: Fair pairing constraints validation failed", file=sys.stderr)
        sys.exit(1)

    # Read retrieval_k from the manifest.
    manifest_path = run_dir / "run_manifest.json"
    retrieval_k = 8
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            retrieval_k = manifest.get("retrieval_k", 8)
        except Exception:
            pass

    # Compute the average number of retained public evidence events.
    mean_events = {}
    for method in BASELINE_METHODS:
        total_evs = 0
        for scenario in selected:
            m_view = build_method_view(method, public_scenario_view(scenario), retrieval_k)
            total_evs += len(m_view.get("public_input", {}).get("event_trace", []))
        mean_events[method] = total_evs / len(selected)

    # MemPatch starts from direct_json and only falls back to typed projection.
    mean_events["mempatch"] = mean_events["direct_json"]
    mean_events["mempatch_noguard"] = mean_events["direct_json"]

    # Evaluate predictions and collect validation-error details.
    report = {
        "cases": len(raw_rows),
        "methods": {}
    }
    
    total_val_errors = 0
    total_missing_predictions = 0

    print("\n" + "="*50)
    print("MEMENTO RUN VALIDATION REPORT")
    print("="*50)

    for method in ALL_METHODS:
        path = run_dir / f"{method}.predictions.jsonl"
        predictions = load_predictions(path)
        result = evaluate_predictions(selected, predictions, strict=False, allow_missing=False)
        
        errors_list = result["errors"]
        missing_count = result["missing_prediction_count"]
        
        # Only accumulate errors from mempatch methods to trigger strict validation failures.
        # Baseline validation errors are recorded in the report but do not cause execution pipeline crashes.
        if method in ("mempatch", "mempatch_noguard"):
            total_val_errors += len(errors_list)
        total_missing_predictions += missing_count

        error_categories = Counter()
        for err in errors_list:
            error_categories[_classify_error(err)] += 1

        report["methods"][method] = {
            "errors": len(errors_list),
            "warnings": len(result["warnings"]),
            "generation_parse_errors": parse_errors_count[method],
            "mean_retained_events": mean_events[method],
            "error_classification": dict(error_categories),
            "headline_metrics": result["headline_metrics"],
        }

        print(f"\n[Method: {method}]")
        print(f"  Cases: {len(predictions)}")
        print(f"  Mean Retained Events: {mean_events[method]:.2f}")
        print(f"  Generation Parse Errors: {parse_errors_count[method]}")
        print(f"  Validation Errors: {len(errors_list)}")
        if error_categories:
            print("  Error Classification:")
            for cat, count in error_categories.items():
                print(f"    - {cat}: {count}")
        if errors_list:
            print("  Top 5 Validation Errors:")
            for err in errors_list[:5]:
                print(f"    * {err}")

    print("\n" + "="*50)
    print(f"SUMMARY TOTALS:")
    print(f"  Total Validation Errors: {total_val_errors}")
    print(f"  Total Missing Predictions: {total_missing_predictions}")
    print(f"  Total Generation Parse Errors: {sum(parse_errors_count.values())}")
    print("="*50)

    # Write validation_report.json.
    target = run_dir / "validation_report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport written to {target}")

    # Fail fast when strict validation is requested.
    if not args.allow_validation_errors:
        if total_val_errors > 0 or total_missing_predictions > 0 or sum(parse_errors_count.values()) > 0:
            print("\nValidation FAILED! Terminating with exit code 1 because strict mode is active.", file=sys.stderr)
            print("To bypass this, run with --allow-validation-errors (non-strict formatting audit only).", file=sys.stderr)
            sys.exit(1)
        else:
            print("\nValidation PASSED successfully in strict mode.")
    else:
        print("\nValidation check bypassed because --allow-validation-errors is enabled.")


if __name__ == "__main__":
    main()
