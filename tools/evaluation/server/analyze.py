"""Strict scoring, paired cluster bootstrap, and paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools._root import bootstrap_from

bootstrap_from(__file__)

from mempatch.benchmark.api import evaluate_predictions, load_predictions, load_scenarios
from mempatch.benchmark.method_names import FINAL_METHODS, normalize_method_name

METHODS = FINAL_METHODS
LEGACY_BY_FINAL = {
    "direct_json": "frozen_direct",
    "full_context_json": "full_context",
    "summary_memory_json": "summary_memory",
    "bm25_rag_json": "lexical_rag",
    "time_aware_rag_json": "time_aware_rag",
    "mempatch_noguard": "mempatch_no_guard",
    "mempatch": "mempatch",
}
BOOTSTRAP_METRICS = (
    "joint_revision_success",
    "black_box_decision_accuracy",
    "memory_operation_accuracy",
    "memory_state_accuracy",
    "evidence_f1",
    "failure_diagnosis_accuracy",
    "followup_task_accuracy",
    "downstream_contamination_rate",
    "stale_reuse_rate",
    "scenario_exact_state_match",
)


def _gold_state(scenario: dict[str, Any]) -> dict[str, str]:
    gold = scenario.get("hidden_gold") or {}
    for key in ("expected_memory_state", "memory_state", "gold_memory_state"):
        value = gold.get(key)
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
    rubric = gold.get("rubric") or {}
    for key in ("expected_memory_state", "memory_state"):
        value = rubric.get(key)
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
    raise KeyError(f"No gold memory state found for {scenario.get('scenario_id')}")


def _prediction_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["scenario_id"]): row.get("response", row) for row in rows}


def _state_scores(
    scenarios: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels: set[str] = set()
    confusion: Counter[tuple[str, str]] = Counter()
    changed_correct = 0
    changed_total = 0
    exact = 0
    per_scenario: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        sid = str(scenario["scenario_id"])
        gold = _gold_state(scenario)
        pred = predictions.get(sid, {}).get("memory_state") or {}
        exact_case = True
        correct = 0
        for memory_id, gold_label in gold.items():
            pred_label = str(pred.get(memory_id, "__missing__"))
            labels.add(gold_label)
            confusion[(gold_label, pred_label)] += 1
            is_correct = pred_label == gold_label
            correct += int(is_correct)
            exact_case = exact_case and is_correct
            if gold_label != "current":
                changed_total += 1
                changed_correct += int(is_correct)
        exact += int(exact_case)
        per_scenario[sid] = {
            "memory_state_accuracy": correct / max(len(gold), 1),
            "scenario_exact_state_match": float(exact_case),
        }
    f1s = []
    for label in sorted(labels):
        tp = confusion[(label, label)]
        fp = sum(value for (gold, pred), value in confusion.items() if pred == label and gold != label)
        fn = sum(value for (gold, pred), value in confusion.items() if gold == label and pred != label)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "state_macro_f1": sum(f1s) / max(len(f1s), 1),
        "changed_record_accuracy": changed_correct / max(changed_total, 1),
        "scenario_exact_state_match": exact / max(len(scenarios), 1),
        "per_scenario": per_scenario,
    }


def _cluster(scenario: dict[str, Any], key: str) -> str:
    metadata = scenario.get("metadata") or {}
    return str(metadata.get(key) or scenario.get(key) or scenario.get("domain") or "unknown")


def _bootstrap_delta(
    scenarios: list[dict[str, Any]],
    left: dict[str, float],
    right: dict[str, float],
    cluster_key: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    groups: dict[str, list[str]] = defaultdict(list)
    for scenario in scenarios:
        sid = str(scenario["scenario_id"])
        if sid in left and sid in right:
            groups[_cluster(scenario, cluster_key)].append(sid)
    cluster_ids = sorted(groups)
    observed = sum(left[sid] - right[sid] for ids in groups.values() for sid in ids)
    observed /= max(sum(len(ids) for ids in groups.values()), 1)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        drawn = [rng.choice(cluster_ids) for _ in cluster_ids]
        ids = [sid for cluster_id in drawn for sid in groups[cluster_id]]
        samples.append(sum(left[sid] - right[sid] for sid in ids) / max(len(ids), 1))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    p_two_sided = min(
        1.0,
        2 * min(
            sum(value <= 0 for value in samples) / len(samples),
            sum(value >= 0 for value in samples) / len(samples),
        ),
    )
    return {"delta": observed, "ci_low": lo, "ci_high": hi, "p": p_two_sided}


def _cluster_wtl(
    scenarios: list[dict[str, Any]],
    left: dict[str, float],
    right: dict[str, float],
    cluster_key: str,
    tolerance: float = 1e-12,
) -> dict[str, int]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for scenario in scenarios:
        sid = str(scenario["scenario_id"])
        if sid in left and sid in right:
            grouped[_cluster(scenario, cluster_key)].append(sid)
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for ids in grouped.values():
        delta = sum(left[sid] - right[sid] for sid in ids) / max(len(ids), 1)
        if delta > tolerance:
            counts["wins"] += 1
        elif delta < -tolerance:
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    return counts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _generation_record(generations: dict[str, Any], method: str) -> dict[str, Any]:
    final = normalize_method_name(method)
    if final in generations:
        return generations[final]
    legacy = LEGACY_BY_FINAL.get(final)
    if legacy and legacy in generations:
        return generations[legacy]
    raise KeyError(method)


def _efficiency_rows(model: str, raw_path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_jsonl(raw_path)
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in raw_rows:
        generations = row["generations"]
        for method in (
            "direct_json",
            "full_context_json",
            "bm25_rag_json",
            "dense_rag_json",
            "time_aware_rag_json",
            "summary_memory_json",
        ):
            try:
                record = _generation_record(generations, method)
            except KeyError:
                continue
            totals[method]["input_tokens"] += record["input_tokens"]
            totals[method]["output_tokens"] += record["output_tokens"]
            totals[method]["latency_seconds"] += record["latency_seconds"]
            totals[method]["cases"] += 1
        action = generations["mempatch_shared_actions"]
        projection = generations.get("deterministic_projection", {})
        for method in ("mempatch", "mempatch_noguard"):
            direct = _generation_record(generations, "direct_json")
            totals[method]["input_tokens"] += direct["input_tokens"] + action["input_tokens"]
            totals[method]["output_tokens"] += direct["output_tokens"] + action["output_tokens"]
            totals[method]["latency_seconds"] += (
                direct["latency_seconds"]
                + action["latency_seconds"]
                + projection.get(f"{method}_latency_seconds", 0.0)
            )
            totals[method]["cases"] += 1
    result = []
    for method, values in totals.items():
        cases = max(values["cases"], 1)
        result.append(
            {
                "model": model,
                "method": method,
                "mean_input_tokens": values["input_tokens"] / cases,
                "mean_output_tokens": values["output_tokens"] / cases,
                "mean_latency_seconds": values["latency_seconds"] / cases,
            }
        )
    return result


def _funnel_row(model: str, raw_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(raw_path)
    totals: Counter[str] = Counter()
    evtf_sum = 0.0
    for row in rows:
        audit = row["predictions"]["mempatch"].get("dpa_audit") or {}
        parse = audit.get("parse_result") or {}
        totals["cases"] += 1
        totals["valid_json"] += int(bool(parse.get("valid_json")))
        totals["schema_valid"] += int(bool(parse.get("schema_valid")))
        totals["proposed_actions"] += int(parse.get("n_actions") or 0)
        totals["admitted_actions"] += len(audit.get("admitted_actions") or [])
        totals["rejected_actions"] += len(audit.get("rejected_actions") or [])
        totals["engine_errors"] += len(audit.get("engine_errors") or [])
        evtf_sum += float(audit.get("evtf", 0.0))
    cases = max(totals["cases"], 1)
    proposed = max(totals["proposed_actions"], 1)
    return {
        "model": model,
        "cases": totals["cases"],
        "valid_json_rate": totals["valid_json"] / cases,
        "schema_valid_rate": totals["schema_valid"] / cases,
        "proposed_actions": totals["proposed_actions"],
        "admitted_action_rate": totals["admitted_actions"] / proposed,
        "rejected_action_rate": totals["rejected_actions"] / proposed,
        "mean_engine_errors": totals["engine_errors"] / cases,
        "mean_evtf": evtf_sum / cases,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", default="paper_results")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--cluster-key", default="decision_variant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-validation-errors", action="store_true")
    args = parser.parse_args()

    scenarios = load_scenarios(args.data)
    output = Path(args.output)
    summary_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []

    for model in args.models:
        method_details: dict[str, dict[str, Any]] = {}
        raw_path = Path(args.runs_root) / model / "raw_cases.jsonl"
        efficiency_rows.extend(_efficiency_rows(model, raw_path))
        funnel_rows.append(_funnel_row(model, raw_path))
        for method in METHODS:
            path = Path(args.runs_root) / model / f"{method}.predictions.jsonl"
            if not path.exists() and (legacy := LEGACY_BY_FINAL.get(method)):
                path = Path(args.runs_root) / model / f"{legacy}.predictions.jsonl"
            predictions = load_predictions(path)

            # 严格验证长度
            if len(predictions) != len(scenarios):
                print(f"Error: {model}/{method} prediction count mismatch. Expected {len(scenarios)}, found {len(predictions)}", file=sys.stderr)
                sys.exit(1)
            
            # 严格验证 scenario_id 顺序和去重
            pred_ids = [str(p.get("scenario_id", "")) for p in predictions]
            gold_ids = [str(s["scenario_id"]) for s in scenarios]
            
            if len(pred_ids) != len(set(pred_ids)):
                print(f"Error: {model}/{method} contains duplicate scenario IDs", file=sys.stderr)
                sys.exit(1)
            if pred_ids != gold_ids:
                print(f"Error: {model}/{method} scenario ID order mismatch with dataset", file=sys.stderr)
                sys.exit(1)

            official = evaluate_predictions(scenarios, predictions, strict=False)
            if official["missing_prediction_count"]:
                print(f"Error: {model}/{method} is incomplete with {official['missing_prediction_count']} missing predictions", file=sys.stderr)
                sys.exit(1)
            
            if method in ("mempatch", "mempatch_no_guard") and len(official["errors"]) > 0 and not args.allow_validation_errors:
                print(f"Error: {model}/{method} has validation errors in strict mode:", file=sys.stderr)
                for err in official["errors"][:5]:
                    print(f"  * {err}", file=sys.stderr)
                sys.exit(1)

            state = _state_scores(scenarios, _prediction_map(predictions))
            summary_rows.append(
                {
                    "model": model,
                    "method": method,
                    **official["headline_metrics"],
                    "state_macro_f1": state["state_macro_f1"],
                    "changed_record_accuracy": state["changed_record_accuracy"],
                    "scenario_exact_state_match": state["scenario_exact_state_match"],
                    "validation_error_count": len(official["errors"]),
                }
            )
            method_details[method] = {
                "per_scenario": {
                    str(scored["scenario_id"]): {
                        **scored["metrics"],
                        "scenario_exact_state_match": state["per_scenario"][
                            str(scored["scenario_id"])
                        ]["scenario_exact_state_match"],
                    }
                    for scored in official["scored_predictions"]
                }
            }

        mempatch = method_details["mempatch"]["per_scenario"]
        for baseline in METHODS:
            if baseline == "mempatch":
                continue
            baseline_scores = method_details[baseline]["per_scenario"]
            for metric in BOOTSTRAP_METRICS:
                left = {
                    str(s["scenario_id"]): float(mempatch[str(s["scenario_id"])][metric])
                    for s in scenarios
                }
                right = {
                    str(s["scenario_id"]): float(baseline_scores[str(s["scenario_id"])][metric])
                    for s in scenarios
                }
                stats = _bootstrap_delta(
                    scenarios, left, right, args.cluster_key, args.bootstrap, args.seed
                )
                significance_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "comparison": f"mempatch - {baseline}",
                        **stats,
                        **_cluster_wtl(scenarios, left, right, args.cluster_key),
                    }
                )

    _write_csv(output / "main_results.csv", summary_rows)
    _write_csv(output / "paired_cluster_bootstrap.csv", significance_rows)
    _write_csv(output / "efficiency.csv", efficiency_rows)
    _write_csv(output / "interface_funnel.csv", funnel_rows)
    (output / "analysis.json").write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "paired_cluster_bootstrap": significance_rows,
                "efficiency": efficiency_rows,
                "interface_funnel": funnel_rows,
                "bootstrap_replicates": args.bootstrap,
                "cluster_key": args.cluster_key,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
