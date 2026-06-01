#!/usr/bin/env python3
"""Diagnostic template-lookup baseline for ReTrace-Bench leakage analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.analyze_retrace_template_signatures import read_jsonl, scenario_signature


DECISIONS = (
    "use_current_memory",
    "escalate",
    "ask_clarification",
    "mark_unresolved",
    "refuse_due_to_policy",
)


def _gold_decision(row: dict[str, Any]) -> str:
    return row.get("hidden_gold", {}).get("expected_decision") or ""


def _gold_failure(row: dict[str, Any]) -> str:
    return row.get("primary_failure_mode") or row.get("hidden_gold", {}).get("expected_failure_diagnosis") or ""


def _macro_f1(expected: list[str], predicted: list[str]) -> float:
    classes = sorted(set(expected) | set(predicted))
    if not classes:
        return 0.0
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for exp, pred in zip(expected, predicted) if exp == cls and pred == cls)
        fp = sum(1 for exp, pred in zip(expected, predicted) if exp != cls and pred == cls)
        fn = sum(1 for exp, pred in zip(expected, predicted) if exp == cls and pred != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def build_lookup(train_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, str]], str, str]:
    by_sig: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"decision": Counter(), "failure": Counter()})
    majority_decision = Counter(_gold_decision(row) for row in train_rows).most_common(1)[0][0]
    majority_failure = Counter(_gold_failure(row) for row in train_rows).most_common(1)[0][0]
    for row in train_rows:
        sig = scenario_signature(row)
        by_sig[sig]["decision"][_gold_decision(row)] += 1
        by_sig[sig]["failure"][_gold_failure(row)] += 1
    lookup = {
        sig: {
            "decision": values["decision"].most_common(1)[0][0],
            "failure": values["failure"].most_common(1)[0][0],
        }
        for sig, values in by_sig.items()
    }
    return lookup, majority_decision, majority_failure


def evaluate(train_path: Path, test_path: Path) -> dict[str, Any]:
    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    lookup, majority_decision, majority_failure = build_lookup(train_rows)
    expected_decisions: list[str] = []
    predicted_decisions: list[str] = []
    expected_failures: list[str] = []
    predicted_failures: list[str] = []
    covered = 0
    examples: list[dict[str, Any]] = []

    for row in test_rows:
        sig = scenario_signature(row)
        hit = sig in lookup
        if hit:
            covered += 1
            pred_decision = lookup[sig]["decision"]
            pred_failure = lookup[sig]["failure"]
        else:
            pred_decision = majority_decision
            pred_failure = majority_failure
        exp_decision = _gold_decision(row)
        exp_failure = _gold_failure(row)
        expected_decisions.append(exp_decision)
        predicted_decisions.append(pred_decision)
        expected_failures.append(exp_failure)
        predicted_failures.append(pred_failure)
        if len(examples) < 12:
            examples.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "signature": sig,
                    "covered": hit,
                    "expected_decision": exp_decision,
                    "predicted_decision": pred_decision,
                    "expected_failure_mode": exp_failure,
                    "predicted_failure_mode": pred_failure,
                }
            )

    n = len(test_rows) or 1
    decision_correct = sum(1 for exp, pred in zip(expected_decisions, predicted_decisions) if exp == pred)
    failure_correct = sum(1 for exp, pred in zip(expected_failures, predicted_failures) if exp == pred)
    return {
        "train": str(train_path),
        "test": str(test_path),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "coverage_rate": covered / n,
        "covered_count": covered,
        "fallback_decision": majority_decision,
        "fallback_failure_mode": majority_failure,
        "decision_accuracy": decision_correct / n,
        "decision_macro_f1": _macro_f1(expected_decisions, predicted_decisions),
        "failure_mode_accuracy": failure_correct / n,
        "examples": examples,
    }


def write_report(result: dict[str, Any], out: Path) -> None:
    lines = [
        "# Template Lookup Diagnostic",
        "",
        "This is a diagnostic shortcut baseline, not a deployable memory baseline. It predicts from de-identified template signatures learned from the training split.",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| train scenarios | {result['train_count']} |",
        f"| test scenarios | {result['test_count']} |",
        f"| coverage rate | {result['coverage_rate']:.3f} |",
        f"| decision accuracy | {result['decision_accuracy']:.3f} |",
        f"| decision macro-F1 | {result['decision_macro_f1']:.3f} |",
        f"| failure-mode accuracy | {result['failure_mode_accuracy']:.3f} |",
        "",
        f"Fallback for unseen signatures: decision `{result['fallback_decision']}`, failure mode `{result['fallback_failure_mode']}`.",
        "",
        "## Example Predictions",
        "",
        "| scenario | covered | expected decision | predicted decision | expected failure | predicted failure |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for ex in result["examples"]:
        lines.append(
            f"| `{ex['scenario_id']}` | {str(ex['covered']).lower()} | `{ex['expected_decision']}` | "
            f"`{ex['predicted_decision']}` | `{ex['expected_failure_mode']}` | `{ex['predicted_failure_mode']}` |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = evaluate(Path(args.train), Path(args.test))
    write_report(result, Path(args.out))
    print(json.dumps({k: result[k] for k in ("coverage_rate", "decision_accuracy", "decision_macro_f1", "failure_mode_accuracy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
