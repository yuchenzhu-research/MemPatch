#!/usr/bin/env python3
"""Build hard50 summary metrics, markdown table, and manual inspection sample."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import canonical_hidden_gold_fields
from benchmark.retrace_bench.scorers_general import HEADLINE_METRICS, aggregate_metrics, score_prediction
from scripts.run_retrace_bench_baseline import read_jsonl


REPORT_METRICS = (
    "decision_macro_f1",
    "black_box_decision_accuracy",
    "non_answer_decision_accuracy",
    "memory_state_accuracy",
    "evidence_f1",
    "minimal_evidence_exact_match",
    "evidence_precision",
    "overcitation_rate",
    "counterevidence_recall",
    "failure_diagnosis_accuracy",
    "stale_reuse_rate",
    "latest_event_shortcut_failure_rate",
    "answer_state_consistency",
    "joint_revision_success",
    "format_failure_rate",
)

API_MODELS = (
    "Pro/moonshotai/Kimi-K2.6",
    "Pro/zai-org/GLM-5.1",
    "deepseek-ai/DeepSeek-V4-Pro",
)


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("report_metrics") or payload.get("all_metrics") or payload.get("metrics") or {}
    return {k: float(metrics[k]) for k in REPORT_METRICS if k in metrics}


def model_slug(model: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "__", model.strip()).strip("_")


def short_model_name(model: str) -> str:
    return model.split("/")[-1]


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {row["scenario_id"]: row for row in read_jsonl(path)}


def event_summary(scenario: dict[str, Any], limit: int = 5) -> str:
    events = scenario.get("public_input", {}).get("event_trace", [])
    lines = []
    for ev in events[:limit]:
        text = str(ev.get("text", "")).strip()
        if len(text) > 120:
            text = text[:117] + "..."
        lines.append(f"- `{ev.get('event_id')}` ({ev.get('actor_role')}): {text}")
    if len(events) > limit:
        lines.append(f"- ... and {len(events) - limit} more events")
    return "\n".join(lines)


def infer_pattern(scenario: dict[str, Any]) -> str:
    meta = scenario.get("metadata", {})
    factors = scenario.get("difficulty_factors", {})
    for key in (
        "version_or_release_chain",
        "scope_collision",
        "authority_conflict",
        "negative_evidence_required",
        "multi_memory_coupling",
        "policy_or_security_constraint",
    ):
        if factors.get(key):
            return key
    return scenario.get("primary_failure_mode", "unknown")


def pick_manual_cases(
    scenarios: dict[str, dict[str, Any]],
    api_preds: dict[str, dict[str, dict[str, Any]]],
    retrieve_preds: dict[str, dict[str, Any]],
    rng: random.Random,
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    used: set[str] = set()

    def add(sid: str, reason: str) -> None:
        if sid in used or sid not in scenarios:
            return
        selected.append((sid, reason))
        used.add(sid)

    all_fail = []
    mixed = []
    overcites = []
    for sid, scenario in scenarios.items():
        api_metrics = []
        for model in API_MODELS:
            pred = api_preds.get(model, {}).get(sid)
            if pred:
                api_metrics.append(float(pred.get("metrics", {}).get("joint_revision_success", 0.0)))
        if len(api_metrics) == 3 and all(v < 1.0 for v in api_metrics):
            all_fail.append(sid)
        if len(api_metrics) == 3 and any(v >= 1.0 for v in api_metrics) and not all(v >= 1.0 for v in api_metrics):
            mixed.append(sid)
        ret = retrieve_preds.get(sid)
        if ret and float(ret.get("metrics", {}).get("overcitation_rate", 0.0)) > 0.5:
            overcites.append(sid)

    for sid in all_fail[:3]:
        add(sid, "all three API models fail joint_revision_success")
    for sid in mixed[:3]:
        add(sid, "one API model succeeds while others fail joint_revision_success")
    for sid in overcites[:2]:
        add(sid, "retrieve_all overcites evidence")
    pool = [sid for sid in scenarios if sid not in used]
    rng.shuffle(pool)
    for sid in pool[: max(0, 10 - len(selected))]:
        add(sid, "random sample")
    return selected[:10]


def evaluate_recommendation(method_metrics: dict[str, dict[str, float]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    latest = method_metrics.get("latest_only", {})
    retrieve = method_metrics.get("retrieve_all", {})
    api_rows = {k: v for k, v in method_metrics.items() if k not in {"latest_only", "retrieve_all", "gold_oracle"}}

    continue_signals = 0
    fallback_signals = 0

    if latest.get("joint_revision_success", 1.0) <= 0.25:
        continue_signals += 1
        reasons.append("latest_only is weak on joint_revision_success as expected")
    else:
        fallback_signals += 1
        reasons.append("latest_only joint_revision_success is unexpectedly high")

    if retrieve.get("overcitation_rate", 0.0) >= 0.5:
        continue_signals += 1
        reasons.append("retrieve_all shows high overcitation_rate")
    else:
        reasons.append("retrieve_all overcitation_rate is lower than expected")

    strong_api = []
    for name, metrics in api_rows.items():
        if metrics.get("decision_macro_f1", 1.0) < 0.90 or metrics.get("joint_revision_success", 1.0) < 0.60:
            strong_api.append(name)
    if strong_api:
        continue_signals += 1
        reasons.append(f"strong API models still struggle: {', '.join(strong_api)}")
    else:
        fallback_signals += 1
        reasons.append("all API models look too strong on joint metrics")

    diag_vals = [m.get("failure_diagnosis_accuracy", 0.0) for m in api_rows.values()]
    if diag_vals and max(diag_vals) <= 0.85:
        continue_signals += 1
        reasons.append("failure_diagnosis_accuracy is not trivially perfect")
    elif diag_vals and min(diag_vals) >= 0.90:
        fallback_signals += 1
        reasons.append("failure_diagnosis_accuracy looks too easy")

    fmt_vals = [m.get("format_failure_rate", 0.0) for m in api_rows.values()]
    if fmt_vals and max(fmt_vals) > 0.05:
        fallback_signals += 1
        reasons.append("format/schema instability remains in API outputs")

    if not api_rows:
        return (
            "fallback_to_v1_0",
            reasons + ["API model runs were not available; cannot justify continuing final-hardening tonight"],
        )

    if fallback_signals > continue_signals:
        return "fallback_to_v1_0", reasons
    return "continue_final_hardening", reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="outputs/retrace_bench_hard50")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    api_dir = out_dir / "api_models"
    scenarios_list = read_jsonl(Path(args.data))
    scenarios = {row["scenario_id"]: row for row in scenarios_list}

    method_metrics: dict[str, dict[str, float]] = {}
    for baseline in ("latest_only", "retrieve_all"):
        metrics_path = out_dir / f"{baseline}.predictions.metrics.json"
        if metrics_path.exists():
            method_metrics[baseline] = load_metrics(metrics_path)

    gold_path = out_dir / "gold_oracle.metrics.json"
    if gold_path.exists():
        method_metrics["gold_oracle"] = load_metrics(gold_path)

    api_preds: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in API_MODELS}
    for model in API_MODELS:
        pred_path = api_dir / f"{model_slug(model)}.predictions.jsonl"
        metrics_path = api_dir / f"{model_slug(model)}.predictions.metrics.json"
        if metrics_path.exists():
            method_metrics[short_model_name(model)] = load_metrics(metrics_path)
        if pred_path.exists():
            api_preds[model] = load_predictions(pred_path)

    retrieve_preds = load_predictions(out_dir / "retrieve_all.predictions.jsonl") if (out_dir / "retrieve_all.predictions.jsonl").exists() else {}

    summary = {
        "count": len(scenarios_list),
        "methods": method_metrics,
        "report_metrics": list(REPORT_METRICS),
    }
    recommendation, reasons = evaluate_recommendation(method_metrics)
    summary["recommendation"] = recommendation
    summary["recommendation_reasons"] = reasons
    (out_dir / "summary.metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ReTrace-Bench Hard50 Summary",
        "",
        f"Dataset: `{args.data}`",
        f"Scenarios: {len(scenarios_list)}",
        "",
        "## Metrics",
        "",
        "| method | " + " | ".join(REPORT_METRICS) + " |",
        "| --- | " + " | ".join(["---"] * len(REPORT_METRICS)) + " |",
    ]
    for method, metrics in method_metrics.items():
        if method == "gold_oracle":
            continue
        row = " | ".join(f"{metrics.get(k, float('nan')):.3f}" for k in REPORT_METRICS)
        lines.append(f"| {method} | {row} |")
    lines.extend(["", f"## Recommendation", "", f"**{recommendation}**", ""])
    for reason in reasons:
        lines.append(f"- {reason}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rng = random.Random(args.seed)
    picks = pick_manual_cases(scenarios, api_preds, retrieve_preds, rng)
    inspect_lines = ["# Hard50 Manual Inspection (10 cases)", ""]
    for sid, reason in picks:
        scenario = scenarios[sid]
        gold = canonical_hidden_gold_fields(scenario["hidden_gold"])
        inspect_lines.extend(
            [
                f"## {sid}",
                "",
                f"- **Selection reason:** {reason}",
                f"- **Pattern / failure / difficulty:** {infer_pattern(scenario)} / {scenario.get('primary_failure_mode')} / {scenario.get('difficulty')}",
                f"- **Expected decision:** `{gold['expected_decision']}`",
                f"- **Expected evidence:** `{gold['expected_evidence_event_ids']}`",
                "",
                "### Event trace summary",
                event_summary(scenario),
                "",
                "### Model outputs",
            ]
        )
        for model in API_MODELS:
            pred = api_preds.get(model, {}).get(sid, {}).get("response", {})
            inspect_lines.append(
                f"- **{short_model_name(model)}:** decision=`{pred.get('decision')}` "
                f"evidence=`{pred.get('evidence_event_ids')}` diagnosis=`{pred.get('failure_diagnosis')}`"
            )
        ret = retrieve_preds.get(sid, {}).get("response", {})
        inspect_lines.append(
            f"- **retrieve_all:** decision=`{ret.get('decision')}` evidence=`{ret.get('evidence_event_ids')}` "
            f"overcitation={retrieve_preds.get(sid, {}).get('metrics', {}).get('overcitation_rate')}"
        )
        inspect_lines.extend(
            [
                "",
                "**Why hard:** Requires multi-event reasoning over verified/trusted records; "
                "latest-event or retrieve-all shortcuts should not satisfy minimal evidence and joint metrics.",
                "",
            ]
        )
    (out_dir / "manual_inspection_10.md").write_text("\n".join(inspect_lines) + "\n", encoding="utf-8")
    print(json.dumps({"recommendation": recommendation, "summary": str(out_dir / "summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
