#!/usr/bin/env python3
"""Build hard scale-test summary, distributions, and manual inspection sample."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.general_taxonomy import PATTERNS, canonical_hidden_gold_fields
from benchmark.retrace_bench.generation.pattern_spec import PATTERN_SPEC
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

DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Pro"


def model_slug(model: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "__", model.strip()).strip("_")


def short_model_name(model: str) -> str:
    return model.split("/")[-1]


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("report_metrics") or payload.get("all_metrics") or payload.get("metrics") or {}
    return {k: float(metrics[k]) for k in REPORT_METRICS if k in metrics}


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


def compute_distributions(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    pattern = Counter()
    failure_mode = Counter()
    expected_decision = Counter()
    difficulty = Counter()
    source_type = Counter()
    event_counts: list[int] = []
    bg_event_counts: list[int] = []

    for row in scenarios:
        pattern[row.get("pattern") or row.get("metadata", {}).get("pattern", "unknown")] += 1
        failure_mode[row.get("primary_failure_mode", "unknown")] += 1
        gold = canonical_hidden_gold_fields(row["hidden_gold"])
        expected_decision[gold["expected_decision"]] += 1
        difficulty[row.get("difficulty") or row.get("difficulty_level", "unknown")] += 1
        source_type[row.get("source_type", "unknown")] += 1
        events = row.get("public_input", {}).get("event_trace", [])
        event_counts.append(len(events))
        bg_event_counts.append(sum(1 for ev in events if "-bg-" in str(ev.get("event_id", ""))))

    pattern_counts = [pattern[p] for p in PATTERNS if p in pattern]
    pattern_min = min(pattern_counts) if pattern_counts else 0
    pattern_max = max(pattern_counts) if pattern_counts else 0
    decision_total = sum(expected_decision.values()) or 1
    top_decision_share = max(expected_decision.values()) / decision_total if expected_decision else 0.0

    return {
        "pattern": dict(sorted(pattern.items())),
        "failure_mode": dict(sorted(failure_mode.items())),
        "expected_decision": dict(sorted(expected_decision.items())),
        "difficulty": dict(sorted(difficulty.items())),
        "source_type": dict(sorted(source_type.items())),
        "pattern_balance": {
            "patterns_present": len([p for p in PATTERNS if pattern.get(p, 0) > 0]),
            "expected_patterns": len(PATTERNS),
            "min_count": pattern_min,
            "max_count": pattern_max,
            "spread": pattern_max - pattern_min,
        },
        "decision_skew": {
            "top_decision": max(expected_decision, key=expected_decision.get) if expected_decision else None,
            "top_decision_share": round(top_decision_share, 4),
        },
        "filler_stats": {
            "avg_event_count": round(sum(event_counts) / len(event_counts), 2) if event_counts else 0.0,
            "avg_background_events": round(sum(bg_event_counts) / len(bg_event_counts), 2) if bg_event_counts else 0.0,
            "scenarios_with_background_fill": sum(1 for n in bg_event_counts if n > 0),
        },
    }


def semantic_alignment_notes(distributions: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    pattern_balance = distributions["pattern_balance"]
    if pattern_balance["spread"] <= 1:
        notes.append(
            f"Pattern counts are approximately balanced ({pattern_balance['min_count']}-"
            f"{pattern_balance['max_count']} per pattern across {pattern_balance['patterns_present']} patterns)."
        )
    else:
        notes.append(
            f"Pattern counts show moderate imbalance (spread={pattern_balance['spread']}; "
            f"min={pattern_balance['min_count']}, max={pattern_balance['max_count']})."
        )

    skew = distributions["decision_skew"]
    if skew["top_decision_share"] >= 0.80:
        notes.append(
            f"Expected decision distribution is skewed: `{skew['top_decision']}` accounts for "
            f"{100 * skew['top_decision_share']:.1f}% of cases."
        )
    elif skew["top_decision_share"] >= 0.65:
        notes.append(
            f"Expected decision distribution is moderately skewed toward `{skew['top_decision']}` "
            f"({100 * skew['top_decision_share']:.1f}%)."
        )
    else:
        notes.append("Expected decision distribution is reasonably mixed across decision types.")

    filler = distributions["filler_stats"]
    if filler["avg_background_events"] >= 2.5:
        notes.append(
            f"Event traces look filler-heavy (avg {filler['avg_background_events']} background events per scenario)."
        )
    else:
        notes.append(
            f"Background filler is present but moderate (avg {filler['avg_background_events']} bg events; "
            f"{filler['avg_event_count']} total events on average)."
        )

    notes.append(
        "Failure-mode labels are pattern-bound via PATTERN_SPEC; validator + gold oracle passing "
        "indicates semantic alignment at generation time."
    )
    return notes


def scale_to_hard500_recommendation(
    *,
    distributions: dict[str, Any],
    method_metrics: dict[str, dict[str, float]],
    validator_pass: bool,
    gold_pass: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not validator_pass or not gold_pass:
        return "do_not_scale_yet", ["validator or gold oracle did not pass cleanly"]

    deepseek = method_metrics.get(short_model_name(DEEPSEEK_MODEL), {})
    latest = method_metrics.get("latest_only", {})
    retrieve = method_metrics.get("retrieve_all", {})

    signals_for_scale = 0
    signals_against = 0

    if latest.get("joint_revision_success", 1.0) <= 0.05:
        signals_for_scale += 1
        reasons.append("latest_only remains weak on joint_revision_success")
    else:
        signals_against += 1
        reasons.append("latest_only joint_revision_success is higher than expected for hard split")

    if retrieve.get("overcitation_rate", 0.0) >= 0.5:
        signals_for_scale += 1
        reasons.append("retrieve_all still overcites, preserving difficulty signal")
    else:
        signals_against += 1

    if 0.05 <= deepseek.get("joint_revision_success", 0.0) <= 0.45:
        signals_for_scale += 1
        reasons.append("DeepSeek joint_revision_success is low but non-zero (discriminative)")
    elif deepseek.get("joint_revision_success", 0.0) > 0.60:
        signals_against += 1
        reasons.append("DeepSeek joint_revision_success is unexpectedly high for hard split")
    else:
        signals_for_scale += 1
        reasons.append("DeepSeek joint_revision_success is very low (hard but may be too brittle)")

    if 0.10 <= deepseek.get("failure_diagnosis_accuracy", 0.0) <= 0.35:
        signals_for_scale += 1
        reasons.append("failure_diagnosis_accuracy remains meaningful but not trivial")
    elif deepseek.get("failure_diagnosis_accuracy", 0.0) > 0.50:
        signals_against += 1
        reasons.append("failure_diagnosis_accuracy may be too easy at this scale")

    if distributions["pattern_balance"]["spread"] <= 1:
        signals_for_scale += 1
    else:
        signals_against += 1
        reasons.append("pattern distribution imbalance would amplify at hard_500")

    if distributions["decision_skew"]["top_decision_share"] >= 0.80:
        signals_against += 1
        reasons.append("decision distribution too skewed for clean hard_500 scaling")

    if signals_against > signals_for_scale:
        return "fix_generator_balance_before_hard_500", reasons
    if signals_for_scale >= 4:
        return "worth_scaling_to_hard_500", reasons
    return "continue_scale_testing", reasons


def pick_manual_cases(
    scenarios: dict[str, dict[str, Any]],
    deepseek_preds: dict[str, dict[str, Any]],
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

    deepseek_fail = []
    deepseek_success = []
    overcites = []
    for sid in scenarios:
        ds = deepseek_preds.get(sid)
        if ds:
            joint = float(ds.get("metrics", {}).get("joint_revision_success", 0.0))
            if joint >= 1.0:
                deepseek_success.append(sid)
            elif joint <= 0.0:
                deepseek_fail.append(sid)
        ret = retrieve_preds.get(sid)
        if ret and float(ret.get("metrics", {}).get("overcitation_rate", 0.0)) > 0.5:
            overcites.append(sid)

    rng.shuffle(deepseek_fail)
    rng.shuffle(deepseek_success)
    rng.shuffle(overcites)
    for sid in deepseek_fail[:5]:
        add(sid, "DeepSeek joint_revision_success = 0")
    for sid in deepseek_success[:5]:
        add(sid, "DeepSeek joint_revision_success = 1")
    for sid in overcites[:5]:
        add(sid, "retrieve_all overcites evidence")
    pool = [sid for sid in scenarios if sid not in used]
    rng.shuffle(pool)
    for sid in pool[:5]:
        add(sid, "random sample")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--validator-pass", action="store_true")
    parser.add_argument("--gold-pass", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    api_dir = out_dir / "api_models"
    scenarios_list = read_jsonl(Path(args.data))
    scenarios = {row["scenario_id"]: row for row in scenarios_list}
    count = len(scenarios_list)

    distributions = compute_distributions(scenarios_list)
    semantic_notes = semantic_alignment_notes(distributions)

    method_metrics: dict[str, dict[str, float]] = {}
    for baseline in ("latest_only", "retrieve_all"):
        metrics_path = out_dir / f"{baseline}.predictions.metrics.json"
        if metrics_path.exists():
            method_metrics[baseline] = load_metrics(metrics_path)

    gold_path = out_dir / "gold_oracle.metrics.json"
    if gold_path.exists():
        method_metrics["gold_oracle"] = load_metrics(gold_path)

    deepseek_pred_path = api_dir / f"{model_slug(DEEPSEEK_MODEL)}.predictions.jsonl"
    deepseek_metrics_path = api_dir / f"{model_slug(DEEPSEEK_MODEL)}.predictions.metrics.json"
    deepseek_preds: dict[str, dict[str, Any]] = {}
    if deepseek_metrics_path.exists():
        method_metrics[short_model_name(DEEPSEEK_MODEL)] = load_metrics(deepseek_metrics_path)
    if deepseek_pred_path.exists():
        deepseek_preds = load_predictions(deepseek_pred_path)

    retrieve_preds = (
        load_predictions(out_dir / "retrieve_all.predictions.jsonl")
        if (out_dir / "retrieve_all.predictions.jsonl").exists()
        else {}
    )

    scale_rec, scale_reasons = scale_to_hard500_recommendation(
        distributions=distributions,
        method_metrics=method_metrics,
        validator_pass=args.validator_pass,
        gold_pass=args.gold_pass,
    )

    summary = {
        "count": count,
        "split_label": f"hard_{count}_en",
        "release_status": "scale_test_not_final_release",
        "distributions": distributions,
        "semantic_alignment_notes": semantic_notes,
        "methods": method_metrics,
        "report_metrics": list(REPORT_METRICS),
        "scale_to_hard_500_recommendation": scale_rec,
        "scale_to_hard_500_reasons": scale_reasons,
    }
    (out_dir / "summary.metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# ReTrace-Bench Hard{count} Scale Test Summary",
        "",
        "**Status:** scale test only — not a final release.",
        "",
        f"Dataset: `{args.data}`",
        f"Scenarios: {count}",
        "",
        "## Distributions",
        "",
        "### Pattern",
        "```json",
        json.dumps(distributions["pattern"], indent=2, sort_keys=True),
        "```",
        "",
        "### Failure mode",
        "```json",
        json.dumps(distributions["failure_mode"], indent=2, sort_keys=True),
        "```",
        "",
        "### Expected decision",
        "```json",
        json.dumps(distributions["expected_decision"], indent=2, sort_keys=True),
        "```",
        "",
        "### Difficulty / source",
        f"- difficulty: `{distributions['difficulty']}`",
        f"- source_type: `{distributions['source_type']}`",
        f"- pattern balance spread: `{distributions['pattern_balance']['spread']}`",
        f"- top decision share: `{distributions['decision_skew']['top_decision']}` @ "
        f"{100 * distributions['decision_skew']['top_decision_share']:.1f}%",
        "",
        "## Semantic / quality notes",
        "",
    ]
    for note in semantic_notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| method | " + " | ".join(REPORT_METRICS) + " |",
            "| --- | " + " | ".join(["---"] * len(REPORT_METRICS)) + " |",
        ]
    )
    for method, metrics in method_metrics.items():
        if method == "gold_oracle":
            continue
        row = " | ".join(f"{metrics.get(k, float('nan')):.3f}" for k in REPORT_METRICS)
        lines.append(f"| {method} | {row} |")
    lines.extend(["", "## Scale to hard_500", "", f"**{scale_rec}**", ""])
    for reason in scale_reasons:
        lines.append(f"- {reason}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rng = random.Random(args.seed)
    picks = pick_manual_cases(scenarios, deepseek_preds, retrieve_preds, rng)
    inspect_lines = [f"# Hard{count} Manual Inspection (20 cases)", ""]
    for sid, reason in picks:
        scenario = scenarios[sid]
        gold = canonical_hidden_gold_fields(scenario["hidden_gold"])
        pattern = scenario.get("pattern") or scenario.get("metadata", {}).get("pattern", "unknown")
        inspect_lines.extend(
            [
                f"## {sid}",
                "",
                f"- **Selection reason:** {reason}",
                f"- **Pattern / failure / difficulty:** {pattern} / "
                f"{scenario.get('primary_failure_mode')} / {scenario.get('difficulty')}",
                f"- **Expected decision:** `{gold['expected_decision']}`",
                f"- **Expected diagnosis:** `{gold['expected_failure_diagnosis']}`",
                f"- **Expected evidence:** `{gold['expected_evidence_event_ids']}`",
                "",
                "### Event trace summary",
                event_summary(scenario),
                "",
                "### Model outputs",
            ]
        )
        ds = deepseek_preds.get(sid, {}).get("response", {})
        ds_metrics = deepseek_preds.get(sid, {}).get("metrics", {})
        inspect_lines.append(
            f"- **DeepSeek-V4-Pro:** decision=`{ds.get('decision')}` "
            f"evidence=`{ds.get('evidence_event_ids')}` diagnosis=`{ds.get('failure_diagnosis')}` "
            f"joint={ds_metrics.get('joint_revision_success')}"
        )
        ret = retrieve_preds.get(sid, {}).get("response", {})
        inspect_lines.append(
            f"- **retrieve_all:** decision=`{ret.get('decision')}` evidence=`{ret.get('evidence_event_ids')}` "
            f"overcitation={retrieve_preds.get(sid, {}).get('metrics', {}).get('overcitation_rate')}"
        )
        inspect_lines.extend(["", ""])
    (out_dir / "manual_inspection_20.md").write_text("\n".join(inspect_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "scale_to_hard_500": scale_rec,
                "summary": str(out_dir / "summary.md"),
                "manual_inspection": str(out_dir / "manual_inspection_20.md"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
