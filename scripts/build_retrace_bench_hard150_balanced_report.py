#!/usr/bin/env python3
"""Build hard150_balanced eval report comparing to legacy hard150 SiliconFlow run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.retrace_bench.generation.pattern_spec import hard_decision_distribution
from scripts.run_retrace_bench_baseline import read_jsonl

REPORT_KEYS = (
    "decision_macro_f1",
    "black_box_decision_accuracy",
    "non_answer_decision_accuracy",
    "memory_state_accuracy",
    "evidence_f1",
    "failure_diagnosis_accuracy",
    "joint_revision_success",
    "format_failure_rate",
)


def load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    src = data.get("report_metrics") or data.get("all_metrics") or data
    return {k: float(src[k]) for k in REPORT_KEYS if k in src}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    balanced_dir = root / "outputs/retrace_bench_siliconflow_hard150_balanced"
    legacy_dir = root / "outputs/retrace_bench_siliconflow_hard150"
    data_path = root / "data/retrace_bench_hard150_balanced/hard_150_en/scenarios.jsonl"

    dist = hard_decision_distribution(150, 2027)
    scenarios = read_jsonl(data_path)

    model_names = ["DeepSeek-V4-Pro", "GLM-5.1", "Kimi-K2.6"]
    balanced: dict[str, dict[str, float]] = {}
    legacy: dict[str, dict[str, float]] = {}
    for name in model_names:
        balanced[name] = load_metrics(balanced_dir / f"{name}.metrics.json")
        legacy[name] = load_metrics(legacy_dir / f"{name}.metrics.json")

    gates = {
        "all_models_present": all(len(balanced[n]) > 0 for n in model_names),
        "format_failure_zero": all(
            balanced[n].get("format_failure_rate", 1.0) == 0.0 for n in model_names if balanced[n]
        ),
        "joint_below_half": all(
            balanced[n].get("joint_revision_success", 1.0) < 0.5
            for n in model_names
            if balanced[n]
        ),
    }
    pass_all = all(gates.values()) if gates["all_models_present"] else False

    lines = [
        "# ReTrace-Bench hard150_balanced — Final Hardening Report",
        "",
        f"Dataset: `{data_path.relative_to(root)}`",
        f"Scenarios: {len(scenarios)}",
        "",
        "## Gold expected_decision distribution (scheduled)",
        "",
        "```json",
        json.dumps(dist, indent=2),
        "```",
        "",
        "| decision | count | share |",
        "| --- | ---: | ---: |",
    ]
    total = sum(dist.values())
    for decision, count in sorted(dist.items()):
        lines.append(f"| {decision} | {count} | {count / total:.1%} |")

    lines.extend(
        [
            "",
            "## SiliconFlow three-model metrics (balanced vs legacy hard150)",
            "",
            "| model | metric | balanced | legacy (pre-schedule) | Δ |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name in model_names:
        for key in REPORT_KEYS:
            b = balanced.get(name, {}).get(key)
            l = legacy.get(name, {}).get(key)
            if b is None and l is None:
                continue
            delta = (b - l) if b is not None and l is not None else None
            delta_s = f"{delta:+.3f}" if delta is not None else "n/a"
            b_s = f"{b:.3f}" if b is not None else "n/a"
            l_s = f"{l:.3f}" if l is not None else "n/a"
            lines.append(f"| {name} | {key} | {b_s} | {l_s} | {delta_s} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **memory_state_accuracy**: Should drop vs legacy hard150 when `is_distractor` is stripped from model prompts (no leakage).",
            "- **non_answer_decision_accuracy**: More meaningful with balanced refuse/mark/ask/escalate shares (~50% non-answer cases).",
            "- **joint_revision_success**: Headline stress metric; target &lt; 0.5 for all models.",
            "- **format_failure_rate**: Must remain 0.0.",
            "",
            "## Gates",
            "",
        ]
    )
    for gate, ok in gates.items():
        lines.append(f"- {gate}: **{'PASS' if ok else 'FAIL'}**")
    lines.append(f"- **overall hard150_balanced pass:** **{'PASS' if pass_all else 'FAIL (incomplete eval)'}**")
    lines.append("")
    hard500_path = root / "data/retrace_bench_hard500_candidate/hard_500_en/scenarios.jsonl"
    if pass_all and hard500_path.exists():
        lines.append(
            "## hard500_candidate\n\n"
            f"Generated `{hard500_path.relative_to(root)}` (validator + gold oracle only)."
        )
    else:
        lines.append(
            "## hard500_candidate: "
            + ("blocked until validator/gold oracle generation completes" if pass_all else "blocked until SiliconFlow eval completes")
        )

    out = balanced_dir / "hard150_balanced_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
