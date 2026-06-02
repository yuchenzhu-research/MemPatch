#!/usr/bin/env python3
"""Cross-model audit of ReTrace-Bench API baseline outputs.

Loads two (or more) baseline prediction JSONL files (plus their ``*.metrics.json``
companions when present) and emits a compact Markdown audit report covering:

* global headline / auxiliary metrics per model;
* per-expected-decision accuracy and cross-model gaps;
* per-primary-failure-mode breakdown;
* failure-diagnosis confusion matrices and predicted-diagnosis distributions;
* pairwise scenario-level comparisons and a top suspicious-scenario list.

This script is read-only: it does not mutate any benchmark data or outputs. It is
a diagnostic tool for the template-heldout v1 design audit and is safe to re-run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical orderings (kept local so the audit does not depend on import side
# effects, but mirrors benchmark.retrace_bench.general_taxonomy).
DECISIONS = (
    "use_current_memory",
    "escalate",
    "ask_clarification",
    "refuse_due_to_policy",
    "mark_unresolved",
)
FAILURE_MODES = (
    "stale_memory_reuse",
    "under_update",
    "over_update",
    "conflict_collapse",
    "scope_leakage",
    "policy_violation",
    "wrong_source_attribution",
    "memory_hallucination",
    "unnecessary_memory_write",
    "failure_to_forget",
    "failure_to_release_or_restore",
)

GLOBAL_METRIC_KEYS = (
    "black_box_decision_accuracy",
    "decision_macro_f1",
    "decision_balanced_accuracy",
    "non_answer_decision_accuracy",
    "use_current_memory_accuracy",
    "memory_state_accuracy",
    "evidence_f1",
    "failure_diagnosis_accuracy",
    "stale_reuse_rate",
    "answer_key_fact_accuracy",
    "answer_exact_match",
    "format_failure_rate",
    "forbidden_fact_hits",
)


def _infer_model_name(jsonl_path: Path) -> str:
    stem = jsonl_path.stem
    for suffix in ("_llm_json_test800_first200", "_llm_json", "_test800_first200"):
        if suffix in stem:
            return stem.split(suffix)[0]
    return stem


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _pred_diagnosis(row: dict[str, Any]) -> str:
    diag = row.get("response", {}).get("failure_diagnosis")
    if isinstance(diag, list):
        diag = diag[0] if diag else None
    return _norm(diag)


def _pred_decision(row: dict[str, Any]) -> str:
    return _norm(row.get("response", {}).get("decision"))


def load_model(jsonl_path: Path) -> dict[str, Any]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    metrics_path = jsonl_path.with_suffix(".metrics.json")
    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_id = {r["scenario_id"]: r for r in rows}
    return {
        "name": _infer_model_name(jsonl_path),
        "jsonl_path": jsonl_path,
        "metrics_path": metrics_path if metrics_path.exists() else None,
        "rows": rows,
        "by_id": by_id,
        "metrics": metrics,
    }


def _agg_metric(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r.get("metrics", {}).get(key) for r in rows]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def global_table(models: list[dict[str, Any]]) -> str:
    lines = ["## 2.1 Global metrics", ""]
    header = "| metric | " + " | ".join(m["name"] for m in models) + " | max gap |"
    sep = "| --- | " + " | ".join("---" for _ in models) + " | --- |"
    lines += [header, sep]
    for key in GLOBAL_METRIC_KEYS:
        vals = []
        for m in models:
            v = None
            if m["metrics"]:
                v = m["metrics"].get("all_metrics", {}).get(key, m["metrics"].get("metrics", {}).get(key))
            if v is None:
                v = _agg_metric(m["rows"], key)
            vals.append(v)
        numeric = [v for v in vals if isinstance(v, (int, float))]
        gap = (max(numeric) - min(numeric)) if len(numeric) >= 2 else float("nan")
        lines.append("| " + key + " | " + " | ".join(_fmt(v) for v in vals) + f" | {_fmt(gap)} |")
    return "\n".join(lines) + "\n"


def per_decision_table(models: list[dict[str, Any]]) -> str:
    lines = ["## 2.2 Per-expected-decision accuracy", ""]
    header = "| expected_decision | count | " + " | ".join(f"{m['name']} acc" for m in models) + " | gap | flag |"
    sep = "| --- | --- | " + " | ".join("---" for _ in models) + " | --- | --- |"
    lines += [header, sep]
    for dec in DECISIONS:
        counts = []
        accs = []
        for m in models:
            sub = [r for r in m["rows"] if _norm(r.get("expected_decision")) == dec]
            counts.append(len(sub))
            accs.append(_agg_metric(sub, "black_box_decision_accuracy") if sub else float("nan"))
        numeric = [a for a in accs if isinstance(a, float) and a == a]
        gap = (max(numeric) - min(numeric)) if len(numeric) >= 2 else float("nan")
        flag = ""
        if numeric and all(a >= 0.95 for a in numeric):
            flag = "TOO_EASY (>=0.95 both)"
        elif (numeric and (gap == gap and gap >= 0.20)) or any(a < 0.75 for a in numeric):
            flag = "DISCRIMINATIVE"
        lines.append(
            "| " + dec + " | " + str(max(counts)) + " | "
            + " | ".join(_fmt(a) for a in accs) + f" | {_fmt(gap)} | {flag} |"
        )
    return "\n".join(lines) + "\n"


def per_failure_mode_table(models: list[dict[str, Any]]) -> str:
    lines = ["## 2.3 Per-primary-failure-mode breakdown", ""]
    cols = ["count", "decision_acc", "memory_state_acc", "evidence_f1", "diag_acc", "stale_reuse"]
    metric_keys = {
        "decision_acc": "black_box_decision_accuracy",
        "memory_state_acc": "memory_state_accuracy",
        "evidence_f1": "evidence_f1",
        "diag_acc": "failure_diagnosis_accuracy",
        "stale_reuse": "stale_reuse_rate",
    }
    header = "| failure_mode | model | " + " | ".join(cols) + " | flags |"
    sep = "| --- | --- | " + " | ".join("---" for _ in cols) + " | --- |"
    lines += [header, sep]
    for mode in FAILURE_MODES:
        diag_accs = []
        dec_accs = []
        ev_f1s = []
        per_model_cells = []
        for m in models:
            sub = [r for r in m["rows"] if _norm(r.get("primary_failure_mode")) == mode]
            count = len(sub)
            cells = {"count": count}
            for col, key in metric_keys.items():
                cells[col] = _agg_metric(sub, key) if sub else float("nan")
            diag_accs.append(cells["diag_acc"])
            dec_accs.append(cells["decision_acc"])
            ev_f1s.append(cells["evidence_f1"])
            per_model_cells.append((m["name"], cells))

        def _valid(xs):
            return [x for x in xs if isinstance(x, float) and x == x]

        flags = []
        v_diag = _valid(diag_accs)
        v_dec = _valid(dec_accs)
        v_ev = _valid(ev_f1s)
        if v_diag and all(x == 0 for x in v_diag):
            flags.append("DIAG_ZERO_BOTH")
        elif v_diag and all(x < 0.15 for x in v_diag):
            flags.append("DIAG_LOW_BOTH")
        if v_dec and all(x >= 0.95 for x in v_dec):
            flags.append("DECISION_SATURATED")
        if v_ev and all(x < 0.30 for x in v_ev):
            flags.append("EVIDENCE_LOW_BOTH")
        flag_str = ";".join(flags)
        for i, (name, cells) in enumerate(per_model_cells):
            row_flag = flag_str if i == 0 else ""
            lines.append(
                "| " + (mode if i == 0 else "") + " | " + name + " | "
                + " | ".join(_fmt(cells[c]) for c in cols) + f" | {row_flag} |"
            )
    return "\n".join(lines) + "\n"


def confusion_section(models: list[dict[str, Any]]) -> str:
    lines = ["## 2.4 Failure-diagnosis confusion + predicted distribution", ""]
    for m in models:
        lines.append(f"### {m['name']}")
        lines.append("")
        pred_dist = Counter(_pred_diagnosis(r) for r in m["rows"])
        lines.append("Predicted diagnosis distribution (pred -> count):")
        lines.append("")
        for label, c in sorted(pred_dist.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{label or '(none)'}`: {c}")
        lines.append("")
        # Confusion: gold -> predicted counts, compact (only nonzero).
        conf: dict[str, Counter] = defaultdict(Counter)
        for r in m["rows"]:
            gold = _norm(r.get("primary_failure_mode"))
            conf[gold][_pred_diagnosis(r)] += 1
        lines.append("Confusion (gold primary_failure_mode -> top predicted):")
        lines.append("")
        lines.append("| gold | n | top predicted (count) | diag_acc |")
        lines.append("| --- | --- | --- | --- |")
        for mode in FAILURE_MODES:
            row_counter = conf.get(mode, Counter())
            n = sum(row_counter.values())
            if n == 0:
                continue
            top = ", ".join(f"{k or '(none)'}={v}" for k, v in row_counter.most_common(3))
            correct = row_counter.get(mode, 0)
            lines.append(f"| {mode} | {n} | {top} | {_fmt(correct / n)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def pairwise_section(models: list[dict[str, Any]], top_n: int = 30) -> str:
    if len(models) < 2:
        return "## 2.5 Pairwise comparison\n\n(Only one model provided.)\n"
    a, b = models[0], models[1]
    common_ids = [sid for sid in a["by_id"] if sid in b["by_id"]]
    lines = ["## 2.5 Pairwise model comparison", "", f"Models: **{a['name']}** vs **{b['name']}**; {len(common_ids)} common scenarios.", ""]

    both_dec_right = 0
    both_diag_wrong = 0
    both_same_wrong_diag = 0
    both_ev_low = 0
    both_mem_low = 0
    both_overpred_scope = 0
    suspicious: list[dict[str, Any]] = []

    for sid in common_ids:
        ra, rb = a["by_id"][sid], b["by_id"][sid]
        gold_dec = _norm(ra.get("expected_decision"))
        gold_mode = _norm(ra.get("primary_failure_mode"))
        ma, mb = ra.get("metrics", {}), rb.get("metrics", {})
        dec_a_ok = ma.get("black_box_decision_accuracy", 0) >= 1.0
        dec_b_ok = mb.get("black_box_decision_accuracy", 0) >= 1.0
        diag_a_ok = ma.get("failure_diagnosis_accuracy", 0) >= 1.0
        diag_b_ok = mb.get("failure_diagnosis_accuracy", 0) >= 1.0
        pda, pdb = _pred_diagnosis(ra), _pred_diagnosis(rb)
        ev_a, ev_b = ma.get("evidence_f1", 0.0), mb.get("evidence_f1", 0.0)
        mem_a, mem_b = ma.get("memory_state_accuracy", 0.0), mb.get("memory_state_accuracy", 0.0)
        kf_a, kf_b = ma.get("answer_key_fact_accuracy", 0.0), mb.get("answer_key_fact_accuracy", 0.0)

        if dec_a_ok and dec_b_ok:
            both_dec_right += 1
        if (not diag_a_ok) and (not diag_b_ok):
            both_diag_wrong += 1
            if pda == pdb:
                both_same_wrong_diag += 1
        if ev_a < 0.5 and ev_b < 0.5:
            both_ev_low += 1
        if mem_a < 0.75 and mem_b < 0.75:
            both_mem_low += 1
        if gold_mode != "scope_leakage" and pda == "scope_leakage" and pdb == "scope_leakage":
            both_overpred_scope += 1

        # Suspicion scoring.
        reasons = []
        if dec_a_ok and dec_b_ok and (not diag_a_ok) and (not diag_b_ok):
            reasons.append("dec_ok_diag_wrong")
        if gold_mode != "scope_leakage" and pda == "scope_leakage" and pdb == "scope_leakage":
            reasons.append("both_overpred_scope")
        if dec_a_ok and dec_b_ok and ev_a < 0.5 and ev_b < 0.5:
            reasons.append("dec_ok_evidence_low")
        if dec_a_ok and dec_b_ok and kf_a == 0 and kf_b == 0:
            reasons.append("dec_ok_keyfact_zero")
        if gold_dec == "refuse_due_to_policy" and (not dec_a_ok) and (not dec_b_ok):
            reasons.append("both_fail_refuse")
        if reasons:
            suspicious.append({
                "sid": sid, "domain": ra.get("domain"), "gold_mode": gold_mode,
                "gold_dec": gold_dec, "dec_a": _pred_decision(ra), "dec_b": _pred_decision(rb),
                "diag_a": pda, "diag_b": pdb, "mem_a": mem_a, "mem_b": mem_b,
                "ev_a": ev_a, "ev_b": ev_b, "stale_a": ma.get("stale_reuse_rate", 0.0),
                "stale_b": mb.get("stale_reuse_rate", 0.0), "reasons": reasons,
            })

    lines += [
        "Aggregate pairwise counts (out of common scenarios):",
        "",
        f"- both decision correct: {both_dec_right}",
        f"- both diagnosis wrong: {both_diag_wrong}",
        f"- both predicted the SAME wrong diagnosis: {both_same_wrong_diag}",
        f"- both evidence_f1 < 0.5: {both_ev_low}",
        f"- both memory_state_accuracy < 0.75: {both_mem_low}",
        f"- both over-predicted scope_leakage (gold != scope_leakage): {both_overpred_scope}",
        "",
        f"### Top {top_n} suspicious scenarios",
        "",
        "| scenario_id | domain | gold_mode | gold_dec | dec(A/B) | diag(A/B) | mem(A/B) | ev_f1(A/B) | stale(A/B) | reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    suspicious.sort(key=lambda d: (-len(d["reasons"]), d["sid"]))
    for d in suspicious[:top_n]:
        lines.append(
            f"| {d['sid']} | {d['domain']} | {d['gold_mode']} | {d['gold_dec']} | "
            f"{d['dec_a']}/{d['dec_b']} | {d['diag_a']}/{d['diag_b']} | "
            f"{_fmt(d['mem_a'])}/{_fmt(d['mem_b'])} | {_fmt(d['ev_a'])}/{_fmt(d['ev_b'])} | "
            f"{_fmt(d['stale_a'])}/{_fmt(d['stale_b'])} | {','.join(d['reasons'])} |"
        )
    lines.append("")
    lines.append(f"Total suspicious scenarios flagged: {len(suspicious)}")
    return "\n".join(lines) + "\n"


def inventory_section(models: list[dict[str, Any]]) -> str:
    lines = ["## Phase 1 inventory", ""]
    lines.append("| model | jsonl | predictions | metrics.json | metrics count | counts match |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for m in models:
        n = len(m["rows"])
        mcount = m["metrics"].get("count") if m["metrics"] else None
        match = "n/a" if mcount is None else ("yes" if mcount == n else "NO")
        rel = m["jsonl_path"].relative_to(REPO_ROOT) if str(m["jsonl_path"]).startswith(str(REPO_ROOT)) else m["jsonl_path"].name
        lines.append(f"| {m['name']} | {rel} | {n} | {'yes' if m['metrics'] else 'MISSING'} | {mcount} | {match} |")
    return "\n".join(lines) + "\n"


def build_report(model_paths: list[Path]) -> str:
    models = [load_model(p) for p in model_paths]
    parts = [
        "# ReTrace-Bench cross-model output audit",
        "",
        "Auto-generated by `scripts/audit_retrace_bench_outputs.py` (read-only).",
        "",
        inventory_section(models),
        global_table(models),
        per_decision_table(models),
        per_failure_mode_table(models),
        confusion_section(models),
        pairwise_section(models),
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=[
            "outputs/retrace_bench/deepseek_v3_llm_json_test800_first200.jsonl",
            "outputs/retrace_bench/glm_51_llm_json_test800_first200.jsonl",
        ],
        help="Prediction JSONL files (metrics.json companions auto-detected).",
    )
    parser.add_argument("--out", default=None, help="Optional Markdown output path; prints to stdout if omitted.")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args(argv)

    paths = [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in args.predictions]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"prediction file not found: {p}")
    report = build_report(paths)
    if args.out:
        out_path = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
