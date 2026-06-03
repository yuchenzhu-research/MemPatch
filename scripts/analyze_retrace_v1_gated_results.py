#!/usr/bin/env python3
"""Read-only analysis of the ReTrace-Bench v1.0 gated pilot runs.

Reads frozen benchmark scenarios and the committed prediction/metrics dumps and
emits the aggregate tables consumed by
``docs/retrace_bench/v1_0_gated_model_pilot.md`` plus machine-readable
``gated_summary.{json,jsonl}``.

This script NEVER writes to ``data/retrace_bench/`` and never mutates scoring,
labels, generators, or the HF package. It only reads dataset + prediction files
and writes analysis artifacts under ``outputs/`` / ``docs/``. The
"primary-or-secondary" diagnosis number it reports is analysis-only; the strict
single-label ``failure_diagnosis_accuracy`` remains the official metric.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.retrace_bench.general_taxonomy import DECISIONS, FAILURE_MODES
from benchmark.retrace_bench.scorers_general import (
    decision_matches,
    normalize_failure_mode,
)

ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "hard_300_full": {
        "data": ROOT / "data/retrace_bench/hard_300_en/scenarios.jsonl",
        "dir": ROOT / "outputs/retrace_bench/v1_0/hard_300",
        "n_expected": 300,
        "models": {
            "Kimi-K2.6": "kimi_k26_hard300_full.jsonl",
            "GLM-5.1": "glm_51_hard300_full.jsonl",
            "DeepSeek-V4-Pro": "deepseek_v4_pro_hard300_full.jsonl",
        },
    },
    "main_3000_first500": {
        "data": ROOT / "data/retrace_bench/main_3000_en/scenarios.jsonl",
        "dir": ROOT / "outputs/retrace_bench/v1_0/main_3000_first500",
        "n_expected": 500,
        "models": {
            "Kimi-K2.6": "kimi_k26_main3000_first500.jsonl",
            "GLM-5.1": "glm_51_main3000_first500.jsonl",
            "DeepSeek-V4-Pro": "deepseek_v4_pro_main3000_first500.jsonl",
        },
    },
}

HEADLINE = [
    "format_failure_rate",
    "decision_macro_f1",
    "non_answer_decision_accuracy",
    "memory_state_accuracy",
    "evidence_f1",
    "failure_diagnosis_accuracy",
    "stale_reuse_rate",
]


def load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                s = json.loads(line)
                out[s["scenario_id"]] = s
    return out


def parse_response(rec: dict[str, Any]) -> dict[str, Any]:
    resp = rec.get("response", rec)
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except json.JSONDecodeError:
            resp = {}
    return resp if isinstance(resp, dict) else {}


def analyze_run(scen: dict[str, Any], pred_path: Path, metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text())
    headline = metrics.get("headline_metrics", {})
    all_metrics = metrics.get("all_metrics", {})

    rows = [json.loads(l) for l in pred_path.open() if l.strip()]

    joint = 0
    correct_decision = 0
    cd_any_struct_wrong = 0
    cd_mem_wrong = cd_ev_wrong = cd_diag_wrong = cd_all_struct_ok = 0

    strict_diag_correct = 0
    pri_or_sec_correct = 0
    pred_dist = Counter()
    confusion = defaultdict(Counter)        # gold_mode -> Counter(pred)
    mode_tot = Counter()
    mode_correct_diag = Counter()
    mode_dec = defaultdict(list)
    mode_mem = defaultdict(list)
    mode_ev = defaultdict(list)
    mode_stale = defaultdict(list)

    dec_tot = Counter()                      # gold decision -> count
    dec_correct = Counter()                  # gold decision -> #correct
    dec_pred_for_gold = defaultdict(Counter)  # gold decision -> Counter(pred decision)

    for r in rows:
        sid = r["scenario_id"]
        s = scen[sid]
        gold = s["hidden_gold"]
        gdec = gold.get("expected_decision")
        gdiag = gold.get("expected_failure_diagnosis")
        gmode = s.get("primary_failure_mode")
        sec = set(s.get("secondary_failure_modes") or [])
        aliases = gold.get("decision_aliases") or r.get("decision_aliases")

        resp = parse_response(r)
        pdec = resp.get("decision")
        pdiag = normalize_failure_mode(resp.get("failure_diagnosis"))

        m = r.get("metrics", {})
        dec_ok = bool(m.get("black_box_decision_accuracy", 0.0) == 1.0)
        mem_ok = m.get("memory_state_accuracy", 0.0) == 1.0
        ev_ok = m.get("evidence_f1", 0.0) == 1.0
        diag_ok = bool(m.get("failure_diagnosis_accuracy", 0.0) == 1.0)

        if dec_ok and mem_ok and ev_ok and diag_ok:
            joint += 1
        if dec_ok:
            correct_decision += 1
            mwrong = not mem_ok
            ewrong = not ev_ok
            dwrong = not diag_ok
            if mwrong or ewrong or dwrong:
                cd_any_struct_wrong += 1
            else:
                cd_all_struct_ok += 1
            cd_mem_wrong += int(mwrong)
            cd_ev_wrong += int(ewrong)
            cd_diag_wrong += int(dwrong)

        # diagnosis
        strict_diag_correct += int(diag_ok)
        pri_or_sec_correct += int(pdiag == gdiag or pdiag in sec)
        pred_dist[pdiag] += 1
        confusion[gmode][pdiag] += 1
        mode_tot[gmode] += 1
        mode_correct_diag[gmode] += int(diag_ok)
        mode_dec[gmode].append(float(dec_ok))
        mode_mem[gmode].append(m.get("memory_state_accuracy", 0.0))
        mode_ev[gmode].append(m.get("evidence_f1", 0.0))
        mode_stale[gmode].append(m.get("stale_reuse_rate", m.get("stale_memory_reuse_rate", 0.0)))

        # per-decision
        dec_tot[gdec] += 1
        dec_correct[gdec] += int(dec_ok)
        dec_pred_for_gold[gdec][pdec if pdec in DECISIONS else f"_other:{pdec}"] += 1

    n = len(rows)
    per_mode = {}
    for mode in FAILURE_MODES:
        if mode_tot[mode]:
            per_mode[mode] = {
                "n": mode_tot[mode],
                "diagnosis_accuracy": mode_correct_diag[mode] / mode_tot[mode],
                "decision_accuracy": sum(mode_dec[mode]) / len(mode_dec[mode]),
                "memory_state_accuracy": sum(mode_mem[mode]) / len(mode_mem[mode]),
                "evidence_f1": sum(mode_ev[mode]) / len(mode_ev[mode]),
                "stale_reuse_rate": sum(mode_stale[mode]) / len(mode_stale[mode]),
            }
    per_decision = {}
    for d in DECISIONS:
        if dec_tot[d]:
            per_decision[d] = {
                "n": dec_tot[d],
                "accuracy": dec_correct[d] / dec_tot[d],
                "predicted_breakdown": dict(dec_pred_for_gold[d].most_common()),
            }

    return {
        "n_cases": n,
        "headline_metrics": {k: headline.get(k, all_metrics.get(k)) for k in HEADLINE},
        "joint_all_correct": {
            "count": joint,
            "rate_all": joint / n if n else 0.0,
            "correct_decision_count": correct_decision,
            "rate_among_correct_decision": joint / correct_decision if correct_decision else 0.0,
        },
        "correct_decision_breakdown": {
            "correct_decision": correct_decision,
            "any_structure_wrong": cd_any_struct_wrong,
            "memory_wrong": cd_mem_wrong,
            "evidence_wrong": cd_ev_wrong,
            "diagnosis_wrong": cd_diag_wrong,
            "all_structure_correct": cd_all_struct_ok,
        },
        "diagnosis": {
            "strict_accuracy": strict_diag_correct / n if n else 0.0,
            "primary_or_secondary_accuracy_ANALYSIS_ONLY": pri_or_sec_correct / n if n else 0.0,
            "predicted_distribution": dict(pred_dist.most_common()),
            "confusion_gold_to_pred": {g: dict(c.most_common()) for g, c in confusion.items()},
            "per_mode": per_mode,
        },
        "per_decision": per_decision,
    }


def main() -> None:
    full: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for split, cfg in SPLITS.items():
        scen = load_scenarios(cfg["data"])
        full[split] = {}
        for model, fname in cfg["models"].items():
            pred = cfg["dir"] / fname
            met = cfg["dir"] / fname.replace(".jsonl", ".metrics.json")
            if not pred.exists() or not met.exists():
                print(f"MISSING {split}/{model}: {pred.name}")
                continue
            res = analyze_run(scen, pred, met)
            full[split][model] = res
            row = {"model": model, "split": split, "n_cases": res["n_cases"]}
            row.update({k: res["headline_metrics"][k] for k in HEADLINE})
            row["joint_all_correct_rate"] = res["joint_all_correct"]["rate_all"]
            row["joint_all_correct_count"] = res["joint_all_correct"]["count"]
            row["strict_diagnosis_accuracy"] = res["diagnosis"]["strict_accuracy"]
            row["primary_or_secondary_diagnosis_accuracy_analysis_only"] = res["diagnosis"][
                "primary_or_secondary_accuracy_ANALYSIS_ONLY"
            ]
            summary_rows.append(row)

    out_dir = ROOT / "outputs/retrace_bench/v1_0"
    summary = {
        "benchmark": "ReTrace-Bench v1.0 gated model pilot",
        "splits": {"hard_300_en": "full 300", "main_3000_en": "first 500"},
        "note": "gated pilot before main_3000 full; realistic_100 excluded (annotation pending). "
        "primary_or_secondary_diagnosis_accuracy is analysis-only; strict single-label "
        "failure_diagnosis_accuracy is the official metric.",
        "rows": summary_rows,
    }
    (out_dir / "gated_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "gated_summary.jsonl").open("w") as fh:
        for row in summary_rows:
            fh.write(json.dumps(row) + "\n")
    # Verbose per-mode/per-decision dump for report authoring only; written
    # outside the tracked tree so it is never committed.
    Path("/tmp/gated_analysis_full.json").write_text(json.dumps(full, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
