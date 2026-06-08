#!/usr/bin/env python3
"""MemPatch benchmark-paper export helpers (figures + tables JSON/CSV)."""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.api import evaluate_predictions, load_predictions, load_scenarios
from benchmark.general_taxonomy import (
    DECISIONS,
    DIFFICULTY_DEFINITIONS,
    PRIMARY_DOMAINS,
    PRIMARY_FAILURE_MODES,
    TASK_TYPES,
    canonical_hidden_gold_fields,
)

HEADLINE_KEYS = (
    "joint_revision_success",
    "decision_macro_f1",
    "memory_state_accuracy",
    "evidence_f1",
    "failure_diagnosis_accuracy",
)

CAPABILITY_COLUMNS = (
    ("decision", "decision_macro_f1"),
    ("memory_state", "memory_state_accuracy"),
    ("evidence", "evidence_f1"),
    ("diagnosis", "failure_diagnosis_accuracy"),
    ("joint", "joint_revision_success"),
)

SPLIT_SCENARIOS: dict[str, Path] = {}


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_scenarios_path(split_tag: str | None) -> Path:
    root = _root()
    if split_tag and split_tag in SPLIT_SCENARIOS:
        return SPLIT_SCENARIOS[split_tag]
    mapping = {
        "test500": root / "local/train_data/paper/test500/scenarios.jsonl",
        "test": root / "hf_release/mempatch/test/scenarios.jsonl",
        "validation": root / "hf_release/mempatch/validation/scenarios.jsonl",
    }
    if split_tag in mapping and mapping[split_tag].is_file():
        return mapping[split_tag]
    return root / "hf_release/mempatch/test/scenarios.jsonl"


@dataclass
class EvalRun:
    model: str
    path: str
    variant: str
    split: str
    predictions_path: Path
    metrics_path: Path | None
    scored: list[dict[str, Any]]
    headline: dict[str, float]
    count: int


def _parse_run_stem(stem: str) -> dict[str, str]:
    # pathB_lora_test500, pathA_base_l3
    m = re.match(r"path([AB])_(\w+)_(.+)", stem)
    if not m:
        return {"path": "?", "variant": "?", "split": stem}
    path_letter, variant, split = m.group(1), m.group(2), m.group(3)
    return {"path": path_letter, "variant": variant, "split": split}


def discover_runs(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*_predictions.jsonl"))


def load_model_cards(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def load_selection_protocol(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    protocol = raw.get("_selection_protocol")
    return protocol if isinstance(protocol, dict) else {}


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return {
        "mean": sum(values) / n,
        "ci_low": means[lo_idx],
        "ci_high": means[hi_idx],
        "n": n,
        "n_bootstrap": n_boot,
    }


def _scenario_index(scenarios: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(s["scenario_id"]): s for s in scenarios}


def _joint_values(scored: list[dict[str, Any]]) -> list[float]:
    return [float(r.get("metrics", {}).get("joint_revision_success", 0.0)) for r in scored]


def load_eval_run(pred_path: Path, scenarios_path: Path | None = None) -> EvalRun:
    stem = pred_path.name.replace("_predictions.jsonl", "")
    metrics_path = pred_path.parent / f"{stem}_metrics.json"
    meta: dict[str, Any] = {}
    if metrics_path.is_file():
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
    parsed = _parse_run_stem(stem)
    model = str(meta.get("model") or pred_path.parent.name)
    path = str(meta.get("path") or parsed["path"])
    variant = str(meta.get("variant") or parsed["variant"])
    split = str(meta.get("split") or parsed["split"])
    scenarios_path = scenarios_path or resolve_scenarios_path(split)
    scenarios = load_scenarios(scenarios_path)
    predictions = load_predictions(pred_path)
    result = evaluate_predictions(scenarios, predictions, strict=False, allow_missing=True)
    return EvalRun(
        model=model,
        path=path,
        variant=variant,
        split=split,
        predictions_path=pred_path,
        metrics_path=metrics_path if metrics_path.is_file() else None,
        scored=result["scored_predictions"],
        headline=result["headline_metrics"],
        count=result["count"],
    )


def subset_metrics(scored: list[dict[str, Any]], predicate) -> dict[str, float]:
    rows = [r for r in scored if predicate(r)]
    if not rows:
        return {k: 0.0 for k in HEADLINE_KEYS}
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for key in HEADLINE_KEYS:
            totals[key] += float(row.get("metrics", {}).get(key, 0.0))
    n = len(rows)
    return {k: totals[k] / n for k in HEADLINE_KEYS}


def build_table1_dataset_stats(data_dirs: list[Path]) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for data_dir in data_dirs:
        path = data_dir / "scenarios.jsonl"
        if not path.is_file():
            continue
        rows = load_scenarios(path)
        split_name = rows[0].get("public_split_name") if rows else data_dir.name
        by_domain = Counter(r.get("domain") for r in rows)
        by_difficulty = Counter(r.get("difficulty") for r in rows)
        by_failure = Counter(r.get("primary_failure_mode") for r in rows)
        by_decision = Counter(
            canonical_hidden_gold_fields(r.get("hidden_gold") or {}).get("expected_decision")
            for r in rows
        )
        splits[str(split_name)] = {
            "count": len(rows),
            "domains": dict(by_domain),
            "difficulties": dict(by_difficulty),
            "primary_failure_modes": dict(by_failure),
            "expected_decisions": dict(by_decision),
            "difficulty_definitions": DIFFICULTY_DEFINITIONS,
            "task_types": list(TASK_TYPES),
            "decision_labels": list(DECISIONS),
            "failure_mode_labels": list(PRIMARY_FAILURE_MODES),
            "domain_labels": list(PRIMARY_DOMAINS),
        }
    return {"splits": splits, "total_scenarios": sum(s["count"] for s in splits.values())}


def build_table2_model_setup(runs: list[EvalRun], model_cards: dict[str, Any]) -> list[dict[str, Any]]:
    models = sorted({r.model for r in runs})
    rows: list[dict[str, Any]] = []
    for model in models:
        card = model_cards.get(model, {})
        run = next(r for r in runs if r.model == model)
        rows.append(
            {
                "model": model,
                "display_name": card.get("display_name", model),
                "family": card.get("family"),
                "model_generation": card.get("model_generation"),
                "paper_role": card.get("paper_role"),
                "selection_note": card.get("selection_note"),
                "total_params_b": card.get("total_params_b"),
                "active_params_b": card.get("active_params_b"),
                "context_length": card.get("context_length"),
                "license": card.get("license"),
                "quantization": card.get("quantization"),
                "inference": card.get("inference"),
                "hardware": card.get("hardware"),
                "eval_path": run.path,
                "eval_variant": run.variant,
                "eval_split": run.split,
            }
        )
    return rows


def build_table3_main_results(runs: list[EvalRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        base = {
            "model": run.model,
            "path": run.path,
            "variant": run.variant,
            "split": run.split,
            "count": run.count,
            **{k: run.headline.get(k) for k in HEADLINE_KEYS},
        }
        rows.append(base)
        for domain in PRIMARY_DOMAINS:
            dm = subset_metrics(run.scored, lambda r, d=domain: r.get("domain") == d)
            if dm["joint_revision_success"] == 0.0 and not any(
                r.get("domain") == domain for r in run.scored
            ):
                continue
            n = sum(1 for r in run.scored if r.get("domain") == domain)
            rows.append(
                {
                    **base,
                    "category_type": "domain",
                    "category": domain,
                    "count": n,
                    **dm,
                }
            )
        for difficulty in ("L3", "L4"):
            dm = subset_metrics(run.scored, lambda r, d=difficulty: r.get("difficulty") == d)
            n = sum(1 for r in run.scored if r.get("difficulty") == difficulty)
            if n:
                rows.append(
                    {
                        **base,
                        "category_type": "difficulty",
                        "category": difficulty,
                        "count": n,
                        **dm,
                    }
                )
    return rows


def build_table4_evaluator_reliability() -> dict[str, Any]:
    """MemPatch-Bench uses deterministic rule scoring (not LLM-as-judge)."""
    return {
        "evaluator": "MemPatch-Bench rule scorer (benchmark.api.evaluate_predictions)",
        "judge_type": "deterministic_hidden_gold",
        "agreement_with_gold": 1.0,
        "note": (
            "Primary evaluation is deterministic against hidden_gold fields. "
            "No LLM judge variance; reproduce by re-running evaluate_predictions on the same file."
        ),
        "recommended_human_subset": {
            "size": 100,
            "purpose": "failure_diagnosis and evidence quality spot-check",
            "metrics": ["failure_diagnosis_accuracy", "evidence_f1"],
        },
    }


def build_fig2_leaderboard(
    runs: list[EvalRun],
    *,
    path: str = "B",
    variant: str = "lora",
    split: str | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    selected = [
        r
        for r in runs
        if r.path == path and r.variant == variant and (split is None or r.split == split)
    ]
    bars: list[dict[str, Any]] = []
    for run in sorted(selected, key=lambda r: r.headline.get("joint_revision_success", 0), reverse=True):
        joints = _joint_values(run.scored)
        ci = bootstrap_ci(joints, n_boot=n_boot)
        bars.append(
            {
                "model": run.model,
                "score": ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n": ci["n"],
                "metric": "joint_revision_success",
            }
        )
    return {
        "figure": "fig2_leaderboard_ci",
        "metric": "joint_revision_success",
        "path": path,
        "variant": variant,
        "split": split,
        "n_bootstrap": n_boot,
        "bars": bars,
    }


def build_fig3_capability_heatmap(
    runs: list[EvalRun],
    *,
    path: str = "B",
    variant: str = "lora",
    split: str | None = None,
) -> dict[str, Any]:
    selected = [
        r
        for r in runs
        if r.path == path and r.variant == variant and (split is None or r.split == split)
    ]
    models = sorted({r.model for r in selected})
    cap_cols = [c[0] for c in CAPABILITY_COLUMNS]
    cap_matrix: dict[str, dict[str, float | None]] = {}
    for model in models:
        run = next(r for r in selected if r.model == model)
        cap_matrix[model] = {
            col: float(run.headline.get(metric, 0.0)) for col, metric in CAPABILITY_COLUMNS
        }

    domain_matrix: dict[str, dict[str, float | None]] = {}
    for model in models:
        run = next(r for r in selected if r.model == model)
        domain_matrix[model] = {}
        for domain in PRIMARY_DOMAINS:
            vals = [
                float(r.get("metrics", {}).get("joint_revision_success", 0.0))
                for r in run.scored
                if r.get("domain") == domain
            ]
            domain_matrix[model][domain] = sum(vals) / len(vals) if vals else None

    decision_matrix: dict[str, dict[str, float | None]] = {}
    for model in models:
        run = next(r for r in selected if r.model == model)
        by_dec: dict[str, list[float]] = defaultdict(list)
        for row in run.scored:
            dec = row.get("expected_decision")
            if dec:
                by_dec[str(dec)].append(float(row.get("metrics", {}).get("joint_revision_success", 0.0)))
        decision_matrix[model] = {
            dec: (sum(v) / len(v) if v else None) for dec, v in by_dec.items()
        }

    return {
        "figure": "fig3_capability_heatmap",
        "models": models,
        "capability_columns": cap_cols,
        "capability_matrix": cap_matrix,
        "domain_columns": list(PRIMARY_DOMAINS),
        "domain_joint_matrix": domain_matrix,
        "decision_columns": list(DECISIONS),
        "decision_joint_matrix": decision_matrix,
    }


def build_fig4_difficulty_breakdown(
    runs: list[EvalRun],
    *,
    path: str = "B",
    variant: str = "lora",
    split: str | None = None,
) -> dict[str, Any]:
    selected = [
        r
        for r in runs
        if r.path == path and r.variant == variant and (split is None or r.split == split)
    ]
    models = sorted({r.model for r in selected})
    groups: list[dict[str, Any]] = []
    for model in models:
        run = next(r for r in selected if r.model == model)
        entry: dict[str, Any] = {"model": model, "L3": {}, "L4": {}}
        for difficulty in ("L3", "L4"):
            rows = [r for r in run.scored if r.get("difficulty") == difficulty]
            if not rows:
                continue
            entry[difficulty] = {
                "n": len(rows),
                **subset_metrics(rows, lambda _: True),
            }
        groups.append(entry)
    return {"figure": "fig4_difficulty_breakdown", "models": models, "groups": groups}


def build_fig5_cost_performance(
    runs: list[EvalRun],
    model_cards: dict[str, Any],
    *,
    path: str = "B",
    variant: str = "lora",
    split: str | None = None,
) -> dict[str, Any]:
    selected = [
        r
        for r in runs
        if r.path == path and r.variant == variant and (split is None or r.split == split)
    ]
    points: list[dict[str, Any]] = []
    for run in selected:
        card = model_cards.get(run.model, {})
        points.append(
            {
                "model": run.model,
                "display_name": card.get("display_name", run.model),
                "family": card.get("family"),
                "family_color": card.get("family_color"),
                "active_params_b": card.get("active_params_b"),
                "total_params_b": card.get("total_params_b"),
                "joint_revision_success": run.headline.get("joint_revision_success"),
                "latency_ms_per_case": card.get("latency_ms_per_case"),
            }
        )
    return {"figure": "fig5_cost_performance", "points": points}


def build_fig6_pairwise_winrate(
    runs: list[EvalRun],
    *,
    path: str = "B",
    variant: str = "lora",
    split: str | None = None,
) -> dict[str, Any]:
    selected = [
        r
        for r in runs
        if r.path == path and r.variant == variant and (split is None or r.split == split)
    ]
    models = sorted({r.model for r in selected})
    scored_by_model: dict[str, dict[str, float]] = {}
    battle_counts: dict[str, dict[str, int]] = {m: {n: 0 for n in models} for m in models}
    win_rates: dict[str, dict[str, float | None]] = {m: {n: None for n in models} for m in models}

    for run in selected:
        scored_by_model[run.model] = {
            str(r["scenario_id"]): float(r.get("metrics", {}).get("joint_revision_success", 0.0))
            for r in run.scored
        }

    for a in models:
        for b in models:
            if a == b:
                win_rates[a][b] = 0.5
                continue
            ids_a = set(scored_by_model.get(a, {}))
            ids_b = set(scored_by_model.get(b, {}))
            common = ids_a & ids_b
            battle_counts[a][b] = len(common)
            wins_a = ties = 0
            for sid in common:
                sa = scored_by_model[a][sid]
                sb = scored_by_model[b][sid]
                if sa > sb:
                    wins_a += 1
                elif sa == sb:
                    ties += 1
            denom = len(common) - ties
            win_rates[a][b] = (wins_a / denom if denom > 0 else None)

    return {
        "figure": "fig6_pairwise_winrate",
        "models": models,
        "win_rate": win_rates,
        "battle_counts": battle_counts,
        "metric": "joint_revision_success",
        "tie_policy": "exclude_ties_from_denominator",
    }


def build_fig7_failure_taxonomy(runs: list[EvalRun]) -> dict[str, Any]:
    out: dict[str, Any] = {"figure": "fig7_failure_taxonomy", "models": {}}
    failure_keys = (
        "decision_failure",
        "memory_failure",
        "evidence_failure",
        "diagnosis_failure",
    )
    for run in runs:
        counts = Counter()
        for row in run.scored:
            if float(row.get("metrics", {}).get("joint_revision_success", 0.0)) >= 1.0:
                continue
            m = row.get("metrics", {})
            if float(m.get("black_box_decision_accuracy", 0.0)) < 1.0:
                counts["decision_failure"] += 1
            if float(m.get("memory_state_accuracy", 0.0)) < 1.0:
                counts["memory_failure"] += 1
            if float(m.get("evidence_f1", 0.0)) < 1.0:
                counts["evidence_failure"] += 1
            if float(m.get("failure_diagnosis_accuracy", 0.0)) < 1.0:
                counts["diagnosis_failure"] += 1
        total_fail = sum(counts.values()) or 1
        out["models"][f"{run.model}|path{run.path}|{run.variant}|{run.split}"] = {
            "counts": dict(counts),
            "fractions": {k: counts[k] / total_fail for k in failure_keys if counts[k]},
            "primary_failure_mode_breakdown": dict(
                Counter(r.get("primary_failure_mode") for r in run.scored if float(r.get("metrics", {}).get("joint_revision_success", 0.0)) < 1.0)
            ),
        }
    return out


def build_fig1_pipeline_mermaid() -> str:
    return "\n".join(
        [
            "flowchart LR",
            "  subgraph data [Data layer]",
            "    S[Scenario generator v1.3]",
            "    P[public_input event trace]",
            "    G[hidden_gold labels]",
            "  end",
            "  subgraph infer [Inference]",
            "    V[public_scenario_view or revision view]",
            "    M[Open-weight LLM]",
            "  end",
            "  subgraph eval [Evaluation]",
            "    R[response JSON interface]",
            "    E[Rule scorer evaluate_predictions]",
            "    D[DPA authorize optional Path A]",
            "  end",
            "  S --> P",
            "  S --> G",
            "  P --> V --> M --> R",
            "  R --> E",
            "  G --> E",
            "  R --> D --> E",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def export_all(
    results_dir: Path,
    out_dir: Path,
    *,
    model_cards_path: Path | None = None,
    data_dirs: list[Path] | None = None,
    primary_split: str | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    root = _root()
    cards_path = model_cards_path or root / "config/paper_model_cards.json"
    model_cards = load_model_cards(cards_path)
    selection_protocol = load_selection_protocol(cards_path)
    data_dirs = data_dirs or [
        root / "hf_release/mempatch/train",
        root / "hf_release/mempatch/validation",
        root / "hf_release/mempatch/test",
    ]
    pred_files = discover_runs(results_dir)
    runs = [load_eval_run(p) for p in pred_files]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"runs": len(runs), "outputs": []}
    if selection_protocol:
        (out_dir / "model_selection_protocol.json").write_text(
            json.dumps(selection_protocol, indent=2) + "\n", encoding="utf-8"
        )
        manifest["outputs"].append("model_selection_protocol.json")
        manifest["model_selection"] = selection_protocol

    t1 = build_table1_dataset_stats(data_dirs)
    (out_dir / "table1_dataset_stats.json").write_text(json.dumps(t1, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"].append("table1_dataset_stats.json")

    t2 = build_table2_model_setup(runs, model_cards)
    write_csv(
        out_dir / "table2_model_setup.csv",
        t2,
        [
            "model",
            "display_name",
            "family",
            "model_generation",
            "paper_role",
            "selection_note",
            "total_params_b",
            "active_params_b",
            "context_length",
            "license",
            "quantization",
            "eval_path",
            "eval_variant",
            "eval_split",
        ],
    )
    manifest["outputs"].append("table2_model_setup.csv")

    t3 = build_table3_main_results(runs)
    write_csv(
        out_dir / "table3_main_results.csv",
        [r for r in t3 if "category_type" not in r],
        ["model", "path", "variant", "split", "count", *HEADLINE_KEYS],
    )
    manifest["outputs"].append("table3_main_results.csv")

    t4 = build_table4_evaluator_reliability()
    (out_dir / "table4_evaluator_reliability.json").write_text(json.dumps(t4, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"].append("table4_evaluator_reliability.json")

    fig1 = build_fig1_pipeline_mermaid()
    (out_dir / "fig1_pipeline.mmd").write_text(fig1 + "\n", encoding="utf-8")
    manifest["outputs"].append("fig1_pipeline.mmd")

    fig2 = build_fig2_leaderboard(runs, split=primary_split, n_boot=n_boot)
    (out_dir / "fig2_leaderboard_ci.json").write_text(json.dumps(fig2, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"].append("fig2_leaderboard_ci.json")

    fig3 = build_fig3_capability_heatmap(runs, split=primary_split)
    (out_dir / "fig3_capability_heatmap.json").write_text(json.dumps(fig3, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"].append("fig3_capability_heatmap.json")

    fig4 = build_fig4_difficulty_breakdown(runs, split=primary_split)
    (out_dir / "fig4_difficulty_breakdown.json").write_text(json.dumps(fig4, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"].append("fig4_difficulty_breakdown.json")

    fig5 = build_fig5_cost_performance(runs, model_cards, split=primary_split)
    (out_dir / "fig5_cost_performance.json").write_text(json.dumps(fig5, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"].append("fig5_cost_performance.json")

    fig6 = build_fig6_pairwise_winrate(runs, split=primary_split)
    (out_dir / "fig6_pairwise_winrate.json").write_text(json.dumps(fig6, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"].append("fig6_pairwise_winrate.json")

    fig7 = build_fig7_failure_taxonomy(runs)
    (out_dir / "fig7_failure_taxonomy.json").write_text(json.dumps(fig7, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"].append("fig7_failure_taxonomy.json")

    paper_map = {
        "main_text_minimum": [
            "fig1_pipeline.mmd",
            "fig2_leaderboard_ci.json",
            "fig3_capability_heatmap.json",
            "fig5_cost_performance.json",
            "table3_main_results.csv",
            "table4_evaluator_reliability.json",
        ],
        "appendix": [
            "fig4_difficulty_breakdown.json",
            "fig6_pairwise_winrate.json",
            "fig7_failure_taxonomy.json",
            "table1_dataset_stats.json",
            "table2_model_setup.csv",
        ],
    }
    manifest["paper_map"] = paper_map
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
