#!/usr/bin/env python3
"""Publication-quality MemPatch method-paper figures (Direct vs MemPatch).

Produces fig2–fig5 under figures/paper/ as vector PDF + high-res PNG + CSV.
Reuses data-loading helpers from benchmark_paper_export.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_paper_export import (  # noqa: E402
    discover_runs,
    load_eval_run,
    load_model_cards,
    write_csv,
)

MODEL_ORDER = ["qwen3_14b", "gemma3_12b", "mistral_nemo_12b", "llama3_1_8b"]

DISPLAY_NAMES = {
    "qwen3_14b": "Qwen3-14B",
    "gemma3_12b": "Gemma-3-12B",
    "mistral_nemo_12b": "Mistral-Nemo-12B",
    "llama3_1_8b": "Llama-3.1-8B",
}

METHOD_LABEL = {"A": "Direct", "B": "MemPatch"}

FIG2_METRICS = (
    ("decision_macro_f1", "Decision F1"),
    ("memory_state_accuracy", "Memory State Acc."),
    ("evidence_f1", "Evidence F1"),
    ("failure_diagnosis_accuracy", "Diagnosis Acc."),
    ("joint_revision_success", "Joint Success"),
)

FIG3_METRICS = FIG2_METRICS + (
    ("structural_revision_success", "Structural Success"),
    ("answer_state_consistency", "Answer Consistency"),
)

ERROR_METRICS = (
    ("latest_event_shortcut_failure_rate", "Latest-event shortcut"),
    ("overcitation_rate", "Over-citation"),
    ("scope_leakage_rate", "Scope leakage"),
    ("under_update_rate", "Under-update"),
    ("policy_violation_rate", "Policy violation"),
    ("wrong_source_attribution_rate", "Wrong source"),
    ("memory_hallucination_rate", "Memory hallucination"),
    ("format_failure_rate", "Format failure"),
)

COLOR_DIRECT = "#7B8FA1"
COLOR_MEMPATCH = "#D4A056"
HATCH_DIRECT = "//"
HATCH_MEMPATCH = "\\\\"


@dataclass
class PlotState:
    generated: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    input_files: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "local/results/paper")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=ROOT / "local/train_data/paper/test500/scenarios.jsonl",
    )
    parser.add_argument("--model-cards", type=Path, default=ROOT / "config/paper_model_cards.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/paper")
    parser.add_argument("--split", default="test500")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "local/logs/paper")
    return parser.parse_args(argv)


def setup_matplotlib() -> Any:
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8
    import matplotlib.pyplot as plt

    return plt


def apply_style(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D0D0D0", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def display_name(model: str, cards: dict[str, Any]) -> str:
    if model in DISPLAY_NAMES:
        return DISPLAY_NAMES[model]
    return str(cards.get(model, {}).get("display_name", model))


def discover_metrics_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*_metrics.json"))


def load_metrics_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_value(meta: dict[str, Any], key: str) -> float | None:
    for block in ("headline_metrics", "all_metrics"):
        section = meta.get(block) or {}
        if key in section and section[key] is not None:
            return float(section[key])
    return None


def build_run_metrics_df(
    results_dir: Path,
    split: str,
    cards: dict[str, Any],
    state: PlotState,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover_metrics_files(results_dir):
        state.input_files.append(str(path))
        meta = load_metrics_json(path)
        run_split = str(meta.get("split") or "")
        if split and run_split and run_split != split:
            continue
        model = str(meta.get("model") or path.parent.name)
        path_letter = str(meta.get("path") or "?")
        variant = str(meta.get("variant") or "?")
        method = METHOD_LABEL.get(path_letter, path_letter)
        row: dict[str, Any] = {
            "model": model,
            "display_name": display_name(model, cards),
            "path": path_letter,
            "method": method,
            "variant": variant,
            "split": run_split,
            "count": meta.get("count"),
            "metrics_path": str(path),
        }
        for block in ("headline_metrics", "all_metrics"):
            for k, v in (meta.get(block) or {}).items():
                if isinstance(v, (int, float)):
                    row[k] = float(v)
        rows.append(row)
    return rows


def scenario_features(scenarios: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from benchmark.general_taxonomy import canonical_hidden_gold_fields

    out: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        sid = str(s["scenario_id"])
        pub = s.get("public_input") or {}
        events = pub.get("event_trace") or []
        memories = pub.get("initial_memory") or []
        gold = canonical_hidden_gold_fields(s.get("hidden_gold") or {})
        factors = s.get("difficulty_factors") or {}
        token_chars = 0
        for ev in events:
            token_chars += len(str(ev.get("text", "")))
        for mem in memories:
            token_chars += len(str(mem.get("text", "")))
        out[sid] = {
            "difficulty": s.get("difficulty"),
            "num_events": factors.get("num_events", len(events)),
            "num_memories": factors.get("num_memories", len(memories)),
            "num_expected_evidence": len(gold.get("expected_evidence_event_ids") or []),
            "num_gold_memory_states": len(gold.get("expected_memory_state") or {}),
            "approx_token_count": max(1, token_chars // 4),
        }
    return out


def build_per_case_df(
    results_dir: Path,
    scenarios_path: Path,
    split: str,
    cards: dict[str, Any],
    state: PlotState,
) -> list[dict[str, Any]]:
    if not scenarios_path.is_file():
        state.warnings.append(f"Scenarios file missing: {scenarios_path}")
        return []

    from benchmark.api import load_scenarios

    scenarios = load_scenarios(scenarios_path)
    feat = scenario_features(scenarios)
    rows: list[dict[str, Any]] = []
    pred_files = [
        p
        for p in discover_runs(results_dir)
        if split in p.name or split in str(p)
    ]
    if not pred_files:
        pred_files = discover_runs(results_dir)

    for pred_path in pred_files:
        state.input_files.append(str(pred_path))
        try:
            run = load_eval_run(pred_path, scenarios_path)
        except Exception as exc:
            state.warnings.append(f"Failed to load {pred_path}: {exc}")
            continue
        if split and run.split != split:
            continue
        method = METHOD_LABEL.get(run.path, run.path)
        for scored in run.scored:
            sid = str(scored["scenario_id"])
            sf = feat.get(sid, {})
            m = scored.get("metrics") or {}
            rows.append(
                {
                    "scenario_id": sid,
                    "model": run.model,
                    "display_name": display_name(run.model, cards),
                    "method": method,
                    "path": run.path,
                    "variant": run.variant,
                    "split": run.split,
                    "difficulty": scored.get("difficulty") or sf.get("difficulty"),
                    "num_events": sf.get("num_events"),
                    "num_memories": sf.get("num_memories"),
                    "num_expected_evidence": sf.get("num_expected_evidence"),
                    "approx_token_count": sf.get("approx_token_count"),
                    "joint_revision_success": float(m.get("joint_revision_success", 0.0)),
                    "memory_state_accuracy": float(m.get("memory_state_accuracy", 0.0)),
                    "evidence_f1": float(m.get("evidence_f1", 0.0)),
                    "decision_macro_f1": float(m.get("decision_macro_f1", 0.0)),
                }
            )
    return rows


def build_model_metadata_df(cards: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        card = cards.get(model, {})
        rows.append(
            {
                "model": model,
                "display_name": display_name(model, cards),
                "family": card.get("family"),
                "total_params_b": card.get("total_params_b"),
                "active_params_b": card.get("active_params_b"),
                "latency_ms_per_case": card.get("latency_ms_per_case"),
            }
        )
    return rows


def get_run_row(rows: list[dict[str, Any]], model: str, path: str) -> dict[str, Any] | None:
    for row in rows:
        if row["model"] == model and row["path"] == path:
            return row
    return None


def save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")


def _bar_value_label(ax: Any, x: float, height: float, text: str) -> None:
    if height >= 0.08:
        ax.text(x, height + 0.02, text, ha="center", va="bottom", fontsize=6.5, color="#333333")


def plot_fig2(run_rows: list[dict[str, Any]], out_dir: Path, state: PlotState, plt: Any) -> None:
    models_present = [m for m in MODEL_ORDER if get_run_row(run_rows, m, "A") and get_run_row(run_rows, m, "B")]
    if not models_present:
        state.skipped.append(("fig2_direct_vs_mempatch_metrics", "No paired Direct/MemPatch runs found"))
        return

    n = len(models_present)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.6 * nrows), squeeze=False)
    csv_rows: list[dict[str, Any]] = []

    for idx, model in enumerate(models_present):
        ax = axes[idx // ncols][idx % ncols]
        apply_style(ax)
        row_a = get_run_row(run_rows, model, "A")
        row_b = get_run_row(run_rows, model, "B")
        assert row_a and row_b

        labels = [lbl for _, lbl in FIG2_METRICS]
        x = np.arange(len(labels))
        width = 0.36
        vals_a = [float(row_a.get(k, 0.0) or 0.0) for k, _ in FIG2_METRICS]
        vals_b = [float(row_b.get(k, 0.0) or 0.0) for k, _ in FIG2_METRICS]

        ax.bar(
            x - width / 2,
            vals_a,
            width,
            label="Direct",
            color=COLOR_DIRECT,
            edgecolor="#5A6B7A",
            hatch=HATCH_DIRECT,
            linewidth=0.6,
        )
        ax.bar(
            x + width / 2,
            vals_b,
            width,
            label="MemPatch",
            color=COLOR_MEMPATCH,
            edgecolor="#A67C2E",
            hatch=HATCH_MEMPATCH,
            linewidth=0.6,
        )

        for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
            _bar_value_label(ax, x[i] - width / 2, va, f"{va:.2f}")
            _bar_value_label(ax, x[i] + width / 2, vb, f"{vb:.2f}")

        joint_gain = vals_b[-1] - vals_a[-1]
        if abs(joint_gain) >= 0.05:
            ax.text(
                0.98,
                0.96,
                f"+{joint_gain:.2f} Joint" if joint_gain > 0 else f"{joint_gain:.2f} Joint",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="#8B5A14",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(display_name(model, {}))

        for (key, lbl), va, vb in zip(FIG2_METRICS, vals_a, vals_b):
            csv_rows.append(
                {
                    "model": model,
                    "display_name": display_name(model, {}),
                    "metric": key,
                    "metric_label": lbl,
                    "direct": va,
                    "mempatch": vb,
                    "gain": vb - va,
                }
            )

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Structured RMI Metrics: Direct vs MemPatch", y=1.06, fontsize=10)
    fig.tight_layout()

    stem = "fig2_direct_vs_mempatch_metrics"
    save_figure(fig, out_dir, stem)
    plt.close(fig)
    write_csv(
        out_dir / f"{stem}.csv",
        csv_rows,
        ["model", "display_name", "metric", "metric_label", "direct", "mempatch", "gain"],
    )
    state.generated.append(stem)


def plot_fig3(run_rows: list[dict[str, Any]], out_dir: Path, state: PlotState, plt: Any) -> None:
    """One panel per metric; models on x-axis (cross-model robustness)."""
    models_present = [m for m in MODEL_ORDER if get_run_row(run_rows, m, "A") and get_run_row(run_rows, m, "B")]
    if not models_present:
        state.skipped.append(("fig3_model_metric_profile", "No paired runs"))
        return

    metrics = FIG3_METRICS[:5]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 2.8), squeeze=False)
    csv_rows: list[dict[str, Any]] = []
    x = np.arange(len(models_present))
    width = 0.34
    tick_labels = [display_name(m, {}) for m in models_present]

    for ax, (key, lbl) in zip(axes[0], metrics):
        apply_style(ax)
        vals_a = [float(get_run_row(run_rows, m, "A").get(key, 0.0) or 0.0) for m in models_present]
        vals_b = [float(get_run_row(run_rows, m, "B").get(key, 0.0) or 0.0) for m in models_present]
        ax.bar(x - width / 2, vals_a, width, color=COLOR_DIRECT, hatch=HATCH_DIRECT, edgecolor="#5A6B7A", linewidth=0.5, label="Direct")
        ax.bar(x + width / 2, vals_b, width, color=COLOR_MEMPATCH, hatch=HATCH_MEMPATCH, edgecolor="#A67C2E", linewidth=0.5, label="MemPatch")
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(lbl, fontsize=8.5)
        for m, va, vb in zip(models_present, vals_a, vals_b):
            csv_rows.append(
                {
                    "metric": key,
                    "metric_label": lbl,
                    "model": m,
                    "display_name": display_name(m, {}),
                    "direct": va,
                    "mempatch": vb,
                    "gain": vb - va,
                }
            )

    axes[0][0].set_ylabel("Score")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Cross-Model Metric Profile", y=1.12, fontsize=10)
    fig.tight_layout()

    stem = "fig3_model_metric_profile"
    save_figure(fig, out_dir, stem)
    plt.close(fig)
    write_csv(
        out_dir / f"{stem}.csv",
        csv_rows,
        ["metric", "metric_label", "model", "display_name", "direct", "mempatch", "gain"],
    )
    state.generated.append(stem)


def plot_fig4(run_rows: list[dict[str, Any]], out_dir: Path, state: PlotState, plt: Any) -> None:
    """Error-rate breakdown: aggregate mean across four models."""
    models_present = [m for m in MODEL_ORDER if get_run_row(run_rows, m, "A") and get_run_row(run_rows, m, "B")]
    if not models_present:
        state.skipped.append(("fig4_error_breakdown", "No paired runs"))
        return

    available: list[tuple[str, str]] = []
    for key, lbl in ERROR_METRICS:
        has_any = any(
            get_run_row(run_rows, m, path) and get_run_row(run_rows, m, path).get(key) is not None
            for m in models_present
            for path in ("A", "B")
        )
        if has_any:
            available.append((key, lbl))
    if not available:
        state.skipped.append(("fig4_error_breakdown", "No error metrics in metrics JSON"))
        return

    direct_vals: list[float] = []
    memp_vals: list[float] = []
    labels: list[str] = []
    csv_rows: list[dict[str, Any]] = []

    for key, lbl in available:
        d_list: list[float] = []
        b_list: list[float] = []
        for m in models_present:
            ra = get_run_row(run_rows, m, "A")
            rb = get_run_row(run_rows, m, "B")
            if ra and ra.get(key) is not None:
                d_list.append(float(ra[key]))
            if rb and rb.get(key) is not None:
                b_list.append(float(rb[key]))
        if not d_list or not b_list:
            continue
        d_mean = float(np.mean(d_list))
        b_mean = float(np.mean(b_list))
        direct_vals.append(d_mean)
        memp_vals.append(b_mean)
        labels.append(lbl)
        csv_rows.append(
            {
                "error_metric": key,
                "error_label": lbl,
                "direct_mean": d_mean,
                "mempatch_mean": b_mean,
                "delta": b_mean - d_mean,
                "aggregation": "mean_over_models",
                "models": ",".join(models_present),
            }
        )

    if not labels:
        state.skipped.append(("fig4_error_breakdown", "Error metrics all null"))
        return

    y = np.arange(len(labels))
    height = 0.34
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.38 * len(labels) + 1.2)))
    apply_style(ax)
    ax.grid(axis="x", color="#D0D0D0", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.barh(y - height / 2, direct_vals, height, label="Direct", color=COLOR_DIRECT, hatch=HATCH_DIRECT, edgecolor="#5A6B7A", linewidth=0.5)
    ax.barh(y + height / 2, memp_vals, height, label="MemPatch", color=COLOR_MEMPATCH, hatch=HATCH_MEMPATCH, edgecolor="#A67C2E", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, min(1.0, max(direct_vals + memp_vals) * 1.25 + 0.05))
    ax.set_xlabel("Rate (↓ lower is better)")
    ax.set_title("Error-Mode Rates: Direct vs MemPatch (4-model mean)")
    ax.legend(loc="lower right", frameon=False)

    for i, (vd, vm) in enumerate(zip(direct_vals, memp_vals)):
        ax.text(vd + 0.01, y[i] - height / 2, f"{vd:.2f}", va="center", fontsize=7, color="#444444")
        ax.text(vm + 0.01, y[i] + height / 2, f"{vm:.2f}", va="center", fontsize=7, color="#444444")

    fig.tight_layout()
    stem = "fig4_error_breakdown"
    save_figure(fig, out_dir, stem)
    plt.close(fig)
    write_csv(
        out_dir / f"{stem}.csv",
        csv_rows,
        ["error_metric", "error_label", "direct_mean", "mempatch_mean", "delta", "aggregation", "models"],
    )
    state.generated.append(stem)


def search_latency(
    run_rows: list[dict[str, Any]],
    cards: dict[str, Any],
    results_dir: Path,
    logs_dir: Path,
    state: PlotState,
) -> list[dict[str, Any]]:
    """Return latency rows if recoverable; else empty."""
    points: list[dict[str, Any]] = []

    latency_keys = (
        "latency_ms_per_case",
        "latency_s_per_case",
        "mean_latency_ms",
        "mean_latency_s",
        "inference_latency_ms",
    )

    for row in run_rows:
        model = row["model"]
        method = row["method"]
        lat_s: float | None = None
        source = ""

        for key in latency_keys:
            if row.get(key) is not None:
                val = float(row[key])
                lat_s = val / 1000.0 if "ms" in key else val
                source = f"metrics:{key}"
                break

        card = cards.get(model, {})
        if lat_s is None and card.get("latency_ms_per_case") is not None:
            lat_s = float(card["latency_ms_per_case"]) / 1000.0
            source = "model_card:latency_ms_per_case"

        metrics_path = row.get("metrics_path")
        if lat_s is None and metrics_path:
            meta = load_metrics_json(Path(metrics_path))
            for key in latency_keys:
                for block in (meta, meta.get("runtime") or {}, meta.get("inference_stats") or {}):
                    if isinstance(block, dict) and block.get(key) is not None:
                        val = float(block[key])
                        lat_s = val / 1000.0 if "ms" in key else val
                        source = f"metrics_json:{key}"
                        break
                if lat_s is not None:
                    break

        pred_stem = Path(str(metrics_path or "")).name.replace("_metrics.json", "")
        pred_path = Path(str(metrics_path or "")).parent / f"{pred_stem}_predictions.jsonl"
        if lat_s is None and pred_path.is_file():
            lats: list[float] = []
            with pred_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    for key in ("latency_s", "latency_ms", "elapsed_s", "elapsed_ms"):
                        if key in rec.get("metadata", {}) or key in rec:
                            raw = rec.get("metadata", {}).get(key, rec.get(key))
                            if raw is not None:
                                lats.append(float(raw) / (1000.0 if "ms" in key else 1.0))
            if lats:
                lat_s = float(np.mean(lats))
                source = "predictions:per_case_mean"

        if lat_s is not None:
            points.append(
                {
                    "model": model,
                    "display_name": display_name(model, cards),
                    "method": method,
                    "path": row["path"],
                    "latency_s_per_case": lat_s,
                    "joint_revision_success": float(row.get("joint_revision_success", 0.0) or 0.0),
                    "memory_state_accuracy": float(row.get("memory_state_accuracy", 0.0) or 0.0),
                    "source": source,
                }
            )

    if not points and logs_dir.is_dir():
        for log in logs_dir.rglob("*.log"):
            text = log.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"latency|elapsed|seconds per case", text, re.I):
                state.warnings.append(f"Found latency-like text in {log} but no structured parse implemented")

    return points


def plot_fig6_latency_frontier(
    run_rows: list[dict[str, Any]],
    cards: dict[str, Any],
    results_dir: Path,
    logs_dir: Path,
    out_dir: Path,
    state: PlotState,
    plt: Any,
) -> None:
    points = search_latency(run_rows, cards, results_dir, logs_dir, state)
    models_needed = set(MODEL_ORDER)
    have = {(p["model"], p["method"]) for p in points}

    if len(have) < len(models_needed) * 2:
        reason = (
            "No per-case latency in metrics JSON, predictions metadata, model cards "
            f"(latency_ms_per_case is null), or parseable logs under {logs_dir}. "
            f"Recovered {len(points)} of {len(models_needed)*2} model×method points."
        )
        state.skipped.append(("fig6_accuracy_latency_frontier", reason))
        warnings.warn(f"Skipping fig6 latency frontier: {reason}")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    apply_style(ax)
    ax.grid(True, color="#D0D0D0", linewidth=0.5, alpha=0.7)

    marker_map = {"Direct": "s", "MemPatch": "o"}
    color_map = {"Direct": COLOR_DIRECT, "MemPatch": COLOR_MEMPATCH}
    csv_rows = points.copy()

    for model in MODEL_ORDER:
        pts = [p for p in points if p["model"] == model]
        if len(pts) < 2:
            continue
        pts = sorted(pts, key=lambda p: p["method"])
        xs = [p["latency_s_per_case"] for p in pts]
        ys = [p["joint_revision_success"] for p in pts]
        ax.plot(xs, ys, color="#BBBBBB", linewidth=0.8, zorder=1)
        for p in pts:
            ax.scatter(
                p["latency_s_per_case"],
                p["joint_revision_success"],
                marker=marker_map.get(p["method"], "o"),
                s=55,
                color=color_map.get(p["method"], "#666666"),
                edgecolors="#333333",
                linewidths=0.5,
                zorder=2,
                label=p["method"] if model == MODEL_ORDER[0] else None,
            )
            ax.annotate(
                f"{p['display_name']}\n{p['method']}",
                (p["latency_s_per_case"], p["joint_revision_success"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6.5,
            )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="lower right", frameon=False)
    ax.set_xlabel("Latency (s / case)")
    ax.set_ylabel("Joint Revision Success")
    ax.set_title("Accuracy–Latency Frontier")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    stem = "fig6_accuracy_latency_frontier"
    save_figure(fig, out_dir, stem)
    plt.close(fig)
    write_csv(
        out_dir / f"{stem}.csv",
        csv_rows,
        [
            "model",
            "display_name",
            "method",
            "path",
            "latency_s_per_case",
            "joint_revision_success",
            "memory_state_accuracy",
            "source",
        ],
    )
    state.generated.append(stem)


def _quantile_bin_edges(values: list[float], n_bins: int = 5) -> np.ndarray:
    arr = np.array(values, dtype=float)
    edges = np.unique(np.percentile(arr, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(arr.min(), arr.max(), min(n_bins + 1, len(np.unique(arr))))
    return edges


def _assign_bin(value: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    return max(0, min(idx, len(edges) - 2))


def plot_fig5(case_rows: list[dict[str, Any]], out_dir: Path, state: PlotState, plt: Any) -> None:
    if not case_rows:
        state.skipped.append(("fig5_case_difficulty_scatter", "No per-case scored predictions"))
        return

    y_key = "joint_revision_success"
    y_label = "Joint Success (binned mean)"

    # Prefer variables with cross-scenario variation; fall back to quantile bins on token count.
    x_key = "approx_token_count"
    x_label = "Approx. Input Tokens (quantile bins)"
    raw_vals = [float(r[x_key]) for r in case_rows if r.get(x_key) is not None]
    if len(set(raw_vals)) < 3:
        for alt, lbl in (
            ("num_expected_evidence", "Expected Evidence Count"),
            ("num_events", "Event Count"),
            ("num_memories", "Memory Count"),
        ):
            alt_vals = [float(r[alt]) for r in case_rows if r.get(alt) is not None]
            if len(set(alt_vals)) >= 2:
                x_key, x_label = alt, f"{lbl} (binned)"
                raw_vals = alt_vals
                break

    use_quantiles = len(set(raw_vals)) >= 3
    edges = _quantile_bin_edges(raw_vals, n_bins=5) if use_quantiles else np.array([])

    if use_quantiles:
        for r in case_rows:
            if r.get(x_key) is not None:
                b = _assign_bin(float(r[x_key]), edges)
                r["_bin_idx"] = b
                r["_bin_center"] = float((edges[b] + edges[b + 1]) / 2.0)
    else:
        state.skipped.append(
            (
                "fig5_case_difficulty_scatter",
                "Insufficient difficulty variation (all scenarios share the same event/memory/evidence counts)",
            )
        )
        return

    bin_indices = sorted({int(r["_bin_idx"]) for r in case_rows if "_bin_idx" in r})
    csv_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    apply_style(ax)

    for method, color, marker, ls in (
        ("Direct", COLOR_DIRECT, "x", "--"),
        ("MemPatch", COLOR_MEMPATCH, "o", "-"),
    ):
        xs_plot: list[float] = []
        ys_plot: list[float] = []
        y_lo: list[float] = []
        y_hi: list[float] = []
        for b in bin_indices:
            vals = [
                float(r[y_key])
                for r in case_rows
                if r.get("method") == method and r.get("_bin_idx") == b
            ]
            if not vals:
                continue
            center = float(np.mean([r["_bin_center"] for r in case_rows if r.get("_bin_idx") == b]))
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            xs_plot.append(center)
            ys_plot.append(mean)
            y_lo.append(max(0.0, mean - sem))
            y_hi.append(min(1.0, mean + sem))
            csv_rows.append(
                {
                    "method": method,
                    "bin_index": b,
                    "bin_center": center,
                    "x_key": x_key,
                    "n_cases": len(vals),
                    "mean_joint_success": mean,
                    "sem": sem,
                }
            )
        if xs_plot:
            order = np.argsort(xs_plot)
            xs_plot = [xs_plot[i] for i in order]
            ys_plot = [ys_plot[i] for i in order]
            y_lo = [y_lo[i] for i in order]
            y_hi = [y_hi[i] for i in order]
            ax.plot(xs_plot, ys_plot, color=color, marker=marker, linestyle=ls, linewidth=1.8, label=method)
            ax.fill_between(xs_plot, y_lo, y_hi, color=color, alpha=0.12)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Robustness vs Input Complexity (all models, aggregate)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()

    stem = "fig5_case_difficulty_scatter"
    save_figure(fig, out_dir, stem)
    plt.close(fig)
    write_csv(
        out_dir / f"{stem}.csv",
        csv_rows,
        ["method", "bin_index", "bin_center", "x_key", "n_cases", "mean_joint_success", "sem"],
    )
    state.generated.append(stem)


def try_plot_training_curve(logs_dir: Path, out_dir: Path, state: PlotState, plt: Any) -> None:
    candidates = list(logs_dir.rglob("*train*.json")) + list(logs_dir.rglob("*loss*.csv"))
    if not candidates:
        state.skipped.append(
            ("fig7_lora_training_curve", f"No training logs under {logs_dir}"),
        )
        return
    state.skipped.append(
        ("fig7_lora_training_curve", f"Found {len(candidates)} files but no standard training-curve schema"),
    )


def write_readme(out_dir: Path, state: PlotState, args: argparse.Namespace) -> None:
    lines = [
        "# MemPatch Paper Figures",
        "",
        "Generated by `scripts/plot_mempatch_paper_figures.py`.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=.:src .venv/bin/python scripts/plot_mempatch_paper_figures.py \\",
        f"  --results-dir {args.results_dir} \\",
        f"  --scenarios {args.scenarios} \\",
        f"  --model-cards {args.model_cards} \\",
        f"  --out-dir {args.out_dir} \\",
        f"  --split {args.split}",
        "```",
        "",
        "## Generated figures",
        "",
    ]
    if state.generated:
        for stem in state.generated:
            lines.append(f"- `{stem}.pdf` / `.png` / `.csv`")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Skipped figures", ""])
    if state.skipped:
        for stem, reason in state.skipped:
            lines.append(f"- **{stem}**: {reason}")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Warnings", ""])
    if state.warnings:
        for w in state.warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- _(none)_")

    unique_inputs = sorted(set(state.input_files))
    lines.extend(["", "## Input files", ""])
    for p in unique_inputs[:40]:
        lines.append(f"- `{p}`")
    if len(unique_inputs) > 40:
        lines.append(f"- … and {len(unique_inputs) - 40} more")

    lines.extend(
        [
            "",
            "## Story",
            "",
            "Figures emphasize **Direct** (path A, direct-response baseline) vs **MemPatch** (path B, revision module),",
            "not a model leaderboard. Primary metrics: decision F1, memory-state accuracy, evidence F1,",
            "diagnosis accuracy, and joint revision success.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = PlotState()

    if not args.results_dir.is_dir():
        print(f"ERROR: results dir missing: {args.results_dir}", file=sys.stderr)
        return 1

    cards = load_model_cards(args.model_cards)
    if args.model_cards.is_file():
        state.input_files.append(str(args.model_cards))
    if args.scenarios.is_file():
        state.input_files.append(str(args.scenarios))

    run_rows = build_run_metrics_df(args.results_dir, args.split, cards, state)
    if not run_rows:
        state.warnings.append(f"No metrics JSON found under {args.results_dir} for split={args.split}")

    write_csv(
        args.out_dir / "run_metrics.csv",
        run_rows,
        ["model", "display_name", "path", "method", "variant", "split", "count", "joint_revision_success", "memory_state_accuracy"],
    )

    case_rows = build_per_case_df(args.results_dir, args.scenarios, args.split, cards, state)
    if case_rows:
        write_csv(
            args.out_dir / "per_case_scores.csv",
            case_rows,
            [
                "scenario_id",
                "model",
                "display_name",
                "method",
                "joint_revision_success",
                "memory_state_accuracy",
                "num_events",
                "num_expected_evidence",
                "approx_token_count",
            ],
        )

    meta_rows = build_model_metadata_df(cards)
    write_csv(args.out_dir / "model_metadata.csv", meta_rows, list(meta_rows[0].keys()) if meta_rows else ["model"])

    plt = setup_matplotlib()

    plot_fig2(run_rows, args.out_dir, state, plt)
    plot_fig3(run_rows, args.out_dir, state, plt)
    plot_fig4(run_rows, args.out_dir, state, plt)
    plot_fig5(case_rows, args.out_dir, state, plt)
    plot_fig6_latency_frontier(run_rows, cards, args.results_dir, args.logs_dir, args.out_dir, state, plt)
    try_plot_training_curve(args.logs_dir, args.out_dir, state, plt)

    write_readme(args.out_dir, state, args)

    print(f"Output directory: {args.out_dir}")
    print(f"Generated: {', '.join(state.generated) or '(none)'}")
    for stem, reason in state.skipped:
        print(f"Skipped {stem}: {reason}")
    for w in state.warnings:
        print(f"Warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
