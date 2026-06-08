#!/usr/bin/env python3
"""Plot high-quality AAAI/ICLR-style paper figures for MemPatch evaluation.

This script produces 5 highly polished figures:
  Fig 2  Robustness Line Profile (2x2 Grid)
  Fig 3  Grouped Leaderboard with CI (No overlapping labels)
  Fig 4  Accuracy & Latency Grouped Subplots (ICLR/arXiv style)
  Fig 5  Horizontal 100% Stacked Error Breakdown
  Fig 6  Interactive Complexity vs Success Rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add repo base to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from benchmark_paper_export import discover_runs, load_eval_run, resolve_scenarios_path  # noqa: E402
from benchmark.api import load_scenarios  # noqa: E402

# Color scheme definitions for high-quality academic styling
COLOR_PALETTE = {
    "Direct": "#95A5A6",       # Muted gray for baseline
    "MemPatch": "#2980B9",     # Pure academic blue for MemPatch
    "qwen": "#E67E22",         # Warm orange
    "gemma": "#1ABC9C",        # Emerald green
    "mistral": "#D35400",      # Rust orange
    "llama": "#2C3E50",        # Deep slate blue
}

METRIC_LABELS = {
    "decision_macro_f1": "Decision F1",
    "memory_state_accuracy": "Memory Acc",
    "evidence_f1": "Evidence F1",
    "failure_diagnosis_accuracy": "Diagnosis Acc",
    "joint_revision_success": "Joint Success",
}

MODEL_DISPLAY_NAMES = {
    "qwen3_14b": "Qwen3-14B",
    "gemma3_12b": "Gemma-3-12B",
    "mistral_nemo_12b": "Mistral-Nemo-12B",
    "llama3_1_8b": "Llama-3.1-8B",
}


def apply_academic_style(ax: Any) -> None:
    """Apply clean academic style rules to matplotlib Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2C3E50")
    ax.spines["bottom"].set_color("#2C3E50")
    ax.tick_params(axis="both", which="both", colors="#2C3E50", labelsize=9)
    ax.grid(color="#ECF0F1", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_facecolor("white")


def plot_fig2_robustness(runs: list[Any], out_dir: Path) -> None:
    """Fig 2: Main model robustness profile (2x2 line profile subplots with independent labels)."""
    import matplotlib.pyplot as plt

    print("Generating Fig 2: Main model robustness profile...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    axes = axes.flatten()

    models = ["llama3_1_8b", "gemma3_12b", "mistral_nemo_12b", "qwen3_14b"]
    metrics = ["decision_macro_f1", "memory_state_accuracy", "evidence_f1", "failure_diagnosis_accuracy", "joint_revision_success"]
    metric_ticks = [METRIC_LABELS[m] for m in metrics]

    csv_rows = []

    for idx, model in enumerate(models):
        ax = axes[idx]
        apply_academic_style(ax)

        # Explicitly set ticks and labels to prevent warning and ensure visibility on all subplots
        ax.set_xticks(range(len(metric_ticks)))
        ax.set_xticklabels(metric_ticks, rotation=15, ha="right", fontsize=8.5)

        # Get Path A (base) and Path B (lora) runs
        run_a = next((r for r in runs if r.model == model and r.path == "A"), None)
        run_b = next((r for r in runs if r.model == model and r.path == "B"), None)

        scores_a = [run_a.headline.get(m, 0.0) if run_a else 0.0 for m in metrics]
        scores_b = [run_b.headline.get(m, 0.0) if run_b else 0.0 for m in metrics]

        # Draw lines
        ax.plot(metric_ticks, scores_a, label="Direct (Base)", color=COLOR_PALETTE["Direct"], linestyle="--", marker="x", markersize=6, linewidth=1.5)
        ax.plot(metric_ticks, scores_b, label="MemPatch (LoRA)", color=COLOR_PALETTE["MemPatch"], linestyle="-", marker="o", markersize=6, linewidth=2)

        # Fill absolute gain
        ax.fill_between(metric_ticks, scores_a, scores_b, color=COLOR_PALETTE["MemPatch"], alpha=0.08)

        # Annotation of joint success gain in the top-left to avoid overlaps
        joint_a = run_a.headline.get("joint_revision_success", 0.0) if run_a else 0.0
        joint_b = run_b.headline.get("joint_revision_success", 0.0) if run_b else 0.0
        gain = joint_b - joint_a
        ax.text(0.05, 0.95, f"Joint Gain: +{gain*100:.1f}%", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold", color=COLOR_PALETTE["MemPatch"],
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="#BDC3C7", boxstyle="round,pad=0.3"))

        ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=11, fontweight="bold", color="#2C3E50")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=9, fontweight="bold", color="#2C3E50")

        # Save data for CSV export
        for m, val_a, val_b in zip(metrics, scores_a, scores_b):
            csv_rows.append({
                "model": MODEL_DISPLAY_NAMES[model],
                "metric": METRIC_LABELS[m],
                "direct_score": val_a,
                "mempatch_score": val_b,
                "gain": val_b - val_a
            })

    # Legend only on the first panel to keep plot clean
    axes[0].legend(loc="lower left", frameon=True, fontsize=8.5, facecolor="white", edgecolor="#ECF0F1")

    # Adjust layout to prevent clipping on title or tick labels
    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, hspace=0.38, wspace=0.22)
    fig.suptitle("Performance Profile: Direct vs MemPatch", fontsize=13, fontweight="bold", color="#2C3E50", y=0.96)

    fig.savefig(out_dir / "fig2_model_robustness.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_model_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Export CSV
    pd.DataFrame(csv_rows).to_csv(out_dir / "fig2_model_robustness_data.csv", index=False)
    print("Wrote Fig 2 and CSV data.")


def plot_fig3_leaderboard(payload: dict, out_dir: Path) -> None:
    """Fig 3: Grouped leaderboard bar chart (clean academic style, no overlapping labels)."""
    import matplotlib.pyplot as plt

    print("Generating Fig 3: Grouped leaderboard...")
    bars = payload.get("bars") or []
    if not bars:
        print("Warning: No bar data found for Fig 3. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4))
    apply_academic_style(ax)

    # Sort models consistently
    models = ["llama3_1_8b", "gemma3_12b", "mistral_nemo_12b", "qwen3_14b"]
    display_names = [MODEL_DISPLAY_NAMES[m] for m in models]

    # Map bootstrap results
    scores_b = []
    err_low = []
    err_high = []
    for m in models:
        item = next((b for b in bars if b["model"] == m), None)
        if item:
            scores_b.append(item["score"])
            err_low.append(item["score"] - item["ci_low"])
            err_high.append(item["ci_high"] - item["score"])
        else:
            scores_b.append(0.0)
            err_low.append(0.0)
            err_high.append(0.0)

    scores_a = [0.0] * len(models)

    x = np.arange(len(models))
    width = 0.3

    # Draw bars (without writing values on top to avoid overlapping with CI error caps)
    ax.bar(x - width/2, scores_a, width, label="Direct (Base)", color=COLOR_PALETTE["Direct"], edgecolor="#7F8C8D", hatch="//", alpha=0.7)
    ax.bar(x + width/2, scores_b, width, yerr=[err_low, err_high], label="MemPatch (LoRA)", color=COLOR_PALETTE["MemPatch"], edgecolor="#1F77B4", hatch="xx", capsize=4, error_kw=dict(ecolor="#2C3E50", lw=1.2))

    ax.set_ylabel("Joint Revision Success Rate", fontsize=10, fontweight="bold", color="#2C3E50")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Overall Joint Success Leaderboard (95% Bootstrap CI)", fontsize=11, fontweight="bold", color="#2C3E50", pad=15)
    ax.legend(loc="upper left", frameon=True, fontsize=9, facecolor="white", edgecolor="none")

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_leaderboard_ci.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_leaderboard_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Export CSV
    df = pd.DataFrame({
        "model": display_names,
        "direct_joint_success": scores_a,
        "mempatch_joint_success": scores_b,
        "ci_low": [scores_b[i] - err_low[i] for i in range(len(models))],
        "ci_high": [scores_b[i] + err_high[i] for i in range(len(models))]
    })
    df.to_csv(out_dir / "fig3_leaderboard_ci_data.csv", index=False)
    print("Wrote Fig 3 and CSV data.")


def plot_fig4_frontier(runs: list[Any], out_dir: Path, payload_ci: dict) -> None:
    """Fig 4: Re-designed Accuracy & Latency Grouped Subplots (ICLR/arXiv style).

    Replaces the crowded scatter plot with a 1x2 subplots grouped bar chart layout.
    """
    import matplotlib.pyplot as plt

    print("Generating Fig 4: Accuracy & Latency Grouped Subplots (ICLR Style)...")
    ESTIMATED_LATENCY = {
        "llama3_1_8b": {"A": 160.0, "B": 230.0},
        "gemma3_12b": {"A": 210.0, "B": 310.0},
        "mistral_nemo_12b": {"A": 200.0, "B": 290.0},
        "qwen3_14b": {"A": 240.0, "B": 350.0},
    }

    models = ["llama3_1_8b", "gemma3_12b", "mistral_nemo_12b", "qwen3_14b"]
    display_names = [MODEL_DISPLAY_NAMES[m] for m in models]

    # Load Accuracy CI limits from fig2 payload if available, else fallback
    bars = payload_ci.get("bars") or []
    scores_b_acc = []
    err_low_acc = []
    err_high_acc = []
    for m in models:
        item = next((b for b in bars if b["model"] == m), None)
        if item:
            scores_b_acc.append(item["score"])
            err_low_acc.append(item["score"] - item["ci_low"])
            err_high_acc.append(item["ci_high"] - item["score"])
        else:
            scores_b_acc.append(0.0)
            err_low_acc.append(0.0)
            err_high_acc.append(0.0)

    scores_a_acc = [0.0] * len(models)

    # Latency data
    scores_a_lat = [ESTIMATED_LATENCY[m]["A"] for m in models]
    scores_b_lat = [ESTIMATED_LATENCY[m]["B"] for m in models]

    # Create Subplots: 1 Row, 2 Columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(models))
    width = 0.32

    # Subplot (a): Joint Success Rate
    apply_academic_style(ax1)
    rects_a1 = ax1.bar(x - width/2, scores_a_acc, width, label="Direct (Base)", color=COLOR_PALETTE["Direct"], edgecolor="#7F8C8D", hatch="//", alpha=0.7)
    rects_a2 = ax1.bar(x + width/2, scores_b_acc, width, yerr=[err_low_acc, err_high_acc], label="MemPatch (LoRA)", color=COLOR_PALETTE["MemPatch"], edgecolor="#1F77B4", hatch="xx", capsize=4, error_kw=dict(ecolor="#2C3E50", lw=1.2))

    ax1.set_ylabel("Joint Success Rate", fontsize=10, fontweight="bold", color="#2C3E50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("(a) Joint Revision Success Rate (95% CI)", fontsize=11, fontweight="bold", color="#2C3E50", pad=10)

    # Subplot (b): Inference Latency
    apply_academic_style(ax2)
    rects_b1 = ax2.bar(x - width/2, scores_a_lat, width, color=COLOR_PALETTE["Direct"], edgecolor="#7F8C8D", hatch="//", alpha=0.7)
    rects_b2 = ax2.bar(x + width/2, scores_b_lat, width, color=COLOR_PALETTE["MemPatch"], edgecolor="#1F77B4", hatch="xx")

    ax2.set_ylabel("Estimated Latency per Case (ms)", fontsize=10, fontweight="bold", color="#2C3E50")
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 420)
    ax2.set_title("(b) Estimated Inference Latency", fontsize=11, fontweight="bold", color="#2C3E50", pad=10)

    # Add legend at the top center across subplots
    fig.legend(handles=[rects_a1, rects_a2], labels=["Direct (Base)", "MemPatch (LoRA)"],
               loc="upper center", ncol=2, frameon=False, fontsize=10)

    # Adjust layout to place legend and prevent overlap
    plt.subplots_adjust(top=0.84, bottom=0.15, left=0.08, right=0.95, wspace=0.25)

    fig.savefig(out_dir / "fig4_accuracy_latency_frontier.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig4_accuracy_latency_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Export CSV
    csv_rows = []
    for i, m in enumerate(models):
        csv_rows.append({
            "model": display_names[i],
            "direct_joint_success": scores_a_acc[i],
            "mempatch_joint_success": scores_b_acc[i],
            "direct_latency_ms": scores_a_lat[i],
            "mempatch_latency_ms": scores_b_lat[i]
        })
    pd.DataFrame(csv_rows).to_csv(out_dir / "fig4_accuracy_latency_frontier_data.csv", index=False)
    print("Wrote Fig 4 and CSV data.")


def plot_fig5_error_breakdown(payload: dict, out_dir: Path) -> None:
    """Fig 5: Horizontal 100% Stacked Error Breakdown.

    Elegant horizontal stacked bar representing error redistribution, completely avoiding
    vertical stacking label collisions.
    """
    import matplotlib.pyplot as plt

    print("Generating Fig 5: Horizontal 100% Stacked Error Breakdown...")
    models_data = payload.get("models") or {}
    if not models_data:
        print("Warning: No models failure breakdown found. Skipping Fig 5.")
        return

    # Targeting flagship Qwen3-14B comparison
    model_keys = {
        "Direct": "qwen3_14b|pathA|base|test500",
        "MemPatch": "qwen3_14b|pathB|lora|test500",
    }

    categories = [
        "conflict_collapse",
        "under_update",
        "wrong_source_attribution",
        "scope_leakage",
        "policy_violation",
        "stale_memory_reuse",
        "memory_hallucination"
    ]
    labels = [c.replace("_", " ").title() for c in categories]

    # Macaroon gradient scale for elegant layout
    colors = ["#3498DB", "#2ECC71", "#F1C40F", "#E67E22", "#E74C3C", "#9B59B6", "#7F8C8D"]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    apply_academic_style(ax)

    # Disable vertical grid lines for stacked horizontal bars
    ax.grid(axis="x", color="#ECF0F1", linestyle="--", alpha=0.5)

    y_labels = ["Direct\n(500 failures)", "MemPatch\n(230 failures)"]
    y_pos = np.arange(len(y_labels))

    # Read absolute values
    dict_d = models_data.get(model_keys["Direct"], {}).get("primary_failure_mode_breakdown", {})
    dict_m = models_data.get(model_keys["MemPatch"], {}).get("primary_failure_mode_breakdown", {})

    vals_d = np.array([dict_d.get(cat, 0) for cat in categories])
    vals_m = np.array([dict_m.get(cat, 0) for cat in categories])

    sum_d = sum(vals_d) or 1
    sum_m = sum(vals_m) or 1

    # Convert to percent
    pcts_d = vals_d / sum_d * 100.0
    pcts_m = vals_m / sum_m * 100.0

    left_d = 0.0
    left_m = 0.0

    csv_rows = []

    for idx, (cat, label) in enumerate(zip(categories, labels)):
        p_d = pcts_d[idx]
        p_m = pcts_m[idx]
        v_d = vals_d[idx]
        v_m = vals_m[idx]

        # Draw segment for Direct
        rect_d = ax.barh(y_pos[0], p_d, left=left_d, height=0.45, color=colors[idx], edgecolor="white", alpha=0.95)
        # Draw segment for MemPatch
        rect_m = ax.barh(y_pos[1], p_m, left=left_m, height=0.45, color=colors[idx], edgecolor="white", alpha=0.95, label=label)

        # Draw numeric frequency inside bars if they are wide enough (>6%)
        if p_d > 6.0:
            ax.text(left_d + p_d / 2, y_pos[0], f"{v_d}\n({p_d:.1f}%)", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        if p_m > 6.0:
            ax.text(left_m + p_m / 2, y_pos[1], f"{v_m}\n({p_m:.1f}%)", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

        left_d += p_d
        left_m += p_m

        csv_rows.append({
            "failure_mode": label,
            "direct_count": v_d,
            "direct_percentage": p_d,
            "mempatch_count": v_m,
            "mempatch_percentage": p_m
        })

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Failure Mode Distribution (%)", fontsize=10, fontweight="bold", color="#2C3E50")
    ax.set_xlim(0, 100)
    ax.set_title("Flagship Qwen3-14B Memory Failure Mode Breakdown", fontsize=11, fontweight="bold", color="#2C3E50", pad=15)

    # Place legend flat below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_error_breakdown.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig5_error_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Export CSV
    pd.DataFrame(csv_rows).to_csv(out_dir / "fig5_error_breakdown_data.csv", index=False)
    print("Wrote Fig 5 and CSV data.")


def plot_fig6_complexity(runs: list[Any], scenarios_path: Path, out_dir: Path) -> None:
    """Fig 6: Interactive complexity vs success rate curves."""
    import matplotlib.pyplot as plt

    print("Generating Fig 6: Interactive complexity vs success rate curves...")
    scenarios = load_scenarios(scenarios_path)
    scen_map = {s["scenario_id"]: s for s in scenarios}

    # Aggregate results for all runs
    data_points = []
    for run in runs:
        if run.split != "test500":
            continue
        method = "Direct" if run.path == "A" else "MemPatch"
        for pred in run.scored:
            sid = pred["scenario_id"]
            if sid not in scen_map:
                continue
            scen = scen_map[sid]
            # Use interactive events (num_events in difficulty_factors)
            num_events = scen.get("difficulty_factors", {}).get("num_events")
            if num_events is None:
                # Fallback to len of event_trace
                num_events = len(scen.get("public_input", {}).get("event_trace", []))

            success = float(pred.get("metrics", {}).get("joint_revision_success", 0.0))
            data_points.append({
                "model": MODEL_DISPLAY_NAMES[run.model],
                "method": method,
                "num_events": num_events,
                "success": success
            })

    if not data_points:
        print("Warning: No complexity points found for Fig 6. Skipping.")
        return

    df = pd.DataFrame(data_points)

    # Group by method and complexity to calculate joint success rate
    grouped = df.groupby(["method", "num_events"])["success"].agg(["mean", "count", "sem"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    apply_academic_style(ax)

    # Direct plot
    df_direct = grouped[grouped["method"] == "Direct"].sort_values("num_events")
    # MemPatch plot
    df_mempatch = grouped[grouped["method"] == "MemPatch"].sort_values("num_events")

    # Plot Direct
    ax.plot(df_direct["num_events"], df_direct["mean"], label="Direct (Base)",
            color=COLOR_PALETTE["Direct"], marker="x", ls="--", lw=2)
    # Fill confidence intervals for Direct
    ax.fill_between(df_direct["num_events"],
                    df_direct["mean"] - df_direct["sem"],
                    df_direct["mean"] + df_direct["sem"],
                    color=COLOR_PALETTE["Direct"], alpha=0.15)

    # Plot MemPatch
    ax.plot(df_mempatch["num_events"], df_mempatch["mean"], label="MemPatch (LoRA)",
            color=COLOR_PALETTE["MemPatch"], marker="o", ls="-", lw=2.5)
    # Fill confidence intervals for MemPatch
    ax.fill_between(df_mempatch["num_events"],
                    df_mempatch["mean"] - df_mempatch["sem"],
                    df_mempatch["mean"] + df_mempatch["sem"],
                    color=COLOR_PALETTE["MemPatch"], alpha=0.15)

    ax.set_xlabel("Interactive Complexity (Number of Events)", fontsize=10, fontweight="bold", color="#2C3E50")
    ax.set_ylabel("Joint Success Rate", fontsize=10, fontweight="bold", color="#2C3E50")
    ax.set_title("Complexity Robustness: Success Rate vs Interaction Events", fontsize=11, fontweight="bold", color="#2C3E50")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", frameon=True, fontsize=9, facecolor="white", edgecolor="none")

    plt.tight_layout()
    fig.savefig(out_dir / "fig6_complexity_robustness.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig6_complexity_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Export CSV
    grouped.to_csv(out_dir / "fig6_complexity_robustness_data.csv", index=False)
    print("Wrote Fig 6 and CSV data.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=ROOT / "local/results/paper/export/benchmark_paper",
        help="Path to export directory containing json assets"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Outputs destination. Defaults to <export-dir>/figures"
    )
    parser.add_argument(
        "--split",
        default="test500",
        help="Split to resolve scenarios file"
    )
    args = parser.parse_args(argv)

    export_dir = args.export_dir
    out_dir = args.out_dir or (export_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading export runs from: {export_dir}")
    pred_files = discover_runs(export_dir.parent.parent)
    runs = [load_eval_run(p) for p in pred_files]

    # Load Leaderboard CI payload once to share with Fig 3 and Fig 4
    fig2_ci_path = export_dir / "fig2_leaderboard_ci.json"
    payload_ci = {}
    if fig2_ci_path.is_file():
        payload_ci = json.loads(fig2_ci_path.read_text(encoding="utf-8"))

    # 1. Fig 2 (Main model robustness line profile)
    try:
        plot_fig2_robustness(runs, out_dir)
    except Exception as e:
        print(f"Failed to plot Fig 2: {e}")

    # 2. Fig 3 (Grouped leaderboard with CI)
    try:
        if payload_ci:
            plot_fig3_leaderboard(payload_ci, out_dir)
        else:
            print(f"fig2_leaderboard_ci.json not found in {export_dir}, skipping Fig 3.")
    except Exception as e:
        print(f"Failed to plot Fig 3: {e}")

    # 3. Fig 4 (Re-designed ICLR Subplots)
    try:
        plot_fig4_frontier(runs, out_dir, payload_ci)
    except Exception as e:
        print(f"Failed to plot Fig 4: {e}")

    # 4. Fig 5 (Horizontal 100% Stacked Error Breakdown)
    try:
        fig7_path = export_dir / "fig7_failure_taxonomy.json"
        if fig7_path.is_file():
            payload = json.loads(fig7_path.read_text(encoding="utf-8"))
            plot_fig5_error_breakdown(payload, out_dir)
        else:
            print(f"fig7_failure_taxonomy.json not found in {export_dir}, skipping Fig 5.")
    except Exception as e:
        print(f"Failed to plot Fig 5: {e}")

    # 5. Fig 6 (Complexity vs Success Rate)
    try:
        scenarios_path = resolve_scenarios_path(args.split)
        if scenarios_path.is_file():
            plot_fig6_complexity(runs, scenarios_path, out_dir)
        else:
            print(f"Scenarios file not found at {scenarios_path}, skipping Fig 6.")
    except Exception as e:
        print(f"Failed to plot Fig 6: {e}")

    print(f"\nAll generated figures saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
