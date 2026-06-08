#!/usr/bin/env python3
"""Render MemPatch benchmark-paper figures from export_benchmark_paper_assets.py output.

Requires: pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FAMILY_FALLBACK = {
    "qwen": "#E87722",
    "gemma": "#2A9D8F",
    "mistral": "#FF7000",
    "llama": "#4267B2",
    "other": "#6C757D",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=root / "local/results/paper/export/benchmark_paper",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <export-dir>/figures",
    )
    return parser.parse_args(argv)


def _load(export_dir: Path, name: str) -> dict:
    return json.loads((export_dir / name).read_text(encoding="utf-8"))


def plot_fig2(ax, payload: dict, model_cards: dict) -> None:
    bars = payload.get("bars") or []
    if not bars:
        return
    labels = [b["model"] for b in bars]
    scores = [b["score"] for b in bars]
    err_lo = [b["score"] - b["ci_low"] for b in bars]
    err_hi = [b["ci_high"] - b["score"] for b in bars]
    colors = [
        (model_cards.get(b["model"]) or {}).get("family_color")
        or FAMILY_FALLBACK.get((model_cards.get(b["model"]) or {}).get("family", "other"), "#6C757D")
        for b in bars
    ]
    y = range(len(labels))
    ax.barh(list(y), scores, xerr=[err_lo, err_hi], color=colors, capsize=3, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Joint revision success")
    ax.set_title("Overall leaderboard (95% bootstrap CI)")
    ax.invert_yaxis()


def plot_fig3_capability(ax, payload: dict, model_cards: dict) -> None:
    import numpy as np

    models = payload.get("models") or []
    cols = payload.get("capability_columns") or []
    matrix = payload.get("capability_matrix") or {}
    if not models or not cols:
        return
    data = np.array([[matrix.get(m, {}).get(c) or 0.0 for c in cols] for m in models])
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    ax.set_title("Model x capability (headline metrics)")
    return im


def plot_fig4(ax, payload: dict) -> None:
    groups = payload.get("groups") or []
    if not groups:
        return
    models = [g["model"] for g in groups]
    x = range(len(models))
    width = 0.35
    l3 = [g.get("L3", {}).get("joint_revision_success", 0.0) for g in groups]
    l4 = [g.get("L4", {}).get("joint_revision_success", 0.0) for g in groups]
    ax.bar([i - width / 2 for i in x], l3, width, label="L3")
    ax.bar([i + width / 2 for i in x], l4, width, label="L4")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Joint revision success")
    ax.set_title("Difficulty breakdown")
    ax.legend()
    ax.set_ylim(0, 1.0)


def plot_fig5(ax, payload: dict, model_cards: dict) -> None:
    points = payload.get("points") or []
    for pt in points:
        x = pt.get("active_params_b")
        y = pt.get("joint_revision_success")
        if x is None or y is None:
            continue
        color = pt.get("family_color") or FAMILY_FALLBACK.get(pt.get("family", "other"), "#6C757D")
        ax.scatter([x], [y], s=120, color=color, label=pt.get("display_name", pt["model"]))
        ax.annotate(pt.get("display_name", pt["model"]), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Active parameters (B)")
    ax.set_ylabel("Joint revision success")
    ax.set_title("Performance vs scale")
    ax.set_ylim(0, 1.0)


def plot_fig6(ax, payload: dict) -> None:
    import numpy as np

    models = payload.get("models") or []
    win = payload.get("win_rate") or {}
    if len(models) < 2:
        return
    data = np.array([[win.get(a, {}).get(b) if win.get(a, {}).get(b) is not None else 0.5 for b in models] for a in models])
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Pairwise win-rate (joint, ties excluded)")
    return im


def plot_fig7(ax, payload: dict, model_key: str) -> None:
    block = (payload.get("models") or {}).get(model_key) or {}
    fr = block.get("fractions") or {}
    if not fr:
        return
    labels = list(fr.keys())
    vals = [fr[k] for k in labels]
    ax.barh(labels, vals, color="#457B9D")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Fraction among failed cases")
    ax.set_title(f"Failure taxonomy — {model_key}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_dir = args.export_dir
    out_dir = args.out_dir or (export_dir / "figures")
    if not export_dir.is_dir():
        print(f"missing export dir: {export_dir}", file=sys.stderr)
        return 1
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib to render figures", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    cards_path = root / "config/paper_model_cards.json"
    model_cards = json.loads(cards_path.read_text(encoding="utf-8")) if cards_path.is_file() else {}

    out_dir.mkdir(parents=True, exist_ok=True)

    fig2_path = export_dir / "fig2_leaderboard_ci.json"
    if fig2_path.is_file():
        fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(_load(export_dir, "fig2_leaderboard_ci.json").get("bars", [])) + 1)))
        plot_fig2(ax, _load(export_dir, "fig2_leaderboard_ci.json"), model_cards)
        fig.tight_layout()
        out = out_dir / "fig2_leaderboard_ci.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")

    fig3_path = export_dir / "fig3_capability_heatmap.json"
    if fig3_path.is_file():
        fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(_load(export_dir, "fig3_capability_heatmap.json").get("models", [])) + 1)))
        im = plot_fig3_capability(ax, _load(export_dir, "fig3_capability_heatmap.json"), model_cards)
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        out = out_dir / "fig3_capability_heatmap.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")

    for name, plot_fn in (
        ("fig4_difficulty_breakdown.json", lambda ax, p: plot_fig4(ax, p)),
        ("fig5_cost_performance.json", lambda ax, p: plot_fig5(ax, p, model_cards)),
    ):
        path = export_dir / name
        if path.is_file():
            fig, ax = plt.subplots(figsize=(7, 4))
            plot_fn(ax, _load(export_dir, name))
            fig.tight_layout()
            out = out_dir / name.replace(".json", ".png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Wrote {out}")

    fig6_path = export_dir / "fig6_pairwise_winrate.json"
    if fig6_path.is_file() and len(_load(export_dir, "fig6_pairwise_winrate.json").get("models", [])) >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = plot_fig6(ax, _load(export_dir, "fig6_pairwise_winrate.json"))
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        out = out_dir / "fig6_pairwise_winrate.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")

    fig7 = _load(export_dir, "fig7_failure_taxonomy.json") if (export_dir / "fig7_failure_taxonomy.json").is_file() else {}
    for key in list((fig7.get("models") or {}).keys())[:3]:
        fig, ax = plt.subplots(figsize=(6, 3))
        plot_fig7(ax, fig7, key)
        fig.tight_layout()
        safe = key.replace("|", "_").replace("/", "_")
        out = out_dir / f"fig7_failure_{safe}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
