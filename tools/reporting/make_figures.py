#!/usr/bin/env python3
"""Render final MemPatch-Bench PDF figures from aggregate CSVs only."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mempatch.benchmark.method_names import FINAL_METHODS, FINAL_MODELS  # noqa: E402


FIGURES = (
    "hazard_heatmap.pdf",
    "method_comparison_bar.pdf",
    "model_scale_curve.pdf",
    "cost_latency_scatter.pdf",
    "evidence_f1_vs_state_success.pdf",
    "unsafe_reuse_by_failure_mode.pdf",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def method_order(method: str) -> int:
    return FINAL_METHODS.index(method) if method in FINAL_METHODS else 999


def model_order(model: str) -> tuple[int, str]:
    if model in FINAL_MODELS:
        return FINAL_MODELS.index(model), model
    return 999, model


def model_size_b(model: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)", model)
    return float(match.group(1)) if match else 0.0


def pending_pdf(path: Path, title: str, reason: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, f"pending: {reason}", ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bar(path: Path, rows: list[dict[str, str]], title: str, metric: str) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        pending_pdf(path, title, "aggregate CSV is absent or empty")
        return
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = numeric(row, metric)
        if value is not None:
            by_method[row.get("method", "")].append(value)
    methods = sorted(by_method, key=method_order)
    values = [sum(by_method[method]) / len(by_method[method]) for method in methods]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(methods)), values, color="#4C78A8")
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def hazard_heatmap(path: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        pending_pdf(path, "Hazard heatmap", "per_failure_mode.csv is absent or empty")
        return
    failure_modes = sorted({row.get("failure_mode", "") for row in rows if row.get("failure_mode")})
    methods = sorted({row.get("method", "") for row in rows if row.get("method")}, key=method_order)
    matrix = []
    for failure_mode in failure_modes:
        line = []
        for method in methods:
            vals = [
                numeric(row, "unsafe_reuse_rate")
                for row in rows
                if row.get("failure_mode") == failure_mode and row.get("method") == method
            ]
            vals = [value for value in vals if value is not None]
            line.append(sum(vals) / len(vals) if vals else 0.0)
        matrix.append(line)
    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 0.8), max(4, len(failure_modes) * 0.35)))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
    ax.set_yticks(range(len(failure_modes)), failure_modes)
    ax.set_title("Unsafe reuse by failure mode")
    fig.colorbar(image, ax=ax, label="unsafe_reuse_rate")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def model_scale_curve(path: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        pending_pdf(path, "Model scale curve", "main_results.csv is absent or empty")
        return
    methods = sorted({row.get("method", "") for row in rows if row.get("method")}, key=method_order)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        points = []
        for row in rows:
            if row.get("method") != method:
                continue
            y = numeric(row, "strict_joint")
            if y is not None:
                points.append((model_size_b(row.get("model", "")), y))
        if points:
            points.sort()
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=method)
    ax.set_xlabel("model size (B parameters, parsed from name)")
    ax.set_ylabel("strict_joint")
    ax.set_ylim(0, 1)
    ax.set_title("Model scale curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def scatter(path: Path, rows: list[dict[str, str]], x_metric: str, y_metric: str, title: str) -> None:
    import matplotlib.pyplot as plt

    points = [(numeric(row, x_metric), numeric(row, y_metric), row.get("method", "")) for row in rows]
    points = [(x, y, method) for x, y, method in points if x is not None and y is not None]
    if not points:
        pending_pdf(path, title, f"{x_metric}/{y_metric} data is absent")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in sorted({method for _, _, method in points}, key=method_order):
        xs = [x for x, _, m in points if m == method]
        ys = [y for _, y, m in points if m == method]
        ax.scatter(xs, ys, label=method, s=34)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def unsafe_by_failure_mode(path: Path, rows: list[dict[str, str]]) -> None:
    save_bar(path, rows, "Unsafe reuse by failure mode", "unsafe_reuse_rate")


def build_figures(aggregate_dir: Path, output_dir: Path, *, strict: bool) -> list[str]:
    main = read_csv(aggregate_dir / "main_results.csv")
    per_model = read_csv(aggregate_dir / "per_model_method_split.csv")
    failure = read_csv(aggregate_dir / "per_failure_mode.csv")
    cost = read_csv(aggregate_dir / "cost_latency.csv")
    if strict:
        missing = [
            name
            for name, rows in {
                "main_results.csv": main,
                "per_model_method_split.csv": per_model,
                "per_failure_mode.csv": failure,
                "cost_latency.csv": cost,
            }.items()
            if not rows
        ]
        if missing:
            raise RuntimeError("missing required aggregate CSVs: " + ", ".join(missing))

    hazard_heatmap(output_dir / "hazard_heatmap.pdf", failure)
    save_bar(output_dir / "method_comparison_bar.pdf", main, "Method comparison", "strict_joint")
    model_scale_curve(output_dir / "model_scale_curve.pdf", main)
    scatter(output_dir / "cost_latency_scatter.pdf", cost, "latency_sec", "total_tokens", "Cost/latency scatter")
    scatter(output_dir / "evidence_f1_vs_state_success.pdf", per_model, "evidence_f1", "exact_state_map", "Evidence F1 vs state success")
    unsafe_by_failure_mode(output_dir / "unsafe_reuse_by_failure_mode.pdf", failure)
    return [str(output_dir / name) for name in FIGURES]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=Path("runs/v1.4/final_synthetic/aggregates"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v1.4/final_synthetic/figures"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = build_figures(args.aggregate_dir, args.output_dir, strict=args.strict)
    except Exception as exc:
        print(f"figure export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
