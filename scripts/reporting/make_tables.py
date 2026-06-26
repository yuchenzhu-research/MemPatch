#!/usr/bin/env python3
"""Render final MemPatch-Bench LaTeX tables from aggregate CSVs only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mempatch.benchmark.method_names import FINAL_METHODS  # noqa: E402
from mempatch.benchmark.reporting_taxonomy import MEMORY_CAPABILITIES  # noqa: E402


TABLES = (
    "table_main_results.tex",
    "table_challenge_results.tex",
    "table_ablation_mempatch.tex",
    "table_capability_breakdown.tex",
    "table_cost_latency.tex",
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


def pct(row: dict[str, str], key: str) -> str:
    value = numeric(row, key)
    return "--" if value is None else f"{100 * value:.1f}"


def num(row: dict[str, str], key: str) -> str:
    value = numeric(row, key)
    return "--" if value is None else f"{value:.1f}"


def method_sort(row: dict[str, str]) -> tuple[str, int]:
    method = row.get("method", "")
    return (row.get("model", ""), FINAL_METHODS.index(method) if method in FINAL_METHODS else 999)


def latex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def pending_table(title: str) -> str:
    return "\n".join(
        [
            "% Auto-generated pending table.",
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{latex_escape(title)} pending}}",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "Status & Reason \\\\",
            "\\midrule",
            "pending & Aggregate CSV is absent or empty \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def results_table(rows: list[dict[str, str]], title: str) -> str:
    if not rows:
        return pending_table(title)
    lines = [
        "% Auto-generated from aggregate CSVs.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(title)}}}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Model & Method & $n$ & Schema & State & Evidence & Joint \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=method_sort):
        lines.append(
            f"{latex_escape(row.get('model', ''))} & {latex_escape(row.get('method', ''))} & "
            f"{row.get('n', '0')} & {pct(row, 'schema_valid_rate')} & "
            f"{pct(row, 'exact_state_map')} & {pct(row, 'evidence_f1')} & "
            f"{pct(row, 'strict_joint')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def ablation_table(rows: list[dict[str, str]]) -> str:
    selected = [row for row in rows if row.get("method") in {"mempatch_noguard", "mempatch"}]
    return results_table(selected, "MemPatch guard ablation")


def cost_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return pending_table("Cost and latency")
    lines = [
        "% Auto-generated from cost_latency.csv.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cost and latency}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Model & Method & Input tok. & Output tok. & Total tok. & Latency s & Unsup. \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=method_sort):
        lines.append(
            f"{latex_escape(row.get('model', ''))} & {latex_escape(row.get('method', ''))} & "
            f"{num(row, 'input_tokens')} & {num(row, 'output_tokens')} & "
            f"{num(row, 'total_tokens')} & {num(row, 'latency_sec')} & "
            f"{pct(row, 'unsupported_or_hallucinated_evidence_rate')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def capability_sort(row: dict[str, str]) -> tuple[str, str, int, str]:
    capability = row.get("capability", "")
    try:
        capability_rank = MEMORY_CAPABILITIES.index(capability)
    except ValueError:
        capability_rank = 999
    method = row.get("method", "")
    method_rank = FINAL_METHODS.index(method) if method in FINAL_METHODS else 999
    return (row.get("split", ""), row.get("model", ""), capability_rank, f"{method_rank:03d}:{method}")


def capability_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return pending_table("Memory capability breakdown")
    lines = [
        "% Auto-generated from per_capability.csv.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Memory capability breakdown}",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Model & Method & Capability & $n$ & State & Evidence & Joint \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=capability_sort):
        lines.append(
            f"{latex_escape(row.get('model', ''))} & {latex_escape(row.get('method', ''))} & "
            f"{latex_escape(row.get('capability', ''))} & {row.get('n', '0')} & "
            f"{pct(row, 'exact_state_map')} & {pct(row, 'evidence_f1')} & "
            f"{pct(row, 'strict_joint')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_tables(aggregate_dir: Path, output_dir: Path, *, strict: bool) -> list[str]:
    main = read_csv(aggregate_dir / "main_results.csv")
    challenge = read_csv(aggregate_dir / "challenge_results.csv")
    per_model = read_csv(aggregate_dir / "per_model_method_split.csv")
    capability = read_csv(aggregate_dir / "per_capability.csv")
    cost = read_csv(aggregate_dir / "cost_latency.csv")
    missing = []
    if strict:
        for name, rows in {
            "main_results.csv": main,
            "challenge_results.csv": challenge,
            "per_model_method_split.csv": per_model,
            "per_capability.csv": capability,
            "cost_latency.csv": cost,
        }.items():
            if not rows:
                missing.append(name)
        if missing:
            raise RuntimeError("missing required aggregate CSVs: " + ", ".join(missing))
    write(output_dir / "table_main_results.tex", results_table(main, "Main synthetic results"))
    write(output_dir / "table_challenge_results.tex", results_table(challenge, "Challenge synthetic results"))
    write(output_dir / "table_ablation_mempatch.tex", ablation_table(per_model))
    write(output_dir / "table_capability_breakdown.tex", capability_table(capability))
    write(output_dir / "table_cost_latency.tex", cost_table(cost))
    return [str(output_dir / name) for name in TABLES]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=Path("results/v1.4/final_synthetic/aggregates"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v1.4/final_synthetic/tables"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = build_tables(args.aggregate_dir, args.output_dir, strict=args.strict)
    except Exception as exc:
        print(f"table export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
