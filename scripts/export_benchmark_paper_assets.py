#!/usr/bin/env python3
"""Export MemPatch benchmark-paper figures (JSON) and tables (CSV).

Maps MemPatch metrics to the benchmark-paper layout:
  Fig 1  pipeline (mermaid)
  Fig 2  leaderboard + 95% bootstrap CI
  Fig 3  model x capability / domain / decision heatmaps
  Fig 4  L3 vs L4 difficulty breakdown
  Fig 5  score vs active params scatter
  Fig 6  pairwise win-rate matrix
  Fig 7  failure taxonomy
  Table 1 dataset statistics
  Table 2 model / inference setup
  Table 3 main results (+ category rows in JSON)
  Table 4 evaluator reliability (deterministic scorer)

Example::

    PYTHONPATH=.:src .venv/bin/python scripts/export_benchmark_paper_assets.py \\
        --results-dir local/results/paper \\
        --primary-split test500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_paper_export import export_all  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "local/results/paper",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <results-dir>/export/benchmark_paper",
    )
    parser.add_argument(
        "--model-cards",
        type=Path,
        default=root / "config/paper_model_cards.json",
    )
    parser.add_argument(
        "--primary-split",
        default=None,
        help="Filter headline figures to this split tag (e.g. test500, l3).",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir or (args.results_dir / "export" / "benchmark_paper")
    manifest = export_all(
        args.results_dir,
        out_dir,
        model_cards_path=args.model_cards,
        primary_split=args.primary_split,
        n_boot=args.bootstrap,
    )
    print(f"Exported {manifest['runs']} runs -> {out_dir}")
    for name in manifest["outputs"]:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
