#!/usr/bin/env python3
"""Discover the best LoRA checkpoint for a slug when RUN_ID is unknown or pick was never run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)


def load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def best_from_metrics(metrics_path: Path) -> dict | None:
    payload = load_metrics(metrics_path)
    if not payload:
        return None
    checkpoints = payload.get("checkpoints") or []
    if not checkpoints:
        return None
    best = min(checkpoints, key=lambda row: float(row["val_loss"]))
    ckpt_dir = Path(best["checkpoint_dir"])
    if not ckpt_dir.is_dir():
        return None
    run_dir = metrics_path.parent
    split_dir = run_dir.parent
    adapter_dir = payload.get("output_dir") or str(ckpt_dir.parent)
    return {
        "best_step": best.get("step"),
        "best_val_loss": best.get("val_loss"),
        "checkpoint_dir": str(ckpt_dir),
        "adapter_dir": str(adapter_dir),
        "log_dir": str(run_dir),
        "split_partition": split_dir.name,
        "run_id": run_dir.name,
        "metrics_path": str(metrics_path),
        "selection_rule": "lowest_val_loss_among_discovered_runs",
    }


def discover_metrics_files(log_root: Path, slug: str) -> list[Path]:
    hits: list[Path] = []
    for pattern in (f"{slug}_split*/**/trainer_metrics.json",):
        for path in sorted(log_root.glob(pattern)):
            if path not in hits:
                hits.append(path)
    return hits


def discover_adapter_metrics(adapter_root: Path, slug: str) -> list[Path]:
    """Fallback: trainer_metrics.json may sit under adapter output dir on older runs."""
    hits: list[Path] = []
    for path in sorted(adapter_root.glob(f"{slug}_multitask_lora/**/trainer_metrics.json")):
        hits.append(path)
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefer-run-id", default=None)
    args = parser.parse_args(argv)

    metrics_files = discover_metrics_files(args.log_root, args.slug)
    metrics_files.extend(
        p for p in discover_adapter_metrics(args.adapter_root, args.slug) if p not in metrics_files
    )
    if not metrics_files:
        print(
            f"error: no trainer_metrics.json found for {args.slug} under {args.log_root}",
            file=sys.stderr,
        )
        return 1

    candidates: list[dict] = []
    for metrics_path in metrics_files:
        row = best_from_metrics(metrics_path)
        if row is not None:
            candidates.append(row)

    if not candidates:
        print(f"error: trainer_metrics.json found but no valid checkpoints for {args.slug}", file=sys.stderr)
        return 1

    if args.prefer_run_id:
        preferred = [c for c in candidates if c.get("run_id") == args.prefer_run_id]
        if not preferred:
            print(
                f"error: no checkpoint found for exact run_id={args.prefer_run_id}",
                file=sys.stderr,
            )
            return 1
        candidates = preferred

    best = min(candidates, key=lambda row: float(row["best_val_loss"]))
    best["slug"] = args.slug
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(best["checkpoint_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
