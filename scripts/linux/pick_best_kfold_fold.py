#!/usr/bin/env python3
"""Pick k-fold with lowest validation loss (reads trainer_metrics.json per fold)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)


def read_best_val(metrics_path: Path) -> float | None:
    if not metrics_path.is_file():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints") or []
    if not checkpoints:
        return payload.get("best_val_loss")
    return min(float(row["val_loss"]) for row in checkpoints)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--run-id", default="full384")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    best_fold: int | None = None
    best_loss: float | None = None
    rows: list[dict] = []

    for fold in range(args.k_folds):
        metrics_path = args.log_root / f"{args.slug}_fold{fold}" / "trainer_metrics.json"
        val = read_best_val(metrics_path)
        rows.append({"fold": fold, "best_val_loss": val, "metrics_path": str(metrics_path)})
        if val is None:
            continue
        if best_loss is None or val < best_loss:
            best_loss = val
            best_fold = fold

    if best_fold is None:
        print("error: no trainer_metrics.json with val loss found", file=sys.stderr)
        return 1

    payload = {
        "slug": args.slug,
        "run_id": args.run_id,
        "best_fold": best_fold,
        "best_val_loss": best_loss,
        "folds": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(best_fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
