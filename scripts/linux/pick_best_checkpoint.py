#!/usr/bin/env python3
"""Pick the checkpoint with lowest validation loss on the fixed split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    metrics_path = args.log_dir / "trainer_metrics.json"
    if not metrics_path.is_file():
        print(f"error: missing {metrics_path}", file=sys.stderr)
        return 1

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints") or []
    if not checkpoints:
        print("error: trainer_metrics.json has no checkpoints", file=sys.stderr)
        return 1

    best = min(checkpoints, key=lambda row: float(row["val_loss"]))
    ckpt_dir = Path(best["checkpoint_dir"])
    if not ckpt_dir.is_dir():
        print(f"error: checkpoint dir missing: {ckpt_dir}", file=sys.stderr)
        return 1

    out_payload = {
        "best_step": best.get("step"),
        "best_val_loss": best.get("val_loss"),
        "checkpoint_dir": str(ckpt_dir),
        "adapter_dir": str(args.adapter_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(ckpt_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
