"""Stage RL-3 eval: DPA-in-the-loop reward over rollouts.

Reports mean total reward, mean of each reward component, and the failure-category
distribution. Operates on the rollouts produced by ``export_rl_rollouts`` (either
in memory or from a JSONL file).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from retrace_learn.data.export_rl_rollouts import build_rows
from retrace_learn.data.jsonl_io import read_jsonl
from retrace_learn.eval.metrics import mean


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    component_keys = list(rows[0]["reward_breakdown"].keys())
    components = {
        k: mean([r["reward_breakdown"][k] for r in rows]) for k in component_keys
    }
    return {
        "n": len(rows),
        "mean_total_reward": mean([r["total_reward"] for r in rows]),
        "mean_components": components,
        "failure_distribution": dict(Counter(r["failure_category"] for r in rows)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", default=None, help="JSONL path (default: build in memory)")
    args = parser.parse_args(argv)
    rows = list(read_jsonl(Path(args.rollouts))) if args.rollouts else build_rows()
    print(json.dumps(evaluate(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
