#!/usr/bin/env python3
"""Alias entrypoint for MemPatch-Bench evaluator (implementation: evaluate_retrace_bench_predictions)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_retrace_bench_predictions import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
