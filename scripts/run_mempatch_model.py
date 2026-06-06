#!/usr/bin/env python3
"""Alias entrypoint for Direct Response baseline (implementation: run_retrace_bench_model)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_retrace_bench_model import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
