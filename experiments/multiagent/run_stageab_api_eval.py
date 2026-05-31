#!/usr/bin/env python3
"""Backward-compatible thin entrypoint for the Stage A/B evaluation runner.

The evaluation engine now lives in the reusable package
``retracemem.evaluation.multiagent`` (config / cases / pipeline / directjudge /
metrics / artifacts / runner). The canonical public command is:

    python3 scripts/evaluate.py stage-a --mock
    python3 scripts/evaluate.py stage-b --mock

This shim preserves the historical invocation path
``experiments/multiagent/run_stageab_api_eval.py`` so existing run commands keep
working; it simply delegates to ``retracemem.evaluation.multiagent.runner.main``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from retracemem.evaluation.multiagent.runner import main

if __name__ == "__main__":
    main()
