#!/usr/bin/env python3
"""Public entrypoint for exporting the Stage C typed-revision SFT dataset.

Thin CLI over the dataset builder in
`retracemem.evaluation.multiagent.data.silver_compositional`, which renders the
Stage C "silver v1" supervised examples (typed revision proposals grounded in
method-visible candidate structure) and runs a strict leakage/sanity checker
before writing train/valid/test JSONL splits.

    python3 scripts/export_stagec_data.py

Outputs are written under the builder's configured output directory
(`outputs/local_training/...`), which is git-ignored. Stage C training itself
lives in `experiments/multiagent/local_training/` and is out of the evaluation
path.

Note: per AGENTS.md, only human-approved reviewed examples may be promoted for
live smoke or training; this exporter renders development candidates for offline
dataset preparation and does not promote anything to "approved".
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retracemem.evaluation.multiagent.data.silver_compositional import main


if __name__ == "__main__":
    main()
