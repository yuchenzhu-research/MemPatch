"""Test the official evaluator CLI on the shipped sample predictions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "scripts" / "evaluate_retrace_bench_predictions.py"
DATA = REPO_ROOT / "data" / "retrace_bench_v1_1" / "calibration_80_en" / "scenarios.jsonl"
PREDICTIONS = REPO_ROOT / "examples" / "retrace_bench" / "sample_predictions.jsonl"


def test_cli_scores_sample_predictions(tmp_path):
    out_metrics = tmp_path / "metrics.json"
    out_scored = tmp_path / "scored.jsonl"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--data",
            str(DATA),
            "--predictions",
            str(PREDICTIONS),
            "--out-metrics",
            str(out_metrics),
            "--out-scored",
            str(out_scored),
            "--print-table",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_metrics.exists()
    assert out_scored.exists()

    payload = json.loads(out_metrics.read_text(encoding="utf-8"))
    assert payload["count"] == 80
    assert "decision_macro_f1" in payload["headline_metrics"]
    assert payload["errors"] == []
    assert "decision_macro_f1" in proc.stdout
