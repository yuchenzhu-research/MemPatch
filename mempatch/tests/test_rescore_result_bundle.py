from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/linux/rescore_result_bundle.py"
SCENARIOS = ROOT / "mempatch/tests/fixtures/smoke_scenarios.jsonl"
PREDICTIONS = ROOT / "mempatch/tests/fixtures/smoke_predictions.jsonl"


def test_rescore_writes_separate_artifacts_and_preserves_predictions(tmp_path: Path) -> None:
    source = tmp_path / "source/model"
    source.mkdir(parents=True)
    prediction_path = source / "test_predictions.jsonl"
    original = PREDICTIONS.read_text(encoding="utf-8")
    prediction_path.write_text(original, encoding="utf-8")
    output = tmp_path / "rescored"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data",
            str(SCENARIOS),
            "--source-results",
            str(tmp_path / "source"),
            "--out-results",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert prediction_path.read_text(encoding="utf-8") == original
    metrics = json.loads((output / "model/test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["rescore_only"] is True
    assert (output / "model/test_scored.jsonl").is_file()
