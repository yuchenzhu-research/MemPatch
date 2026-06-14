from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LIB_PHASES = ROOT / "scripts/linux/lib_phases.sh"
LIB_SELECTION = ROOT / "scripts/linux/lib_selection.sh"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_phase(tmp_path: Path, function: str, run_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "{LIB_PHASES}"; {function} qwen3_14b',
        ],
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHON": sys.executable,
            "RESULTS_ROOT": str(tmp_path / "results"),
            "RUN_ID": run_id,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_phase_pick_requires_exact_run_id_segment(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "results/qwen3_14b/checkpoint_selection.json",
        {"checkpoint_dir": "/adapters/split0/full512_2048/checkpoint-512"},
    )

    assert _run_phase(tmp_path, "phase_pick_done", "full512").returncode != 0
    assert _run_phase(tmp_path, "phase_pick_done", "full512_2048").returncode == 0


def test_selection_helper_requires_exact_run_id_segment(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {"checkpoint_dir": "/adapters/split0/full512_2048/checkpoint-512"},
    )
    command = f'source "{LIB_SELECTION}"; selection_matches_run_id "{selection}" "$RUN_ID"'

    wrong = subprocess.run(
        ["bash", "-c", command],
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable, "RUN_ID": "full512"},
        check=False,
    )
    exact = subprocess.run(
        ["bash", "-c", command],
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable, "RUN_ID": "full512_2048"},
        check=False,
    )

    assert wrong.returncode != 0
    assert exact.returncode == 0


def test_phase_eval_requires_exact_run_id_segment(tmp_path: Path) -> None:
    result_dir = tmp_path / "results/qwen3_14b"
    for name in (
        "test500_base_predictions.jsonl",
        "test500_lora_best_predictions.jsonl",
        "test500_path_a_lora_best_predictions.jsonl",
    ):
        path = result_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    adapter = "/adapters/split0/full512_2048/checkpoint-512"
    _write_json(
        result_dir / "test500_lora_best_manifest.json",
        {"run_meta": {"adapter_path": adapter, "schema_projection": "public_only_v1"}},
    )
    _write_json(
        result_dir / "test500_path_a_lora_best_manifest.json",
        {"run_meta": {"adapter_path": adapter, "method_path": "path_a_typed_actions_dpa"}},
    )
    _write_json(
        result_dir / "test500_path_a_lora_best_no_dpa_manifest.json",
        {"run_meta": {"adapter_path": adapter, "method_path": "path_a_typed_actions_no_dpa"}},
    )

    assert _run_phase(tmp_path, "phase_eval_done", "full512").returncode != 0
    assert _run_phase(tmp_path, "phase_eval_done", "full512_2048").returncode == 0
