from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.linux.smoke_support import BASELINE_IDS, command_verify_eval, command_verify_resume


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_resume_accepts_two_step_checkpoint_chain(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapters"
    log_dir = tmp_path / "logs"
    for step in (1, 2):
        checkpoint = output_dir / f"checkpoint-{step}"
        _write_json(checkpoint / "trainer_state.json", {"global_step": step})
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    _write_json(
        log_dir / "trainer_metrics.json",
        {
            "training_config": {
                "resume_from_checkpoint": str(output_dir / "checkpoint-1"),
                "resume_global_step": 1,
                "final_global_step": 2,
            },
            "package_versions": {"trl": "test"},
        },
    )
    report = tmp_path / "resume.json"
    result = command_verify_resume(
        argparse.Namespace(output_dir=output_dir, log_dir=log_dir, out=report)
    )
    assert result == 0
    assert json.loads(report.read_text())["ok"] is True


def test_verify_eval_requires_all_eight_proxies_and_lora(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    prefix = "smoke1"
    tags = [f"{prefix}_lora_best", *(f"{prefix}_baseline_{name}" for name in BASELINE_IDS)]
    for tag in tags:
        _write_json(result_dir / f"{tag}_manifest.json", {"headline_metrics": {}})
        (result_dir / f"{tag}_predictions.jsonl").write_text(
            '{"scenario_id":"case-1","response":{}}\n', encoding="utf-8"
        )
    no_dpa_tag = f"{prefix}_lora_best_no_dpa"
    _write_json(result_dir / f"{no_dpa_tag}_manifest.json", {"headline_metrics": {}})
    (result_dir / f"{no_dpa_tag}_predictions.jsonl").write_text(
        '{"scenario_id":"case-1","response":{}}\n', encoding="utf-8"
    )
    (result_dir / f"{prefix}_8plus1.done").write_text("done\n", encoding="utf-8")
    report = tmp_path / "eval.json"
    result = command_verify_eval(
        argparse.Namespace(result_dir=result_dir, prefix=prefix, out=report)
    )
    assert result == 0
    assert len(json.loads(report.read_text())["runs"]) == 9
