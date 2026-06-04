"""Tests that non-oracle baselines do not read hidden gold, and that oracle
baselines are clearly marked as oracle in outputs."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import scripts.run_retrace_bench_ablation as ablation
import scripts.run_retrace_bench_baseline as runner

REPO = Path(__file__).resolve().parents[2]
# Smoke/quickstart split is the cheapest valid canonical split to drive baseline tests.
BENCH = REPO / "data" / "retrace_bench_v1_1"
DATA = BENCH / "calibration_80_en" / "scenarios.jsonl"
GENERAL_SPLITS = (
    BENCH / "main_3000_en" / "scenarios.jsonl",
    BENCH / "hard_500_en" / "scenarios.jsonl",
    BENCH / "realistic_200_en" / "scenarios.jsonl",
    BENCH / "calibration_80_en" / "scenarios.jsonl",
)

# Hidden-gold fields a deployable baseline must never read as a prediction.
GOLD_TOKENS = (
    "primary_failure_mode",
    "expected_failure_diagnosis",
    "expected_answer",
    "expected_evidence_event_ids",
    "expected_memory_state",
)


def test_baseline_no_primary_failure_mode_leak():
    for name, fn in runner.BASELINES.items():
        if runner.is_oracle_baseline(name):
            continue
        src = inspect.getsource(fn)
        assert "primary_failure_mode" not in src, f"{name} leaks primary_failure_mode"
        assert "hidden_gold" not in src, f"{name} reads hidden_gold"


def test_non_oracle_baselines_do_not_touch_gold():
    for name, fn in runner.BASELINES.items():
        if runner.is_oracle_baseline(name):
            continue
        src = inspect.getsource(fn)
        for token in GOLD_TOKENS:
            assert token not in src, f"{name} references gold field {token}"


def test_oracle_baseline_may_read_gold():
    # Sanity: the oracle is the one place hidden gold use is expected.
    src = inspect.getsource(runner.retrace_oracle_engine)
    assert "hidden_gold" in src
    assert runner.is_oracle_baseline("retrace_oracle_engine") is True


def test_oracle_marked_as_oracle(tmp_path):
    out = tmp_path / "retrace_oracle_engine.jsonl"
    rc = runner.main(
        [
            "--data",
            str(DATA),
            "--baseline",
            "retrace_oracle_engine",
            "--max-cases",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["is_oracle"] is True
    assert metrics["group"] == "oracle"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and all(row["is_oracle"] is True for row in rows)


def test_non_oracle_marked_not_oracle(tmp_path):
    out = tmp_path / "crud_memory.jsonl"
    rc = runner.main(
        [
            "--data",
            str(DATA),
            "--baseline",
            "crud_memory",
            "--max-cases",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    metrics = json.loads(out.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["is_oracle"] is False
    assert metrics["group"] == "memory_baseline"


def test_llm_json_answerer_accepts_fenced_json_response():
    scenario = {
        "scenario_id": "s1",
        "workflow_context": "Use the latest verified operational note.",
        "public_input": {
            "initial_memory": [{"memory_id": "m1", "text": "old note"}],
            "event_trace": [{"event_id": "e1", "text": "verified note", "related_memory_ids": ["m1"]}],
        },
        "tasks": [{"prompt": "What should be used?"}],
    }

    class FencedProvider:
        def generate(self, _prompt: str, **_kwargs: object) -> str:
            return """```json
{
  "answer": "verified note",
  "decision": "use_current_memory",
  "memory_state": {"m1": "current"},
  "evidence_event_ids": ["e1"],
  "failure_diagnosis": "under_update"
}
```"""

    response = runner.llm_json_answerer(scenario, FencedProvider())

    assert response["decision"] == "use_current_memory"
    assert response["memory_state"] == {"m1": "current"}
    assert response["evidence_event_ids"] == ["e1"]


def test_resume_skips_existing_predictions(tmp_path):
    out = tmp_path / "crud_memory.jsonl"
    first_rc = runner.main(
        [
            "--data",
            str(DATA),
            "--baseline",
            "crud_memory",
            "--max-cases",
            "1",
            "--append",
            "--out",
            str(out),
        ]
    )
    assert first_rc == 0
    first_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(first_rows) == 1

    second_rc = runner.main(
        [
            "--data",
            str(DATA),
            "--baseline",
            "crud_memory",
            "--max-cases",
            "2",
            "--resume",
            "--out",
            str(out),
        ]
    )
    assert second_rc == 0
    resumed_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    second_id = runner.read_jsonl(DATA)[1]["scenario_id"]
    assert [row["scenario_id"] for row in resumed_rows] == [
        first_rows[0]["scenario_id"],
        second_id,
    ]
    assert resumed_rows[0] == first_rows[0]


@pytest.mark.parametrize("data_path", GENERAL_SPLITS, ids=lambda p: p.parent.name)
def test_resume_skips_existing_predictions_on_all_general_splits(tmp_path, data_path):
    expected_ids = [row["scenario_id"] for row in runner.read_jsonl(data_path)[:2]]
    assert len(expected_ids) == 2

    out = tmp_path / f"{data_path.parent.name}.jsonl"
    first_rc = runner.main(
        [
            "--data",
            str(data_path),
            "--baseline",
            "crud_memory",
            "--max-cases",
            "1",
            "--append",
            "--out",
            str(out),
        ]
    )
    assert first_rc == 0
    first_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["scenario_id"] for row in first_rows] == [expected_ids[0]]

    second_rc = runner.main(
        [
            "--data",
            str(data_path),
            "--baseline",
            "crud_memory",
            "--max-cases",
            "2",
            "--resume",
            "--out",
            str(out),
        ]
    )
    assert second_rc == 0
    resumed_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["scenario_id"] for row in resumed_rows] == expected_ids
    assert resumed_rows[0] == first_rows[0]


def test_ablation_resume_passes_through_to_baseline_runner(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run_cmd(cmd: list[str]) -> None:
        commands.append(cmd)

    monkeypatch.setattr(ablation, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(ablation, "load_metrics", lambda _path: {})

    rc = ablation.main(
        [
            "--data",
            str(DATA),
            "--out-dir",
            str(tmp_path),
            "--max-cases",
            "2",
            "--include-llm",
            "--resume",
        ]
    )

    assert rc == 0
    assert commands
    assert all("--resume" in cmd for cmd in commands)
    llm_commands = [cmd for cmd in commands if "llm_json_answerer" in cmd]
    assert llm_commands and "--append" in llm_commands[0]
