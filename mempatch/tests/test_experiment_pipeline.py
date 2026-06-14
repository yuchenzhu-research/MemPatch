from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.memory.context_builders import (
    BASELINE_DISPLAY_NAMES,
    PAPER_MAIN_BASELINE_IDS,
)
from scripts.select_smoke_cases import select_cases
from scripts.paper.build_experiment_artifacts import FORMAL_MODELS, SMOKE_MODELS
from scripts.linux.run_hf_test_eval import prediction_from_output


ROOT = Path(__file__).resolve().parents[2]


def _scenario(index: int, decision: str) -> dict:
    return {
        "scenario_id": f"case_{index:04d}",
        "primary_failure_mode": "under_update",
        "hidden_gold": {"expected_decision": decision},
    }


def test_smoke_selection_is_deterministic_and_stratified() -> None:
    rows = [
        _scenario(index, ("use_current_memory", "mark_unresolved", "escalate")[index % 3])
        for index in range(90)
    ]
    first = select_cases(rows, 30, 20270614)
    second = select_cases(list(reversed(rows)), 30, 20270614)
    assert [row["scenario_id"] for row in first] == [row["scenario_id"] for row in second]
    assert len({row["scenario_id"] for row in first}) == 30
    counts = {}
    for row in first:
        decision = row["hidden_gold"]["expected_decision"]
        counts[decision] = counts.get(decision, 0) + 1
    assert counts == {"escalate": 10, "mark_unresolved": 10, "use_current_memory": 10}


def test_formal_frozen_baselines_and_names() -> None:
    assert PAPER_MAIN_BASELINE_IDS == (
        "structured_direct",
        "full_context",
        "vanilla_rag",
        "time_aware_rag",
        "summary_memory",
    )
    assert BASELINE_DISPLAY_NAMES["structured_direct"] == "Frozen Direct Prompting"
    assert BASELINE_DISPLAY_NAMES["vanilla_rag"] == "Lexical RAG"
    assert [slug for slug, _ in SMOKE_MODELS] == ["qwen3_14b", "gemma3_12b", "phi4"]
    assert [slug for slug, _ in FORMAL_MODELS] == [
        "qwen3_14b",
        "gemma3_12b",
        "phi4",
        "mistral_nemo_12b",
    ]


def test_smoke_dry_run_order_excludes_mistral(tmp_path: Path) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "DRY_RUN": "1",
        "LOCAL_ROOT": str(tmp_path / "local"),
        "PYTHON": "/usr/bin/python3",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/linux/run_smoke_no_lora.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    assert "qwen3_14b" in output
    assert "gemma3_12b" in output
    assert "phi4" in output
    assert "mistral" not in output.lower()
    assert output.index("qwen3_14b") < output.index("gemma3_12b") < output.index("phi4")


def test_formal_dry_run_order_ends_with_mistral(tmp_path: Path) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "DRY_RUN": "1",
        "LOCAL_ROOT": str(tmp_path / "local"),
        "PYTHON": "/usr/bin/python3",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/linux/run_experiment.sh"), "formal"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    first_positions = [
        output.index("SLUG=qwen3_14b"),
        output.index("SLUG=gemma3_12b"),
        output.index("SLUG=phi4"),
        output.index("SLUG=mistral_nemo_12b"),
    ]
    assert first_positions == sorted(first_positions)
    model_lines = [line for line in output.splitlines() if "SLUG=" in line]
    assert "SLUG=mistral_nemo_12b" in model_lines[-1]


def test_selector_writes_exact_id_array(tmp_path: Path) -> None:
    source = tmp_path / "test500.jsonl"
    rows = [_scenario(index, "use_current_memory") for index in range(40)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ids = tmp_path / "smoke_case_ids.json"
    scenarios = tmp_path / "scenarios.jsonl"
    subprocess.run(
        [
            "/usr/bin/python3",
            str(ROOT / "scripts/select_smoke_cases.py"),
            "--input",
            str(source),
            "--ids-out",
            str(ids),
            "--scenarios-out",
            str(scenarios),
        ],
        check=True,
    )
    assert len(json.loads(ids.read_text(encoding="utf-8"))) == 30
    assert len(scenarios.read_text(encoding="utf-8").splitlines()) == 30


def test_final_state_control_does_not_silently_repair_invalid_schema() -> None:
    prediction = prediction_from_output(
        "case_1",
        '{"decision":"invalid"}',
        scenario_public_view={"public_input": {"initial_memory": [], "event_trace": []}},
        project_schema=False,
    )
    assert prediction["response"] == {"decision": "invalid"}
    assert "schema_repairs" not in prediction
