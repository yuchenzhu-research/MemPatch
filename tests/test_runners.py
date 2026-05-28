import json
import subprocess
import sys
from pathlib import Path

from retracemem.evaluation import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_boundary_audit_retrieval_baseline_writes_jsonl(tmp_path) -> None:
    cases_path = tmp_path / "boundary.jsonl"
    output_path = tmp_path / "boundary_out.jsonl"
    cases = [
        {
            "case_id": "ba_001",
            "old_belief": "The user commutes by bicycle.",
            "new_evidence": "The user now drives to work.",
            "query": "How does the user commute?",
            "expected_relation": "SUPERSEDE",
            "expected_authorized": True,
            "protected_beliefs": ["The user likes Thai food."],
        }
    ]
    cases_path.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")

    result = _run_script(
        "scripts/run_boundary_audit.py",
        "--cases",
        str(cases_path),
        "--method",
        "retrieval_baseline",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert "cases_total: 1" in result.stdout
    records = read_jsonl(output_path)
    assert len(records) == 1
    assert records[0]["query_id"] == "ba_001"
    assert records[0]["method"] == "retrieval_baseline"


def test_boundary_audit_missing_cases_exits_zero(tmp_path) -> None:
    output_path = tmp_path / "empty.jsonl"

    result = _run_script(
        "scripts/run_boundary_audit.py",
        "--cases",
        str(tmp_path / "missing.jsonl"),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert "No BoundaryAudit cases found" in result.stdout
    assert read_jsonl(output_path) == []


def test_stale_runner_writes_records_from_minimal_main_file(tmp_path) -> None:
    reference_root = tmp_path / "STALE"
    main_path = reference_root / "demo_MAIN.json"
    output_path = tmp_path / "stale.jsonl"
    main_path.parent.mkdir(parents=True)
    main_path.write_text(
        json.dumps(
            [
                {
                    "uid": "sample-1",
                    "M_old": "The user commutes by bicycle.",
                    "M_new": "The user drives now.",
                    "haystack_session": [["The user commutes by bicycle."]],
                    "probing_queries": {
                        "dim1_query": "Does the user commute by bicycle?",
                        "dim2_query": "Does the user drive?",
                        "dim3_query": "",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    result = _run_script(
        "scripts/run_stale.py",
        "--reference-root",
        str(reference_root),
        "--limit",
        "1",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert "samples_loaded: 1" in result.stdout
    records = read_jsonl(output_path)
    assert [record["query_id"] for record in records] == ["sample-1:dim1_query", "sample-1:dim2_query"]
    assert all(record["method"] == "retrieval_baseline" for record in records)


def test_stale_runner_without_data_exits_zero(tmp_path) -> None:
    output_path = tmp_path / "stale_empty.jsonl"

    result = _run_script(
        "scripts/run_stale.py",
        "--reference-root",
        str(tmp_path / "missing"),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert "No STALE MAIN files found" in result.stdout
    assert read_jsonl(output_path) == []


def test_memora_runner_writes_records_from_minimal_root(tmp_path) -> None:
    persona_root = tmp_path / "Memora" / "data" / "weekly" / "engineer"
    conversations = persona_root / "conversations"
    conversations.mkdir(parents=True)
    (conversations / "session_0001.json").write_text(
        json.dumps(
            {
                "session_id": 1,
                "date": "2025-06-01",
                "persona": "engineer",
                "conversation": [{"speaker": "user", "message": "I use a red notebook for tasks."}],
            }
        ),
        encoding="utf-8",
    )
    (persona_root / "evaluation_questions_engineer.json").write_text(
        json.dumps(
            {
                "persona": "engineer",
                "questions": {
                    "remembering": [
                        {
                            "question_id": "q1",
                            "question": "What notebook does the user use?",
                            "evaluation": {},
                        }
                    ],
                    "reasoning": [],
                    "recommending": [],
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "memora.jsonl"

    result = _run_script(
        "scripts/run_memora.py",
        "--reference-root",
        str(tmp_path / "Memora"),
        "--limit",
        "1",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert "records_written: 1" in result.stdout
    records = read_jsonl(output_path)
    assert len(records) == 1
    assert records[0]["query_id"] == "q1"
    assert records[0]["method"] == "retrieval_baseline"


def test_memora_runner_without_data_exits_zero(tmp_path) -> None:
    output_path = tmp_path / "memora_empty.jsonl"

    result = _run_script(
        "scripts/run_memora.py",
        "--reference-root",
        str(tmp_path / "missing"),
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert "No Memora data roots found" in result.stdout
    assert read_jsonl(output_path) == []


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def test_run_retrace_internal_dev_script() -> None:
    result = _run_script("scripts/run_retrace_internal_dev.py")
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "All Dev Cases Passed Cleanly!" in result.stdout
    assert "Typed Fixtures" in result.stdout

