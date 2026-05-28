import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_stale_root(tmp_path: Path) -> Path:
    root = tmp_path / "STALE"
    root.mkdir()
    records = []
    for idx in range(1, 4):
        records.append(
            {
                "uid": f"sample-{idx}",
                "M_old": f"The user previously preferred tea {idx}.",
                "M_new": f"The user now prefers coffee {idx}.",
                "haystack_session": [
                    [f"Earlier, the user preferred tea {idx}."],
                    [f"Later, the user now prefers coffee {idx}."],
                ],
                "probing_queries": {
                    "dim1_query": "What does the user prefer now?",
                    "dim2_query": "What did the user prefer previously?",
                    "dim3_query": "What should the assistant adapt to now?",
                },
            }
        )
    (root / "dev_MAIN.json").write_text(json.dumps(records), encoding="utf-8")
    return root


def test_stale_development_runner_replay_progress_and_accounting(tmp_path: Path) -> None:
    stale_root = _make_stale_root(tmp_path)
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stale_development_eval.py",
            "--mode",
            "replay",
            "--reference-root",
            str(stale_root),
            "--limit-scenarios",
            "3",
            "--progress-mode",
            "line",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "[PLAN] STALE development-only run" in combined
    assert "[Stage A] scenarios 3/3" in combined
    assert "[Stage B] scenarios 3/3" in combined
    assert "Bearer " not in combined
    assert "GEMINI_API_KEY" not in combined
    assert "The user now prefers" not in combined

    report = json.loads((output_dir / "stale_development_report.json").read_text())
    manifest = json.loads((output_dir / "stale_development_manifest.json").read_text())
    assert report["scenario_count"] == 3
    assert report["query_count_per_stage"] == 9
    assert report["provider"] == "gemini"
    assert report["model"] == "gemini-3.5-flash"
    assert report["aggregate"]["A"]["queries"] == 9
    assert report["aggregate"]["B"]["queries"] == 9
    assert manifest["config"]["provider_name"] == "gemini"
    assert manifest["config"]["model_id"] == "gemini-3.5-flash"
    assert "GEMINI_API_KEY" not in json.dumps(report)
    assert "GEMINI_API_KEY" not in json.dumps(manifest)

    stage_a = report["cost"]["stage_a"]
    stage_b = report["cost"]["stage_b"]
    assert stage_a["semantic_invocations"] == 9
    assert stage_b["semantic_invocations"] == 9
    assert report["cap_usage"]["outbound_network_calls"] == 0
    assert report["cap_usage"]["tokens_from_outbound_calls"] == 0


def test_stale_development_runner_refuses_live_overwrite(tmp_path: Path) -> None:
    stale_root = _make_stale_root(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "stale_development_report.json").write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stale_development_eval.py",
            "--mode",
            "live-dev",
            "--live-approved",
            "--reference-root",
            str(stale_root),
            "--limit-scenarios",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "GEMINI_API_KEY": "test-secret-should-not-appear"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "Refusing to overwrite existing live output directory" in combined
    assert "test-secret-should-not-appear" not in combined
