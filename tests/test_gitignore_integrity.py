from __future__ import annotations

import subprocess
import os

def test_gitignore_sensitive_artifacts_not_committed():
    # Run git ls-files to get all currently tracked files in the repo
    res = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True
    )
    files = res.stdout.splitlines()

    # NOTE: outputs/ is intentionally TRACKED (benchmark prediction dumps and
    # metrics are committed for reproducibility/sharing; see AGENTS.md and the
    # !outputs/** allowlist in .gitignore), so it is NOT in this prefix list.
    ignored_prefixes = (
        ".external_repos/",
        ".reference_cache/",
        "artifacts/",
        "checkpoints/",
        ".pytest_cache/",
        ".venv/",
        "venv/",
        "__pycache__/",
        "models/",
        "adapters/",
    )

    allowed_jsonl_prefixes = (
        "tests/",
        "data/fixtures/",
        "experiments/multiagent/local_training/data/",
        "data/retrace_learn/v1/boundary_audit/",
        # ReTrace-Bench v1.0 paper-facing splits.
        "data/retrace_bench/main_3000_en/",
        "data/retrace_bench/hard_300_en/",
        "data/retrace_bench/realistic_100_en/",
        "data/retrace_bench/calibration_80_en/",
        # ReTrace-Learn supervision pools (moved out of the benchmark tree).
        "data/retrace_learn/supervision_train_3000_en/",
        "data/retrace_learn/supervision_dev_400_en/",
        "release/huggingface/ReTrace-Bench/",
        "examples/retrace_bench/",
        "outputs/",
    )

    for f in files:
        # Check against sensitive directory prefixes
        assert not f.startswith(ignored_prefixes), (
            f"Git-ignored sensitive file '{f}' is tracked in git! "
            "Please remove it from tracking."
        )

        # Check for unwhitelisted JSONL files
        if f.endswith(".jsonl"):
            assert any(f.startswith(p) for p in allowed_jsonl_prefixes), (
                f"JSONL file '{f}' is tracked in git but is not in a whitelisted directory! "
                "Please remove it from tracking or update the allowed paths in test_gitignore_integrity.py."
            )
