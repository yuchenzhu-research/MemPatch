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

    ignored_prefixes = (
        "reference/",
        "outputs/",
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
