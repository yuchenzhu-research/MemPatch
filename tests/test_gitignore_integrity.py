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
        # Canonical ReTrace-Bench (internal "v1.1") public paper-facing splits.
        # private_hidden scenarios are intentionally NOT committed.
        "data/retrace_bench_v1_1/main_3000_en/",
        "data/retrace_bench_v1_1/hard_500_en/",
        "data/retrace_bench_v1_1/realistic_200_en/",
        "data/retrace_bench_v1_1/calibration_80_en/",
        # Legacy v1.0 + development split package (kept for provenance).
        "data_legacy/retrace_bench_v1_0/",
        "data_legacy/retrace_bench_dev_artifacts/",
        # Gold-free + gold annotation packets (gold mirrors already-public splits).
        "annotation_packets/retrace_bench_v1_1/",
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
