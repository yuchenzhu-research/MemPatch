#!/usr/bin/env python3
"""Generate and evaluate a scaled hard split (hard_150_en / hard_200_en scale test)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
GOLD_CHECKS = (
    "joint_revision_success",
    "decision_macro_f1",
    "memory_state_accuracy",
    "evidence_f1",
    "failure_diagnosis_accuracy",
)


def run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def patch_scale_test_metadata(split_dir: Path, count: int) -> None:
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_release_name"] = "hard_scale_test"
    manifest["role"] = (
        f"Scale-test hard split ({count} scenarios) for pattern-driven generator validation; "
        "not a final release."
    )
    manifest["intended_use"] = "Internal scale-test evaluation only; not paper-facing final release."
    manifest["generation_commit"] = "hard_scale_test"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = split_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# ReTrace-Bench `hard_{count}_en` (Scale test — NOT final release)",
                "",
                f"Public Split Name: **`hard`**",
                f"Count: {count}",
                "Source: controlled_synthetic only",
                "Difficulty: L3/L4 only",
                "",
                "This split is for internal hard-generator scale validation.",
                "Do NOT treat it as the paper-facing final release.",
                "",
                "## CONTAMINATION WARNING",
                "Do NOT use this evaluation dataset for training models.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hard split scale test pipeline.")
    parser.add_argument("--count", type=int, default=150, choices=(150, 200))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    py = root / ".venv" / "bin" / "python"
    count = args.count
    data_root = root / f"data/retrace_bench_hard{count}"
    split_dir = data_root / f"hard_{count}_en"
    data_path = split_dir / "scenarios.jsonl"
    out_dir = root / f"outputs/retrace_bench_hard{count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_prefix = ["env", "PYTHONPATH=.", f"PYTHONPYCACHEPREFIX={root / '.pycache_compile'}"]

    gen = run(
        [
            *env_prefix,
            str(py),
            "scripts/generate_retrace_bench_final.py",
            "--seed",
            str(args.seed),
            "--out",
            str(data_root),
            "--only",
            "hard",
            "--count",
            f"hard={count}",
        ],
        cwd=root,
    )
    if gen.returncode != 0:
        print(gen.stdout)
        print(gen.stderr, file=sys.stderr)
        return gen.returncode

    patch_scale_test_metadata(split_dir, count)

    val = run(
        [*env_prefix, str(py), "scripts/validate_retrace_bench_dataset.py", "--data", str(data_path)],
        cwd=root,
    )
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr, file=sys.stderr)
        return val.returncode
    val_payload = json.loads(val.stdout)
    validator_pass = not val_payload.get("errors")

    gold = run(
        [
            *env_prefix,
            str(py),
            "scripts/check_retrace_bench_gold_oracle.py",
            "--data",
            str(data_path),
            "--out",
            str(out_dir / "gold_oracle.metrics.json"),
        ],
        cwd=root,
    )
    print(gold.stdout)
    if gold.returncode != 0:
        print(gold.stderr, file=sys.stderr)
        return gold.returncode
    gold_payload = json.loads(gold.stdout)
    gold_pass = bool(gold_payload.get("pass"))
    for metric in GOLD_CHECKS:
        got = gold_payload.get("gold_oracle_metrics", {}).get(metric)
        if got != 1.0:
            gold_pass = False

    for baseline in ("latest_only", "retrieve_all"):
        pred_path = out_dir / f"{baseline}.predictions.jsonl"
        if pred_path.exists():
            pred_path.unlink()
        metrics_path = out_dir / f"{baseline}.predictions.metrics.json"
        if metrics_path.exists():
            metrics_path.unlink()
        base = run(
            [
                *env_prefix,
                str(py),
                "scripts/run_retrace_bench_baseline.py",
                "--data",
                str(data_path),
                "--baseline",
                baseline,
                "--out",
                str(pred_path),
            ],
            cwd=root,
        )
        print(base.stdout)
        if base.returncode != 0:
            print(base.stderr, file=sys.stderr)
            return base.returncode

    if not args.skip_api:
        api_dir = out_dir / "api_models"
        api_dir.mkdir(parents=True, exist_ok=True)
        import re

        slug = re.sub(r"[^A-Za-z0-9._-]+", "__", DEEPSEEK_MODEL.strip()).strip("_")
        for stale in api_dir.glob(f"{slug}.*"):
            stale.unlink()
        api = run(
            [
                *env_prefix,
                str(py),
                "scripts/run_retrace_bench_api_models.py",
                "--data",
                str(data_path),
                "--models",
                DEEPSEEK_MODEL,
                "--out-dir",
                str(api_dir),
                "--disable-thinking",
            ],
            cwd=root,
        )
        print(api.stdout)
        if api.returncode != 0:
            print(api.stderr, file=sys.stderr)
            return api.returncode

    summary = run(
        [
            *env_prefix,
            str(py),
            "scripts/build_retrace_bench_hard_scale_summary.py",
            "--data",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            str(args.seed),
        ]
        + (["--validator-pass"] if validator_pass else [])
        + (["--gold-pass"] if gold_pass else []),
        cwd=root,
    )
    print(summary.stdout)
    if summary.returncode != 0:
        print(summary.stderr, file=sys.stderr)
        return summary.returncode

    print(
        json.dumps(
            {
                "count": count,
                "data": str(data_path),
                "outputs": str(out_dir),
                "validator_pass": validator_pass,
                "gold_pass": gold_pass,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
