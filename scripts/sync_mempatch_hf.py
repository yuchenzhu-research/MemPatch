#!/usr/bin/env python3
"""Download MemPatch scenarios from HF and regenerate local SFT train/valid/hard_probe."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hf_env import load_hf_token

DEFAULT_REPO = "Sylvan-Vale-Moon/MemPatch"
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync MemPatch HF dataset into local training data.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=Path("local/MemPatch"),
        help="Where to write {split}/scenarios.jsonl",
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=Path("local/train_data/mempatch_qwen14b_v2_1024"),
        help="Output directory for train.jsonl / valid.jsonl / hard_probe.jsonl",
    )
    parser.add_argument("--revision", default=None, help="Optional HF dataset revision / tag")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--valid-size", type=int, default=128)
    parser.add_argument("--hard-probe-size", type=int, default=50)
    parser.add_argument(
        "--target-style",
        choices=("default", "evidence_compact", "decision_balanced"),
        default="evidence_compact",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--splits",
        default="train,main,hard",
        help="Splits to download if present on HF",
    )
    return parser.parse_args(argv)


def run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_hf_token(required=True)

    download_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "download_mempatch_dataset.py"),
        "--repo",
        args.repo,
        "--out-dir",
        str(args.scenarios_dir),
        "--splits",
        args.splits,
    ]
    if args.revision:
        download_cmd.extend(["--revision", args.revision])
    if run(download_cmd) != 0:
        return 1

    train_split = args.scenarios_dir / "train" / "scenarios.jsonl"
    main_split = args.scenarios_dir / "main" / "scenarios.jsonl"
    hard_split = args.scenarios_dir / "hard" / "scenarios.jsonl"

    if not hard_split.is_file():
        print(f"error: hard split missing after download: {hard_split}", file=sys.stderr)
        return 1

    prepare_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_mempatch_sft.py"),
        "--hard",
        str(hard_split),
        "--out-dir",
        str(args.sft_dir),
        "--train-size",
        str(args.train_size),
        "--valid-size",
        str(args.valid_size),
        "--hard-probe-size",
        str(args.hard_probe_size),
        "--target-style",
        args.target_style,
        "--seed",
        str(args.seed),
    ]

    if train_split.is_file():
        prepare_cmd.extend(["--train", str(train_split)])
        print("Using v1.2 train split for SFT train/valid.")
    elif main_split.is_file():
        prepare_cmd.extend(["--main", str(main_split)])
        print("No train split on HF; using main split for SFT train/valid (v1.1).")
    else:
        print(
            f"error: neither train nor main split found under {args.scenarios_dir}",
            file=sys.stderr,
        )
        return 1

    if run(prepare_cmd) != 0:
        return 1

    print(f"Synced scenarios -> {args.scenarios_dir}")
    print(f"Regenerated SFT -> {args.sft_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
