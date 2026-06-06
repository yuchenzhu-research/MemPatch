#!/usr/bin/env python3
"""Download MemPatch scenario JSONL from Hugging Face into a local directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hf_env import load_hf_token

DEFAULT_REPO = "Sylvan-Vale-Moon/MemPatch"
DEFAULT_SPLITS = ("main", "hard")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MemPatch HF dataset JSONL splits.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("local/MemPatch"),
        help="Output root (writes {split}/scenarios.jsonl)",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split names to download",
    )
    parser.add_argument("--revision", default=None, help="Optional HF dataset revision / tag")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("error: install huggingface_hub (pip install huggingface_hub)", file=sys.stderr)
        return 1

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    token = load_hf_token()

    for split in splits:
        remote_path = f"{split}/scenarios.jsonl"
        print(f"Downloading {args.repo}:{remote_path} ...", flush=True)
        try:
            cached = hf_hub_download(
                repo_id=args.repo,
                repo_type="dataset",
                filename=remote_path,
                revision=args.revision,
                token=token,
            )
        except Exception as exc:
            print(f"warning: skip {remote_path}: {exc}", file=sys.stderr)
            continue
        dest = args.out_dir / split / "scenarios.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(cached).read_bytes())
        print(f"Wrote {dest}")

    print(f"Done. {len(splits)} split(s) in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
