#!/usr/bin/env python3
"""Upload MemPatch release bundle to Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.hf_env import load_hf_token

DEFAULT_REPO = "Sylvan-Vale-Moon/MemPatch"
SCENARIO_JSONL = "scenarios.jsonl"
DEFAULT_SPLITS = ("train", "main", "hard")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload MemPatch HF dataset release.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    parser.add_argument(
        "--release-dir",
        type=Path,
        required=True,
        help="Release directory (e.g. hf_release/mempatch_v1_2)",
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="Optional subfolder in the dataset repo (empty = repo root)",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated splits to require before upload",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Upload even if some splits are missing",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Upload manifest/README/checksums only (no scenarios.jsonl required)",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="HF commit message (default: derived from manifest release_version)",
    )
    return parser.parse_args(argv)


def discover_splits(release_dir: Path, splits: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for split in splits:
        candidate = release_dir / split / SCENARIO_JSONL
        if candidate.is_file():
            found[split] = candidate
    return found


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    release_dir = args.release_dir.resolve()
    if not release_dir.is_dir():
        print(f"error: release directory not found: {release_dir}", file=sys.stderr)
        return 1

    split_names = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    found = discover_splits(release_dir, split_names)
    if not found and not args.metadata_only:
        print(
            f"error: no {SCENARIO_JSONL} under {release_dir}/{{{','.join(split_names)}}}",
            file=sys.stderr,
        )
        return 1
    if (
        found
        and not args.allow_partial
        and not args.metadata_only
        and len(found) != len(split_names)
    ):
        missing = [s for s in split_names if s not in found]
        print(
            f"error: incomplete release; missing splits: {', '.join(missing)}",
            file=sys.stderr,
        )
        print("note: use --allow-partial for metadata-only preview uploads.", file=sys.stderr)
        return 1

    token = load_hf_token(required=True)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: install huggingface_hub (pip install huggingface_hub)", file=sys.stderr)
        return 1

    manifest_path = release_dir / "manifest.json"
    version = "unknown"
    if manifest_path.is_file():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = str(manifest.get("release_version", version))

    commit_message = args.commit_message or f"MemPatch release v{version}"
    api = HfApi(token=token)

    print(f"Uploading {release_dir} -> {args.repo}:{args.path_in_repo or '/'}")
    if found:
        print(f"splits present: {', '.join(sorted(found))}")
    elif args.metadata_only:
        print("metadata-only upload (no scenarios.jsonl)")

    if args.metadata_only:
        allow = {
            "manifest.json",
            "README.md",
            "dataset_info.json",
            "checksums.json",
            "VERSION",
            "DATASET_LICENSE.md",
            "LICENSE",
        }
        uploaded = 0
        for path in sorted(release_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(release_dir).as_posix()
            if rel.endswith(SCENARIO_JSONL):
                continue
            if path.name not in allow and rel not in allow:
                continue
            remote = f"{args.path_in_repo}/{rel}" if args.path_in_repo else rel
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=remote,
                repo_id=args.repo,
                repo_type="dataset",
                commit_message=commit_message if uploaded == 0 else None,
            )
            print(f"  uploaded {rel}")
            uploaded += 1
        if uploaded == 0:
            print("error: no metadata files found to upload", file=sys.stderr)
            return 1
    else:
        api.upload_folder(
            folder_path=str(release_dir),
            repo_id=args.repo,
            repo_type="dataset",
            path_in_repo=args.path_in_repo or None,
            commit_message=commit_message,
        )
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
