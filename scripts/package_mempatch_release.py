#!/usr/bin/env python3
"""Package MemPatch HF release metadata (checksums + manifest refresh) from JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCENARIO_JSONL = "scenarios.jsonl"
DEFAULT_SPLITS = ("train", "main", "hard")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def discover_splits(input_dir: Path, splits: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for split in splits:
        candidate = input_dir / split / SCENARIO_JSONL
        if candidate.is_file():
            found[split] = candidate
    return found


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build_manifest(
    *,
    base_manifest: dict[str, Any],
    split_paths: dict[str, Path],
    release_version: str,
    dataset_name: str,
) -> dict[str, Any]:
    manifest = dict(base_manifest)
    manifest["release_version"] = release_version
    manifest["dataset"] = dataset_name
    manifest["data_files"] = {split: f"{split}/{SCENARIO_JSONL}" for split in split_paths}
    manifest["public_split_name_counts"] = {
        split: read_jsonl_count(path) for split, path in split_paths.items()
    }
    manifest["public_total"] = sum(manifest["public_split_name_counts"].values())
    if "train" in manifest["public_split_name_counts"]:
        manifest["notes"] = {
            **(manifest.get("notes") or {}),
            "train": "local/SFT-only split in v1.2; not for benchmark leaderboard eval",
        }
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package MemPatch release checksums and manifest.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing {train,main,hard}/scenarios.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Release bundle directory (e.g. hf_release/mempatch_v1_2)",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated splits to include if present (default: train,main,hard)",
    )
    parser.add_argument("--release-version", default="1.2.0")
    parser.add_argument("--dataset-name", default="MemPatch")
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=None,
        help="Optional existing manifest.json to merge (decision_quotas, notes, ...)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_mempatch_bench_dataset.py --packaging-final on each split",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Run report_split_decision_distribution.py after packaging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_names = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    split_paths = discover_splits(args.input_dir, split_names)
    if not split_paths:
        print(
            f"error: no scenarios.jsonl found under {args.input_dir}/{{{','.join(split_names)}}}",
            file=sys.stderr,
        )
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checksums: dict[str, str] = {}
    for split, src in split_paths.items():
        rel = f"{split}/{SCENARIO_JSONL}"
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        checksums[rel] = sha256_file(dest)

    base_manifest_path = args.base_manifest or (out_dir / "manifest.json")
    base_manifest = load_manifest(base_manifest_path) if base_manifest_path else {}
    manifest = build_manifest(
        base_manifest=base_manifest,
        split_paths={split: out_dir / split / SCENARIO_JSONL for split in split_paths},
        release_version=args.release_version,
        dataset_name=args.dataset_name,
    )

    (out_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_info = {
        "dataset_name": args.dataset_name,
        "version": args.release_version,
        "data_files": manifest["data_files"],
        "public_split_name_counts": manifest["public_split_name_counts"],
        "total_public_examples": manifest["public_total"],
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Packaged {len(split_paths)} split(s) into {out_dir}")
    print(f"checksums: {out_dir / 'checksums.json'}")
    for rel, digest in sorted(checksums.items()):
        print(f"  {rel}: {digest[:16]}... n={manifest['public_split_name_counts'].get(rel.split('/')[0])}")

    repo_root = Path(__file__).resolve().parent.parent
    python = sys.executable
    if args.validate:
        failed = False
        for split, path in split_paths.items():
            cmd = [
                python,
                str(repo_root / "scripts" / "validate_mempatch_bench_dataset.py"),
                "--data",
                str(out_dir / split / SCENARIO_JSONL),
                "--packaging-final",
            ]
            print(f"\n$ {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                failed = True
        if failed:
            return 1

    if args.report:
        cmd = [python, str(repo_root / "scripts" / "report_split_decision_distribution.py")]
        for split in split_paths:
            cmd.extend(["--split", split, str(out_dir / split)])
        if (out_dir / "manifest.json").is_file():
            cmd.extend(["--manifest", str(out_dir / "manifest.json")])
        print(f"\n$ {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
