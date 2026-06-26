#!/usr/bin/env python3
"""Safe HuggingFace dataset upload preflight for MemPatch-Bench.

Default behavior is validation plus dry-run listing. The script refuses risky
folders, never prints tokens, and requires explicit upload flags before it will
call the Hub API.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_PATTERNS = (
    ".env",
    ".env.*",
    "*.env",
    ".tokens",
    ".tokens/*",
    "*.token",
    "*_token",
    "*_token.txt",
    "hf_token*",
    ".hf_token",
    "*_api_key*",
    "*.apikey",
    "*.pem",
    "*.key",
    "secrets.*",
    "secret.*",
    "*.secret",
    ".git",
    ".git/*",
    "__pycache__",
    "__pycache__/*",
    ".DS_Store",
    "*.safetensors",
    "*.gguf",
    "*.ckpt",
    "*.pt",
    "*.pth",
)

FORBIDDEN_PARTS = {
    "raw_internal",
    "results",
    "runs",
    "predictions",
    "scores",
    "checkpoints",
    "model_cache",
    "cache",
    ".git",
}


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GiB"


def iter_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file())


def rel(path: Path, folder: Path) -> str:
    return path.relative_to(folder).as_posix()


def matches_forbidden(relative_path: str) -> str | None:
    parts = set(Path(relative_path).parts)
    if parts & FORBIDDEN_PARTS:
        return f"path contains forbidden component: {sorted(parts & FORBIDDEN_PARTS)[0]}"
    name = Path(relative_path).name
    for pattern in FORBIDDEN_PATTERNS:
        if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(name, pattern):
            return f"matches forbidden pattern: {pattern}"
    return None


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def find_dataset_card(folder: Path) -> Path | None:
    for candidate in ("README.md", "HF_DATASET_CARD.md"):
        path = folder / candidate
        if path.exists() and path.is_file():
            return path
    return None


def find_license(folder: Path) -> Path | None:
    for candidate in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        path = folder / candidate
        if path.exists() and path.is_file():
            return path
    return None


def has_checksums(folder: Path) -> bool:
    for name in ("checksums.sha256", "SHA256SUMS", "SHA256SUMS.txt"):
        if (folder / name).exists():
            return True
    for path in folder.rglob("*.json"):
        payload = load_json(path)
        if payload and isinstance(payload.get("checksums"), dict) and payload["checksums"]:
            return True
    return False


def audit_payload_is_clean(payload: dict[str, Any]) -> bool | None:
    violation_lists = {
        key: value
        for key, value in payload.items()
        if "violation" in key and isinstance(value, list)
    }
    if violation_lists:
        return all(len(value) == 0 for value in violation_lists.values())
    if "violations" in payload:
        violations = payload.get("violations")
        return isinstance(violations, list) and len(violations) == 0
    if "public_forbidden_field_violations" in payload:
        violations = payload.get("public_forbidden_field_violations")
        return isinstance(violations, list) and len(violations) == 0
    summary = payload.get("audit_summary")
    if isinstance(summary, dict) and "public_forbidden_field_violation_count" in summary:
        return int(summary.get("public_forbidden_field_violation_count") or 0) == 0
    return None


def find_clean_audit(folder: Path) -> Path | None:
    for path in folder.rglob("*.json"):
        lowered = path.name.lower()
        if "audit" not in lowered:
            continue
        payload = load_json(path)
        if not payload:
            continue
        clean = audit_payload_is_clean(payload)
        if clean is True:
            return path
        if clean is False:
            raise RuntimeError(f"public leakage audit has failures: {path.relative_to(folder)}")
    return None


def validate_folder(folder: Path, *, include_labels: bool = False) -> tuple[list[Path], int]:
    if not folder.exists() or not folder.is_dir():
        raise RuntimeError(f"folder does not exist or is not a directory: {folder}")
    if folder == folder.parent or folder == REPO_ROOT or folder in REPO_ROOT.parents:
        raise RuntimeError("refusing broad upload folder: choose a prepared release subdirectory")

    files = iter_files(folder)
    if not files:
        raise RuntimeError("upload folder is empty")

    for path in files:
        relative = rel(path, folder)
        if path.is_symlink():
            raise RuntimeError(f"symlink refused in upload folder: {relative}")
        reason = matches_forbidden(relative)
        if reason:
            raise RuntimeError(f"forbidden file refused: {relative} ({reason})")
        if not include_labels and ("labels" in Path(relative).parts or relative.endswith(".labels.jsonl")):
            raise RuntimeError(
                f"private label file refused for public upload: {relative} "
                "(rerun with --include-labels only for an explicitly private/gated release)"
            )

    if find_dataset_card(folder) is None:
        raise RuntimeError("missing dataset card: expected README.md or HF_DATASET_CARD.md in upload folder")
    if find_license(folder) is None:
        raise RuntimeError("missing license file in upload folder")
    if not has_checksums(folder):
        raise RuntimeError("missing checksums: expected manifest JSON with checksums or checksums.sha256")
    if find_clean_audit(folder) is None:
        raise RuntimeError("missing clean public leakage audit report")

    total_size = sum(path.stat().st_size for path in files)
    return files, total_size


def token_from_dotenv(dotenv_path: Path) -> str | None:
    if not dotenv_path.exists():
        return None
    first_line = dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    if not first_line:
        return None
    line = first_line[0].strip()
    if not line or line.startswith("#"):
        return None
    if "=" in line:
        key, value = line.split("=", 1)
        if key.strip() != "HF_TOKEN":
            return None
        return value.strip().strip("'\"") or None
    return line


def resolve_token(*, use_dotenv: bool) -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    if use_dotenv:
        return token_from_dotenv(REPO_ROOT / ".env")
    return None


def print_dry_run(folder: Path, files: list[Path], total_size: int) -> None:
    print(f"Dry run passed for folder: {folder}")
    print(f"Files: {len(files)}")
    print(f"Total size: {human_size(total_size)}")
    for path in files:
        print(f"- {rel(path, folder)} ({human_size(path.stat().st_size)})")


def upload_folder_to_hf(args: argparse.Namespace, token: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("missing dependency: pip install huggingface_hub") from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.folder),
        revision=args.revision,
        commit_message=args.commit_message,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Target HF dataset repo id, e.g. org/mempatch-bench")
    parser.add_argument("--folder", required=True, type=Path, help="Prepared upload folder")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print file list without uploading")
    parser.add_argument("--use-dotenv", action="store_true", help="Read only the first .env line as HF_TOKEN if HF_TOKEN env is absent")
    parser.add_argument("--include-labels", action="store_true", help="Allow labels/ *.labels.jsonl in upload folder for an explicitly private/gated release")
    parser.add_argument("--allow-labels", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--private", action="store_true", help="Create/update the HF dataset repo as private")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Upload MemPatch-Bench release")
    parser.add_argument("--execute", action="store_true", help="Actually upload after all validations")
    parser.add_argument("--execute-upload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--yes-i-understand-upload", action="store_true", help="Required with --execute")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    folder = args.folder.resolve()

    try:
        include_labels = bool(args.include_labels or args.allow_labels)
        execute = bool(args.execute or args.execute_upload)
        files, total_size = validate_folder(folder, include_labels=include_labels)
        if args.dry_run or not execute:
            print_dry_run(folder, files, total_size)
            if not args.dry_run:
                print("No upload executed. Pass --execute and --yes-i-understand-upload only after an approved dry run.")
            return 0

        if not args.yes_i_understand_upload:
            raise RuntimeError("--execute requires --yes-i-understand-upload")
        token = resolve_token(use_dotenv=args.use_dotenv)
        if not token:
            raise RuntimeError("missing HF_TOKEN environment variable")
        upload_folder_to_hf(args, token)
        print(f"Upload completed for dataset repo: {args.repo_id}")
        return 0
    except Exception as exc:
        execute = bool(getattr(args, "execute", False) or getattr(args, "execute_upload", False))
        if execute:
            print(f"HF upload failed: {type(exc).__name__}", file=sys.stderr)
        else:
            print(f"HF push preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
