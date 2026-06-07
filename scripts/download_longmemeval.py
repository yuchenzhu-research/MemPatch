#!/usr/bin/env python3
"""Download LongMemEval-cleaned JSON files into local/longmemeval/data/.

Uses huggingface_hub (respects HTTP(S)_PROXY, HF_ENDPOINT, HF_TOKEN).
Official repo: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Profiles:
  smoke — oracle + s (~290 MB); good for wiring baselines
  full  — smoke files + m (~3 GB total)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "xiaowu0162/longmemeval-cleaned"

FILES_BY_PROFILE: dict[str, list[str]] = {
    "smoke": [
        "longmemeval_oracle.json",
        "longmemeval_s_cleaned.json",
    ],
    "full": [
        "longmemeval_oracle.json",
        "longmemeval_s_cleaned.json",
        "longmemeval_m_cleaned.json",
    ],
}


def _default_proxy() -> str | None:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def _print_env_hint() -> None:
    proxy = _default_proxy()
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print("download env:")
    print(f"  HF_ENDPOINT={endpoint or '(default huggingface.co)'}")
    print(f"  proxy={proxy or '(none — set HTTP(S)_PROXY if needed)'}")
    print(f"  HF_TOKEN={'set' if token else 'unset (public dataset OK without token)'}")


def download_files(out_dir: Path, profile: str, force: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required (.venv/bin/pip install huggingface_hub)") from exc

    files = FILES_BY_PROFILE[profile]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in files:
        dest = out_dir / name
        if dest.exists() and not force:
            print(f"skip existing {dest} ({dest.stat().st_size:,} bytes)")
            continue
        print(f"downloading {name} -> {dest}")
        cached = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=name,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        path = Path(cached)
        print(f"done {path.name} ({path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("local/longmemeval/data"),
        help="Output directory (default: local/longmemeval/data)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(FILES_BY_PROFILE),
        default="smoke",
        help="smoke=oracle+s; full=+m (~2.7 GB)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )
    args = parser.parse_args()
    _print_env_hint()
    download_files(args.out_dir.resolve(), args.profile, args.force)
    print(f"LongMemEval ({args.profile}) ready under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
