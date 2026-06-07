#!/usr/bin/env python3
"""Download an MLX base model from Hugging Face into local/models/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PRESETS: dict[str, dict[str, str]] = {
    "gemma3-12b": {
        "repo_id": "mlx-community/gemma-3-12b-it-4bit",
        "local_name": "gemma-3-12b-it-4bit",
    },
    "deepseek-r1-14b": {
        "repo_id": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        "local_name": "DeepSeek-R1-Distill-Qwen-14B-4bit",
    },
    "qwen3-14b": {
        "repo_id": "mlx-community/Qwen3-14B-MLX-4bit",
        "local_name": "Qwen3-14B-MLX-4bit",
    },
}


def download(repo_id: str, out_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    print(f"Downloading {repo_id} -> {out_dir}")
    print(f"  HF_ENDPOINT={endpoint or '(default huggingface.co)'}")
    print(f"  proxy={proxy or '(none)'}")
    print(f"  HF_TOKEN={'set' if token else 'unset — large models may stall without token'}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )
    print(f"Done: {out_dir}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Known MLX model preset.",
    )
    parser.add_argument(
        "--repo-id",
        help="Override Hugging Face repo id.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Override output directory under local/models/.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=root / "local/models",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preset is None and not args.repo_id:
        raise SystemExit("Provide --preset or --repo-id")

    if args.preset is not None:
        preset = PRESETS[args.preset]
        repo_id = args.repo_id or preset["repo_id"]
        out_dir = args.out_dir or (args.models_root / preset["local_name"])
    else:
        repo_id = args.repo_id
        if args.out_dir is None:
            raise SystemExit("--repo-id requires --out-dir")
        out_dir = args.out_dir

    download(repo_id, out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
