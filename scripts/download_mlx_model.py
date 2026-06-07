#!/usr/bin/env python3
"""Download an MLX base model from Hugging Face into local/models/."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import quote

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

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_WORKERS = 1
DEFAULT_RETRIES = 10
PROXY_ENV_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")

KNOWN_REPO_FILES: dict[str, tuple[str, ...]] = {
    "mlx-community/gemma-3-12b-it-4bit": (
        ".gitattributes",
        "README.md",
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ),
    "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit": (
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "mlx-community/Qwen3-14B-MLX-4bit": (
        "config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ),
}


def load_env_file(path: Path) -> bool:
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def active_proxy() -> str:
    return next((os.environ[key] for key in PROXY_ENV_KEYS if os.environ.get(key)), "")


def configure_environment(args: argparse.Namespace) -> None:
    if args.env_file is not None:
        loaded = load_env_file(args.env_file)
        print(f"env_file={args.env_file if loaded else '(not loaded)'}")

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")

    timeout = args.timeout
    if timeout is None:
        timeout = int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(timeout)

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


def hub_settings(args: argparse.Namespace) -> tuple[str, str | None, int, int]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    timeout = int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = int(os.environ.get("HF_HUB_DOWNLOAD_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))
    return token or "", endpoint or None, timeout, max_workers


def print_hub_settings(token: str, endpoint: str | None, timeout: int, max_workers: int) -> None:
    print(f"  HF_ENDPOINT={endpoint or '(default huggingface.co)'}")
    print(f"  proxy={active_proxy() or '(none)'}")
    print(f"  HF_TOKEN={'set' if token else 'unset - set HF_TOKEN for higher rate limits'}")
    print(f"  HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '0')}")
    print(f"  timeout={timeout}s")
    print(f"  max_workers={max_workers}")


def normalized_endpoint(endpoint: str | None) -> str:
    return (endpoint or "https://huggingface.co").rstrip("/")


def is_official_endpoint(endpoint: str | None) -> bool:
    return normalized_endpoint(endpoint) in {
        "https://huggingface.co",
        "https://www.huggingface.co",
    }


def selected_transport(args: argparse.Namespace, endpoint: str | None) -> str:
    if args.transport != "auto":
        return args.transport
    return "hub" if is_official_endpoint(endpoint) else "curl"


def repo_api_url(endpoint: str | None, repo_id: str, revision: str) -> str:
    base = normalized_endpoint(endpoint)
    repo_path = quote(repo_id, safe="/")
    if revision and revision != "main":
        return f"{base}/api/models/{repo_path}/revision/{quote(revision, safe='')}"
    return f"{base}/api/models/{repo_path}"


def resolve_file_url(endpoint: str | None, repo_id: str, revision: str, filename: str) -> str:
    base = normalized_endpoint(endpoint)
    repo_path = quote(repo_id, safe="/")
    return f"{base}/{repo_path}/resolve/{quote(revision, safe='')}/{quote(filename, safe='/')}"


def list_repo_files_via_api(
    repo_id: str,
    endpoint: str | None,
    revision: str,
    token: str,
    timeout: int,
) -> list[str]:
    url = repo_api_url(endpoint, repo_id, revision)
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        str(timeout),
        "--user-agent",
        "MemPatch/download_mlx_model.py",
    ]
    if token:
        cmd.extend(["--header", f"Authorization: Bearer {token}"])
    cmd.append(url)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Repo API request failed with exit code {result.returncode}: {url}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Repo API did not return JSON: {url}") from exc
    files = [item["rfilename"] for item in payload.get("siblings", []) if item.get("rfilename")]
    if not files:
        raise SystemExit(f"No files found from repo API: {url}")
    return files


def download_text_via_curl(url: str, token: str, timeout: int, retries: int) -> str:
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        str(retries),
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        str(timeout),
    ]
    if token:
        cmd.extend(["--header", f"Authorization: Bearer {token}"])
    cmd.append(url)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"curl failed with exit code {result.returncode}: {url}")
    return result.stdout


def indexed_weight_shards_via_curl(
    repo_id: str,
    endpoint: str | None,
    revision: str,
    token: str,
    timeout: int,
    retries: int,
) -> list[str]:
    url = resolve_file_url(endpoint, repo_id, revision, "model.safetensors.index.json")
    try:
        payload = json.loads(download_text_via_curl(url, token, timeout, retries))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Weight index did not return JSON: {url}") from exc
    shards = sorted(set(str(name) for name in (payload.get("weight_map") or {}).values()))
    return shards


def url_is_reachable(url: str, token: str, timeout: int) -> bool:
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--head",
        "--silent",
        "--show-error",
        "--connect-timeout",
        str(timeout),
        "--max-time",
        str(max(timeout, 30)),
    ]
    if token:
        cmd.extend(["--header", f"Authorization: Bearer {token}"])
    cmd.append(url)
    return subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def curl_repo_files(
    repo_id: str,
    endpoint: str | None,
    revision: str,
    token: str,
    timeout: int,
    retries: int,
) -> list[str]:
    try:
        files = list_repo_files_via_api(repo_id, endpoint, revision, token, timeout)
    except SystemExit as exc:
        known_files = KNOWN_REPO_FILES.get(repo_id)
        if not known_files:
            raise
        print(f"  warning: repo API failed ({exc}); using known file list for {repo_id}")
        files = list(known_files)
    if "model.safetensors.index.json" not in files:
        return files

    indexed_shards = indexed_weight_shards_via_curl(
        repo_id,
        endpoint,
        revision,
        token,
        timeout,
        retries,
    )
    if not indexed_shards:
        return files

    api_weight_files = sorted(f for f in files if f.endswith(".safetensors"))
    if api_weight_files != indexed_shards:
        print(f"  api_weight_files={api_weight_files}")
        print(f"  indexed_shards={indexed_shards}")
        reachable_indexed = [
            shard
            for shard in indexed_shards
            if url_is_reachable(
                resolve_file_url(endpoint, repo_id, revision, shard),
                token,
                timeout,
            )
        ]
        if reachable_indexed and len(reachable_indexed) == len(indexed_shards):
            non_weight_files = [f for f in files if not f.endswith(".safetensors")]
            return non_weight_files + indexed_shards
        if api_weight_files:
            print(
                "  warning: indexed shard URLs are missing or unreachable; using API "
                "safetensors files and rebuilding local index after download",
            )
            non_weight_files = [f for f in files if not f.endswith(".safetensors")]
            return non_weight_files + api_weight_files
        print(
            "  warning: repo API safetensors list differs from model.safetensors.index.json; "
            "using indexed shards",
        )

    non_weight_files = [f for f in files if not f.endswith(".safetensors")]
    return non_weight_files + indexed_shards


def curl_download_file(
    url: str,
    dest: Path,
    token: str,
    timeout: int,
    retries: int,
) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  exists {dest.name} ({dest.stat().st_size / (1024**2):.1f} MiB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".incomplete")
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        str(retries),
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        str(timeout),
        "--output",
        str(partial),
    ]
    if token:
        cmd.extend(["--header", f"Authorization: Bearer {token}"])
    cmd.append(url)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"curl failed with exit code {result.returncode}: {dest.name}")
    partial.replace(dest)


def cleanup_stale_artifacts(out_dir: Path) -> None:
    """Remove partial downloads left by interrupted curl runs."""
    if not out_dir.is_dir():
        return
    for path in out_dir.glob("*.incomplete"):
        if path.is_file():
            print(f"  removing stale partial: {path.name}")
            path.unlink()


def expected_weight_shards(out_dir: Path) -> set[str]:
    index_path = out_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return set()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map") or {}
    return {str(name) for name in set(weight_map.values())}


def verify_download(out_dir: Path) -> None:
    repair_index_if_needed(out_dir)
    shards = expected_weight_shards(out_dir)
    if not shards:
        single = out_dir / "model.safetensors"
        if single.is_file():
            print(f"Verified single weight file: {single.name}")
            return
        raise SystemExit(f"Missing weight index or model.safetensors in {out_dir}")

    missing = sorted(shard for shard in shards if not (out_dir / shard).is_file())
    if missing:
        raise SystemExit(f"Incomplete download in {out_dir}; missing shards: {missing}")
    total_bytes = sum((out_dir / shard).stat().st_size for shard in shards)
    print(f"Verified {len(shards)} weight shards ({total_bytes / (1024**3):.2f} GiB)")


def local_weight_files(out_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in out_dir.glob("model-*.safetensors")
        if path.is_file()
    )


def rebuild_safetensors_index(out_dir: Path, weight_files: list[str]) -> None:
    if not weight_files:
        return
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit(
            "Cannot rebuild model.safetensors.index.json without the safetensors package. "
            "Install it in your venv: pip install safetensors"
        ) from exc

    weight_map: dict[str, str] = {}
    for filename in weight_files:
        path = out_dir / filename
        with safe_open(path, framework="numpy") as handle:
            for key in handle.keys():
                weight_map[key] = filename

    index_path = out_dir / "model.safetensors.index.json"
    if index_path.exists():
        backup = out_dir / "model.safetensors.index.json.upstream_mismatch"
        if not backup.exists():
            backup.write_text(index_path.read_text(encoding="utf-8"), encoding="utf-8")
    payload = {
        "metadata": {
            "total_size": sum((out_dir / filename).stat().st_size for filename in weight_files),
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Rebuilt model.safetensors.index.json for {len(weight_files)} local shards")


def repair_index_if_needed(out_dir: Path) -> None:
    weight_files = local_weight_files(out_dir)
    if not weight_files:
        return
    indexed = sorted(expected_weight_shards(out_dir))
    if indexed and indexed == weight_files:
        return
    print(f"  warning: local index shards {indexed or '<missing>'} do not match local weights {weight_files}")
    rebuild_safetensors_index(out_dir, weight_files)


def check_connectivity(repo_id: str, args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    token, endpoint, timeout, max_workers = hub_settings(args)
    print(f"Checking {repo_id}")
    print_hub_settings(token, endpoint, timeout, max_workers)
    transport = selected_transport(args, endpoint)
    print(f"  transport={transport}")

    if transport == "curl":
        files = curl_repo_files(repo_id, endpoint, args.revision, token, timeout, args.retries)
        print(f"  metadata_ok={len(files)} files visible")
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            curl_download_file(
                resolve_file_url(endpoint, repo_id, args.revision, "config.json"),
                config_path,
                token,
                timeout,
                args.retries,
            )
            print(f"  small_file_ok={config_path.name} ({config_path.stat().st_size} bytes)")
        return

    try:
        files = snapshot_download(
            repo_id=repo_id,
            dry_run=True,
            token=token or None,
            endpoint=endpoint,
            revision=args.revision,
            etag_timeout=timeout,
            max_workers=1,
        )
    except Exception as exc:
        raise SystemExit(f"Metadata check failed: {type(exc).__name__}: {exc}") from exc
    print(f"  metadata_ok={len(files)} files visible")

    try:
        with TemporaryDirectory() as tmp_dir:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                    local_dir=tmp_dir,
                    token=token or None,
                    endpoint=endpoint,
                    revision=args.revision,
                    etag_timeout=timeout,
                )
            )
            print(f"  small_file_ok={path.name} ({path.stat().st_size} bytes)")
    except Exception as exc:
        raise SystemExit(
            "Small-file download failed. If metadata succeeded, the proxy is likely "
            f"unstable during HTTPS download: {type(exc).__name__}: {exc}"
        ) from exc


def download(repo_id: str, out_dir: Path, args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_artifacts(out_dir)
    token, endpoint, timeout, max_workers = hub_settings(args)
    print(f"Downloading {repo_id} -> {out_dir}")
    print_hub_settings(token, endpoint, timeout, max_workers)
    transport = selected_transport(args, endpoint)
    print(f"  transport={transport}")

    if transport == "curl":
        files = curl_repo_files(repo_id, endpoint, args.revision, token, timeout, args.retries)
        print(f"  files={len(files)}")
        for index, filename in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] {filename}")
            curl_download_file(
                resolve_file_url(endpoint, repo_id, args.revision, filename),
                out_dir / filename,
                token,
                timeout,
                args.retries,
            )
        repair_index_if_needed(out_dir)
        verify_download(out_dir)
        print(f"Done: {out_dir}")
        return

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        token=token or None,
        endpoint=endpoint,
        revision=args.revision,
        etag_timeout=timeout,
        max_workers=max_workers,
    )
    repair_index_if_needed(out_dir)
    verify_download(out_dir)
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
    parser.add_argument(
        "--env-file",
        type=Path,
        default=root / ".env",
        help="Load environment variables from this file if they are not already set.",
    )
    parser.add_argument(
        "--no-env-file",
        action="store_const",
        const=None,
        dest="env_file",
        help="Do not load .env before contacting Hugging Face.",
    )
    parser.add_argument(
        "--proxy",
        help="HTTP proxy URL, for example http://127.0.0.1:1082.",
    )
    parser.add_argument(
        "--endpoint",
        help="Override Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repo revision, branch, tag, or commit to download.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help=f"Hub metadata/download timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help=f"Parallel downloads (default: {DEFAULT_MAX_WORKERS}; use 1 for unstable proxies).",
    )
    parser.add_argument(
        "--disable-xet",
        action="store_true",
        help="Set HF_HUB_DISABLE_XET=1 before importing huggingface_hub.",
    )
    parser.add_argument(
        "--transport",
        choices=("auto", "hub", "curl"),
        default="auto",
        help="Download backend. auto uses curl for non-official endpoints.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"curl retry count for the curl transport (default: {DEFAULT_RETRIES}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only test repo metadata and config.json download; do not download weights.",
    )
    parser.add_argument(
        "--verify-local",
        action="store_true",
        help="Repair model.safetensors.index.json if needed and verify an existing local dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_environment(args)
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

    if args.check:
        check_connectivity(repo_id, args)
    elif args.verify_local:
        cleanup_stale_artifacts(out_dir)
        repair_index_if_needed(out_dir)
        verify_download(out_dir)
        print(f"Verified local model: {out_dir}")
    else:
        download(repo_id, out_dir.resolve(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
