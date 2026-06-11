#!/usr/bin/env python3
"""Support checks for the Linux resume + 8-baseline smoke campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._root import bootstrap_from

bootstrap_from(__file__)

from scripts.data.prepare_mempatch_v13_smoke import (  # noqa: E402
    assert_label_coverage,
    fixed_stratified_split,
    read_jsonl,
    multitask_sft_examples,
    write_jsonl,
)

REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "datasets",
    "accelerate",
    "huggingface-hub",
)

PACKAGE_MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "trl": "trl",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "huggingface-hub": "huggingface_hub",
}

BASELINE_IDS = (
    "structured_direct",
    "full_context",
    "vanilla_rag",
    "bm25_rag",
    "time_aware_rag",
    "summary_memory",
    "mem0",
    "a_mem",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    errors: list[str] = []
    for name in REQUIRED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package: {name}")
            continue
        try:
            importlib.import_module(PACKAGE_MODULES[name])
        except Exception as exc:  # third-party import failures are part of the smoke gate
            errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")
    return versions, errors


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=root, text=True).strip()

    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "describe": run("git", "describe", "--always", "--dirty"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "describe": None, "dirty": None}


def _model_complete(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    if (path / "model.safetensors").is_file():
        return True
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return bool(shards) and all((path / shard).is_file() for shard in shards)


def command_preflight(args: argparse.Namespace) -> int:
    errors: list[str] = []
    versions, package_errors = _package_versions()
    errors.extend(package_errors)

    api: dict[str, Any] = {}
    cuda: dict[str, Any] = {}
    if not package_errors:
        import torch
        from trl import SFTConfig, SFTTrainer

        config_parameters = inspect.signature(SFTConfig).parameters
        trainer_parameters = inspect.signature(SFTTrainer).parameters
        api = {
            "sft_length_parameter": next(
                (name for name in ("max_length", "max_seq_length") if name in config_parameters),
                None,
            ),
            "sft_eval_parameter": next(
                (name for name in ("eval_strategy", "evaluation_strategy") if name in config_parameters),
                None,
            ),
            "trainer_tokenizer_parameter": next(
                (name for name in ("processing_class", "tokenizer") if name in trainer_parameters),
                None,
            ),
        }
        for name, value in api.items():
            if value is None:
                errors.append(f"unsupported TRL API: {name}")
        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16_supported": bool(
                torch.cuda.is_available()
                and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            ),
            "torch_cuda": torch.version.cuda,
        }
        if not cuda["available"]:
            errors.append("CUDA is not available")
        if not cuda["bf16_supported"]:
            errors.append("CUDA device does not report bfloat16 support")

    datasets: dict[str, Any] = {}
    for name, path, expected in (
        ("train", args.train_data, 3500),
        ("test", args.test_data, 500),
    ):
        if not path.is_file():
            errors.append(f"missing local {name} dataset: {path}")
            continue
        rows = read_jsonl(path)
        datasets[name] = {"path": str(path), "rows": len(rows), "sha256": _sha256(path)}
        if len(rows) != expected:
            errors.append(f"{name} dataset has {len(rows)} rows, expected {expected}")

    models: dict[str, Any] = {}
    for value in args.model:
        slug, raw_path = value.split("=", 1)
        path = Path(raw_path)
        complete = _model_complete(path)
        models[slug] = {"path": str(path), "complete": complete}
        if not complete:
            errors.append(f"local model is incomplete: {slug}={path}")

    offline = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    if any(value != "1" for value in offline.values()):
        errors.append(f"offline flags are not all enabled: {offline}")

    payload = {
        "ok": not errors,
        "errors": errors,
        "python": sys.version,
        "packages": versions,
        "trl_api": api,
        "cuda": cuda,
        "offline": offline,
        "datasets": datasets,
        "models": models,
        "git": _git_state(Path(__file__).resolve().parents[2]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def command_prepare_sft(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.train_data)
    train_rows, valid_rows = fixed_stratified_split(
        rows,
        split_index=args.split_index,
        split_parts=args.split_parts,
        seed=args.seed + 1,
    )
    assert_label_coverage(train_rows, split_name="train")
    assert_label_coverage(valid_rows, split_name="validation")
    selected_train = train_rows[: args.train_rows]
    selected_valid = valid_rows[: args.valid_rows]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.out_dir / "train.jsonl",
        [example for row in selected_train for example in multitask_sft_examples(row)],
    )
    write_jsonl(
        args.out_dir / "valid.jsonl",
        [example for row in selected_valid for example in multitask_sft_examples(row)],
    )
    manifest = {
        "source": str(args.train_data),
        "full_train_rows": len(train_rows),
        "full_validation_rows": len(valid_rows),
        "smoke_train_rows": len(selected_train),
        "smoke_validation_rows": len(selected_valid),
        "train_scenario_ids": [row["scenario_id"] for row in selected_train],
        "validation_scenario_ids": [row["scenario_id"] for row in selected_valid],
        "split_index": args.split_index,
        "split_parts": args.split_parts,
        "seed": args.seed,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_verify_path_a(args: argparse.Namespace) -> int:
    from benchmark.api import load_scenarios
    from mempatch.revision.runtime.revision_module import run_revision_module_on_scenario

    scenario = load_scenarios(args.test_data)[0]
    prediction = run_revision_module_on_scenario(scenario)
    response = prediction.get("response") or {}
    expected_fields = {
        "answer",
        "decision",
        "memory_state",
        "evidence_event_ids",
        "failure_diagnosis",
    }
    errors: list[str] = []
    if set(response) != expected_fields:
        errors.append(f"response fields={sorted(response)} expected={sorted(expected_fields)}")
    payload = {
        "ok": not errors,
        "errors": errors,
        "scenario_id": prediction.get("scenario_id"),
        "response": response,
        "scope": "deterministic no-op proposer through Path A and DPA; not a learned-proposer evaluation",
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_verify_resume(args: argparse.Namespace) -> int:
    errors: list[str] = []
    first = args.output_dir / "checkpoint-1"
    second = args.output_dir / "checkpoint-2"
    for step, path in ((1, first), (2, second)):
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            errors.append(f"missing {state_path}")
            continue
        state = _load_json(state_path)
        if int(state.get("global_step", -1)) != step:
            errors.append(f"{state_path} global_step={state.get('global_step')} expected {step}")
        if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            errors.append(f"missing adapter weights in {path}")
        for name in ("optimizer.pt", "scheduler.pt"):
            if not (path / name).is_file():
                errors.append(f"missing resumable trainer state: {path / name}")

    metrics_path = args.log_dir / "trainer_metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
    config = metrics.get("training_config") or {}
    if config.get("resume_global_step") != 1:
        errors.append(f"resume_global_step={config.get('resume_global_step')} expected 1")
    if config.get("final_global_step") != 2:
        errors.append(f"final_global_step={config.get('final_global_step')} expected 2")
    if Path(str(config.get("resume_from_checkpoint", ""))).name != "checkpoint-1":
        errors.append("trainer metrics do not record checkpoint-1 as resume source")

    payload = {
        "ok": not errors,
        "errors": errors,
        "output_dir": str(args.output_dir),
        "log_dir": str(args.log_dir),
        "resume_from_checkpoint": config.get("resume_from_checkpoint"),
        "resume_global_step": config.get("resume_global_step"),
        "final_global_step": config.get("final_global_step"),
        "package_versions": metrics.get("package_versions"),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def command_verify_eval(args: argparse.Namespace) -> int:
    errors: list[str] = []
    tags = [f"{args.prefix}_lora_best", *(f"{args.prefix}_baseline_{x}" for x in BASELINE_IDS)]
    rows: dict[str, Any] = {}
    for tag in tags:
        manifest_path = args.result_dir / f"{tag}_manifest.json"
        predictions_path = args.result_dir / f"{tag}_predictions.jsonl"
        if not manifest_path.is_file():
            errors.append(f"missing {manifest_path}")
            continue
        if not predictions_path.is_file():
            errors.append(f"missing {predictions_path}")
            continue
        predictions = [line for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
        manifest = _load_json(manifest_path)
        rows[tag] = {
            "prediction_rows": len(predictions),
            "headline_metrics": manifest.get("headline_metrics"),
        }
        if len(predictions) != 1:
            errors.append(f"{tag} has {len(predictions)} predictions, expected 1")
    no_dpa_manifest = args.result_dir / f"{args.prefix}_lora_best_no_dpa_manifest.json"
    no_dpa_predictions = args.result_dir / f"{args.prefix}_lora_best_no_dpa_predictions.jsonl"
    if not no_dpa_manifest.is_file():
        errors.append(f"missing {no_dpa_manifest}")
    if not no_dpa_predictions.is_file():
        errors.append(f"missing {no_dpa_predictions}")
    done = args.result_dir / f"{args.prefix}_8plus1.done"
    if not done.is_file():
        errors.append(f"missing {done}")
    payload = {"ok": not errors, "errors": errors, "runs": rows}
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--out", type=Path, required=True)
    preflight.add_argument("--train-data", type=Path, required=True)
    preflight.add_argument("--test-data", type=Path, required=True)
    preflight.add_argument("--model", action="append", required=True, help="slug=/local/model/path")
    preflight.set_defaults(func=command_preflight)

    prepare = subparsers.add_parser("prepare-sft")
    prepare.add_argument("--train-data", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--train-rows", type=int, default=8)
    prepare.add_argument("--valid-rows", type=int, default=2)
    prepare.add_argument("--split-index", type=int, default=0)
    prepare.add_argument("--split-parts", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=command_prepare_sft)

    path_a = subparsers.add_parser("verify-path-a")
    path_a.add_argument("--test-data", type=Path, required=True)
    path_a.add_argument("--out", type=Path, required=True)
    path_a.set_defaults(func=command_verify_path_a)

    resume = subparsers.add_parser("verify-resume")
    resume.add_argument("--output-dir", type=Path, required=True)
    resume.add_argument("--log-dir", type=Path, required=True)
    resume.add_argument("--out", type=Path, required=True)
    resume.set_defaults(func=command_verify_resume)

    evaluate = subparsers.add_parser("verify-eval")
    evaluate.add_argument("--result-dir", type=Path, required=True)
    evaluate.add_argument("--prefix", default="smoke1")
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.set_defaults(func=command_verify_eval)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
