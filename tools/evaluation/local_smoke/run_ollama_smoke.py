#!/usr/bin/env python3
"""Run a small sequential local Ollama smoke test for MemPatch-Bench final."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mempatch.benchmark.generate import generate_raw_files
from mempatch.benchmark.leakage import audit_public_rows
from mempatch.benchmark.release import export_release, read_jsonl
from mempatch.benchmark.score import score_row
from mempatch.evaluation import GenerationRecord, evaluate_case
from tools.evaluation.local_smoke.ollama_client import OllamaClient, OllamaError


EMPTY_RESPONSE = {
    "answer": "",
    "decision": None,
    "memory_operation": None,
    "memory_state": {},
    "evidence_event_ids": [],
    "failure_diagnosis": None,
    "followup_answer": "",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_sha256() -> str:
    """Fingerprint the shared pipeline and this adapter for safe resume."""
    root = Path(__file__).resolve().parents[3]
    paths = [
        *sorted((root / "mempatch").rglob("*.py")),
        *sorted(Path(__file__).resolve().parent.glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def ensure_medium_data(config: dict[str, Any]) -> dict[str, Any]:
    bench = config["benchmark"]
    raw_dir = Path(bench["raw_internal_dir"])
    release_dir = Path(bench["release_dir"])
    quotas = {str(key): int(value) for key, value in (bench.get("quotas") or {}).items()}
    raw_paths = {split: raw_dir / f"{split}.jsonl" for split in quotas}
    needs_generate = any(not path.exists() or sum(1 for _ in path.open()) < quotas[split] for split, path in raw_paths.items())
    if needs_generate:
        generate_raw_files(raw_dir, quotas)
    release_public_paths = {split: release_dir / "public" / f"{split}.jsonl" for split in quotas}
    release_label_paths = {split: release_dir / "labels" / f"{split}.labels.jsonl" for split in quotas}
    needs_export = needs_generate or any(
        not path.exists() or sum(1 for _ in path.open()) < quotas[split]
        for split in quotas
        for path in (release_public_paths[split], release_label_paths[split])
    )
    if needs_export:
        export_release(raw_paths, release_dir)
    violations: list[dict[str, Any]] = []
    for split, path in release_public_paths.items():
        split_violations = audit_public_rows(read_jsonl(path))
        for violation in split_violations:
            violations.append({"split": split, **violation})
    if violations:
        raise RuntimeError(f"public leakage audit failed: {violations[:5]}")
    return {
        "raw_paths": {split: str(path) for split, path in raw_paths.items()},
        "release_public_paths": {split: str(path) for split, path in release_public_paths.items()},
        "release_label_paths": {split: str(path) for split, path in release_label_paths.items()},
        "split_sizes": {split: sum(1 for _ in raw_paths[split].open()) for split in raw_paths},
        "release_dir": str(release_dir),
    }


def generation_kwargs(config: dict[str, Any], *, kind: str) -> dict[str, Any]:
    gen = config["generation"]
    default_num_predict = 384 if kind == "action" else 512
    return {
        "temperature": float(gen.get("temperature", 0.0)),
        "top_p": float(gen.get("top_p", 1.0)),
        "seed": gen.get("seed", 42),
        "num_ctx": int(gen.get("num_ctx", 8192)),
        "num_predict": int(gen.get("action_num_predict" if kind == "action" else "response_num_predict", default_num_predict)),
        "keep_alive": str(gen.get("keep_alive", "5m")),
    }


class OllamaGenerator:
    """Thin adapter; all benchmark logic stays in ``evaluate_case``."""

    def __init__(self, client: OllamaClient, model: str, config: dict[str, Any]) -> None:
        self.client = client
        self.model = model
        self.config = config

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationRecord:
        kind = "action" if "ACTIONS_JSON:" in prompt else "response"
        options = generation_kwargs(self.config, kind=kind)
        options["num_predict"] = max_new_tokens
        result = self.client.chat(model=self.model, prompt=prompt, **options)
        return GenerationRecord(
            text=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_seconds=result.latency_seconds,
        )


def run_case(
    scenario: dict[str, Any],
    *,
    model: str,
    split: str,
    methods: tuple[str, ...],
    client: OllamaClient,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run all requested methods together so Direct is generated only once."""
    result = evaluate_case(
        scenario,
        OllamaGenerator(client, model, config),
        methods=methods,
        retrieval_k=int(config.get("retrieval_k", 3)),
        response_tokens=int(config["generation"].get("response_num_predict", 512)),
        action_tokens=int(config["generation"].get("action_num_predict", 384)),
        include_no_guard="mempatch" in methods,
    )
    generations = result["generations"]
    direct_generation = generations.get("direct_json", {})
    action_generation = generations.get("mempatch_shared_actions", {})
    pairing = generations.get("pairing", {})
    rows: dict[str, dict[str, Any]] = {}

    for method in methods:
        row = {
            **result["predictions"][method],
            "model": model,
            "split": split,
            "paired_direct_response_sha256": pairing.get(
                "direct_response_sha256"
            ),
            "paired_public_view_sha256": pairing.get("public_view_sha256"),
        }
        if method == "mempatch":
            response_error = direct_generation.get("parse_error")
            action_error = (action_generation.get("parse_result") or {}).get(
                "error_message"
            )
            row.update(
                {
                    "raw_response": direct_generation.get("text", ""),
                    "raw_actions": action_generation.get("text", ""),
                    "actions_text": action_generation.get("actions_text", ""),
                    "response_parse_error": response_error,
                    "action_parse_error": action_error,
                    "parse_error": response_error or action_error,
                    "no_guard_response": result["predictions"]
                    .get("mempatch_noguard", {})
                    .get("response"),
                    "raw_generation": {
                        "response_latency_seconds": direct_generation.get(
                            "latency_seconds"
                        ),
                        "action_latency_seconds": action_generation.get(
                            "latency_seconds"
                        ),
                        "response_input_tokens": direct_generation.get(
                            "input_tokens"
                        ),
                        "response_output_tokens": direct_generation.get(
                            "output_tokens"
                        ),
                        "action_input_tokens": action_generation.get("input_tokens"),
                        "action_output_tokens": action_generation.get(
                            "output_tokens"
                        ),
                    },
                }
            )
        else:
            generation = generations.get(method, {})
            row.update(
                {
                    "raw_response": generation.get("text", ""),
                    "parse_error": generation.get("parse_error"),
                    "raw_generation": {
                        key: generation.get(key)
                        for key in (
                            "latency_seconds",
                            "input_tokens",
                            "output_tokens",
                        )
                    },
                }
            )
        rows[method] = row
    return rows


def empty_prediction(*, scenario_id: str, model: str, method: str, split: str, error: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "model": model,
        "method": method,
        "split": split,
        "response": dict(EMPTY_RESPONSE),
        "raw_response": "",
        "parse_error": error,
        "model_error": error,
    }


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reusable_prediction_rows(
    path: Path,
    *,
    scenario_ids: list[str],
    model: str,
    method: str,
    split: str,
    run_fingerprint: str,
) -> list[dict[str, Any]] | None:
    """Return an exact compatible prediction file, otherwise force a rerun."""
    if not path.exists():
        return None
    try:
        rows = read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return None
    if len(rows) != len(scenario_ids):
        return None
    if [str(row.get("scenario_id")) for row in rows] != scenario_ids:
        return None
    if any(
        str(row.get("model")) != model
        or str(row.get("method")) != method
        or str(row.get("split")) != split
        or str(row.get("run_fingerprint")) != run_fingerprint
        for row in rows
    ):
        return None
    return rows


def reusable_prediction_group(
    paths: dict[str, Path],
    *,
    scenario_ids: list[str],
    model: str,
    split: str,
    run_fingerprint: str,
) -> dict[str, list[dict[str, Any]]] | None:
    """Load one exact method group and verify Direct/MemPatch pairing."""
    loaded: dict[str, list[dict[str, Any]]] = {}
    for method, path in paths.items():
        rows = reusable_prediction_rows(
            path,
            scenario_ids=scenario_ids,
            model=model,
            method=method,
            split=split,
            run_fingerprint=run_fingerprint,
        )
        if rows is None:
            return None
        loaded[method] = rows

    if "direct_json" in loaded and "mempatch" in loaded:
        for direct, mempatch in zip(
            loaded["direct_json"],
            loaded["mempatch"],
            strict=True,
        ):
            direct_hash = direct.get("paired_direct_response_sha256")
            if not direct_hash or direct_hash != mempatch.get(
                "paired_direct_response_sha256"
            ):
                return None
    return loaded


def score_predictions(
    *,
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    out_path: Path,
) -> list[dict[str, Any]]:
    labels_by_id = {row["scenario_id"]: row for row in labels}
    score_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        label = labels_by_id.get(prediction["scenario_id"])
        if label is None:
            continue
        scored = score_row(label, prediction)
        scored["parse_failure"] = bool(prediction.get("parse_error") or prediction.get("model_error"))
        scored["parse_error"] = prediction.get("parse_error")
        score_rows.append(scored)
    write_jsonl_rows(out_path, score_rows)
    return score_rows


def aggregate(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(score_rows)
    if not n:
        return {
            "n": 0,
            "schema_valid_rate": 0.0,
            "exact_state_map": 0.0,
            "contract_valid_state_success": 0.0,
            "decision_accuracy": 0.0,
            "evidence_f1": 0.0,
            "diagnosis_accuracy": 0.0,
            "transition_joint": 0.0,
            "unsafe_reuse_rate": 0.0,
            "parse_failure_rate": 0.0,
        }
    return {
        "n": n,
        "schema_valid_rate": sum(bool(r.get("schema_valid")) for r in score_rows) / n,
        "exact_state_map": sum(bool(r.get("exact_state_map")) for r in score_rows) / n,
        "contract_valid_state_success": sum(bool(r.get("schema_valid")) and bool(r.get("exact_state_map")) for r in score_rows) / n,
        "decision_accuracy": sum(bool(r.get("decision_correct")) for r in score_rows) / n,
        "evidence_f1": sum(float(r.get("evidence_f1") or 0.0) for r in score_rows) / n,
        "diagnosis_accuracy": sum(bool(r.get("diagnosis_correct")) for r in score_rows) / n,
        "transition_joint": sum(bool(r.get("transition_joint")) for r in score_rows) / n,
        "unsafe_reuse_rate": sum(bool(r.get("unsafe_reuse")) for r in score_rows) / n,
        "parse_failure_rate": sum(bool(r.get("parse_failure")) for r in score_rows) / n,
    }


def write_aggregate(path_json: Path, path_csv: Path, row: dict[str, Any]) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def prediction_examples(predictions: list[dict[str, Any]], *, failures: bool, limit: int = 5) -> list[dict[str, Any]]:
    selected = []
    for row in predictions:
        failed = bool(row.get("parse_error") or row.get("model_error"))
        if failed != failures:
            continue
        selected.append(
            {
                "scenario_id": row.get("scenario_id"),
                "model": row.get("model"),
                "method": row.get("method"),
                "split": row.get("split"),
                "parse_error": row.get("parse_error") or row.get("model_error"),
                "response": row.get("response"),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def write_report(
    *,
    path: Path,
    manifest: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    all_predictions: list[dict[str, Any]],
) -> None:
    cases_per_split: dict[str, int] = {}
    for job in manifest.get("jobs") or []:
        split = str(job.get("split"))
        cases_per_split[split] = max(cases_per_split.get(split, 0), int(job.get("cases") or 0))
    overall_parse_failures = sum(1 for row in all_predictions if row.get("parse_error") or row.get("model_error"))
    overall_parse_failure_rate = overall_parse_failures / len(all_predictions) if all_predictions else 0.0
    lines = [
        "# Local Ollama Smoke Report",
        "",
        f"- Timestamp: {manifest['finished_at']}",
        "- Machine note: MacBook Pro M5 Pro, 48GB unified memory",
        f"- Ollama base URL: {manifest['ollama_base_url']}",
        f"- Repository SHA: {manifest.get('repository_sha')}",
        f"- Benchmark split sizes: {json.dumps(manifest['benchmark']['split_sizes'], sort_keys=True)}",
        f"- Methods run: {', '.join(manifest['methods'])}",
        f"- Models requested: {', '.join(manifest['models_requested'])}",
        f"- Models available: {', '.join(manifest['models_available'])}",
        f"- Models skipped: {', '.join(manifest['models_skipped']) if manifest['models_skipped'] else 'none'}",
        f"- Cases per split: {json.dumps(cases_per_split, sort_keys=True)}",
        f"- Prediction count: {len(all_predictions)}",
        f"- Model failures: {len(manifest['model_failures'])}",
        f"- Overall parse failure rate: {overall_parse_failure_rate:.3f}",
        "",
        "## Aggregate Metrics",
        "",
        "| model | method | split | n | schema_valid_rate | exact_state_map | contract_valid_state_success | decision_accuracy | evidence_f1 | diagnosis_accuracy | transition_joint | unsafe_reuse_rate | parse_failure_rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {model} | {method} | {split} | {n} | {schema_valid_rate:.3f} | {exact_state_map:.3f} | "
            "{contract_valid_state_success:.3f} | {decision_accuracy:.3f} | {evidence_f1:.3f} | "
            "{diagnosis_accuracy:.3f} | {transition_joint:.3f} | {unsafe_reuse_rate:.3f} | {parse_failure_rate:.3f} |".format(**row)
        )
    lines.extend(["", "## Parse Failure Examples", ""])
    failures = prediction_examples(all_predictions, failures=True)
    if failures:
        lines.append("```json")
        lines.append(json.dumps(failures, indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("None recorded.")
    lines.extend(["", "## Parse-Successful Prediction Examples", ""])
    successes = prediction_examples(all_predictions, failures=False)
    if successes:
        lines.append("```json")
        lines.append(json.dumps(successes, indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("None recorded.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use this local run only for adapter, prompt, parsing, and scorer validation. Run full benchmark campaigns on a server after local parse/schema stability is acceptable.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config: dict[str, Any],
    *,
    max_cases_override: int | None = None,
    models_override: list[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    output_root = Path(config.get("output_root", "results/local_ollama_smoke"))
    prediction_dir = output_root / "predictions"
    score_dir = output_root / "scores"
    aggregate_dir = output_root / "aggregates"
    manifest_path = output_root / "run_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    data_info = ensure_medium_data(config)
    scenarios_by_split: dict[str, list[dict[str, Any]]] = {}
    labels_by_split: dict[str, list[dict[str, Any]]] = {}
    split_profile: dict[str, dict[str, Any]] = {}
    for split_conf in config["benchmark"]["splits"]:
        split = str(split_conf["name"])
        max_cases = int(
            max_cases_override
            if max_cases_override is not None
            else split_conf.get("max_cases", 20)
        )
        scenarios = read_jsonl(Path(data_info["raw_paths"][split]))[:max_cases]
        labels = read_jsonl(Path(data_info["release_label_paths"][split]))
        scenario_ids = [str(row["scenario_id"]) for row in scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise RuntimeError(f"duplicate scenario IDs in split {split}")
        scenarios_by_split[split] = scenarios
        labels_by_split[split] = labels
        split_profile[split] = {
            "cases": len(scenarios),
            "scenario_ids": scenario_ids,
            "scenario_ids_sha256": stable_sha256(scenario_ids),
            "selected_data_sha256": stable_sha256(scenarios),
            "labels_sha256": stable_sha256(labels),
        }
    base_url = os.environ.get(config.get("ollama_base_url_env", "OLLAMA_BASE_URL")) or config.get("default_ollama_base_url", "http://localhost:11434")
    gen = config["generation"]
    client = OllamaClient(
        base_url=base_url,
        timeout_seconds=float(gen.get("timeout_seconds", 240)),
        retries=int(gen.get("retries", 1)),
        retry_sleep_seconds=float(gen.get("retry_sleep_seconds", 3)),
    )
    model_inventory = client.model_inventory()
    available = list(model_inventory)
    requested = models_override or list(config["models"])
    runnable = [model for model in requested if model in available]
    skipped = [model for model in requested if model not in available]
    if not runnable:
        raise RuntimeError(f"no requested Ollama models are available: {requested}")
    methods = tuple(str(method) for method in config["methods"])
    model_digests = {model: model_inventory[model] for model in runnable}
    model_digests_complete = all(model_digests.values())
    run_profile = {
        "pipeline": "paired_case_v1",
        "source_sha256": source_sha256(),
        "ollama_base_url": base_url,
        "models_requested": requested,
        "models_runnable": runnable,
        "model_digests": model_digests,
        "methods": list(methods),
        "retrieval_k": int(config.get("retrieval_k", 3)),
        "generation": config["generation"],
        "splits": split_profile,
    }
    run_fingerprint = stable_sha256(run_profile)
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume and manifest_path.exists()
        else {}
    )
    resume_compatible = (
        model_digests_complete
        and previous_manifest.get("run_fingerprint") == run_fingerprint
    )
    manifest: dict[str, Any] = {
        "pipeline": "paired_case_v1",
        "run_fingerprint": run_fingerprint,
        "run_profile": run_profile,
        "started_at": utc_now(),
        "repository_sha": git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ollama_base_url": base_url,
        "models_requested": requested,
        "models_runnable": runnable,
        "model_digests": model_digests,
        "model_digests_complete": model_digests_complete,
        "models_available": available,
        "models_skipped": skipped,
        "methods": list(methods),
        "pairing": "one Direct response shared with MemPatch per model-case",
        "benchmark": data_info,
        "model_failures": [],
        "jobs": [],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    all_predictions: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for model in runnable:
        for split_conf in config["benchmark"]["splits"]:
            split = str(split_conf["name"])
            scenarios = scenarios_by_split[split]
            labels = labels_by_split[split]
            scenario_ids = split_profile[split]["scenario_ids"]
            pred_paths = {
                method: prediction_dir
                / f"{slug(model)}.{method}.{split}.predictions.jsonl"
                for method in methods
            }
            reused = (
                reusable_prediction_group(
                    pred_paths,
                    scenario_ids=scenario_ids,
                    model=model,
                    split=split,
                    run_fingerprint=run_fingerprint,
                )
                if resume_compatible
                else None
            )
            resumed = reused is not None
            started = time.perf_counter()

            if resumed:
                predictions_by_method = reused
                print(
                    f"Reused paired predictions model={model} split={split}",
                    flush=True,
                )
            else:
                predictions_by_method = {method: [] for method in methods}
                print(
                    f"Running model={model} split={split} "
                    f"methods={','.join(methods)} cases={len(scenarios)}",
                    flush=True,
                )
                for index, scenario in enumerate(scenarios, start=1):
                    try:
                        rows = run_case(
                            scenario,
                            model=model,
                            split=split,
                            methods=methods,
                            client=client,
                            config=config,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        rows = {
                            method: empty_prediction(
                                scenario_id=str(scenario["scenario_id"]),
                                model=model,
                                method=method,
                                split=split,
                                error=error,
                            )
                            for method in methods
                        }
                        manifest["model_failures"].append(
                            {
                                "model": model,
                                "methods": list(methods),
                                "split": split,
                                "scenario_id": scenario["scenario_id"],
                                "error": error,
                            }
                        )
                    for method in methods:
                        rows[method]["run_fingerprint"] = run_fingerprint
                        predictions_by_method[method].append(rows[method])
                    parse_failures = sum(
                        bool(row.get("parse_error")) for row in rows.values()
                    )
                    print(
                        f"  [{index}/{len(scenarios)}] {scenario['scenario_id']} "
                        f"parse_failures={parse_failures}",
                        flush=True,
                    )
                for method, predictions in predictions_by_method.items():
                    write_jsonl_rows(pred_paths[method], predictions)

            elapsed = 0.0 if resumed else round(time.perf_counter() - started, 3)
            for method, predictions in predictions_by_method.items():
                all_predictions.extend(predictions)
                scores = score_predictions(
                    predictions=predictions,
                    labels=labels,
                    out_path=score_dir / f"{slug(model)}.{method}.{split}.scores.jsonl",
                )
                all_scores.extend(scores)
                agg = {
                    "model": model,
                    "method": method,
                    "split": split,
                    **aggregate(scores),
                }
                aggregate_rows.append(agg)
                write_aggregate(
                    aggregate_dir / f"{slug(model)}.{method}.{split}.aggregate.json",
                    aggregate_dir / f"{slug(model)}.{method}.{split}.aggregate.csv",
                    agg,
                )
                manifest["jobs"].append(
                    {
                        "model": model,
                        "method": method,
                        "split": split,
                        "cases": len(scenarios),
                        "prediction_path": str(pred_paths[method]),
                        "elapsed_seconds": elapsed,
                        "resumed": resumed,
                        "aggregate": agg,
                    }
                )
        client.unload(model)
    manifest["finished_at"] = utc_now()
    manifest["prediction_count"] = len(all_predictions)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_aggregate(
        aggregate_dir / "local_ollama_smoke.aggregate.json",
        aggregate_dir / "local_ollama_smoke.aggregate.csv",
        {"model": "ALL", "method": "ALL", "split": "ALL", **aggregate(all_scores)},
    )
    write_report(
        path=Path("docs/local_ollama_smoke_report.md"),
        manifest=manifest,
        aggregate_rows=aggregate_rows,
        all_predictions=all_predictions,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/local_ollama_smoke.yaml"))
    parser.add_argument("--max-cases", type=int, help="Override per-split max cases for quick debug runs.")
    parser.add_argument("--models", help="Comma-separated model subset in sequential order.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing prediction files when they cover the requested cases.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    models = [item.strip() for item in args.models.split(",") if item.strip()] if args.models else None
    try:
        run(config, max_cases_override=args.max_cases, models_override=models, resume=args.resume)
    except OllamaError as exc:
        print(f"Ollama smoke stopped: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Ollama smoke stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
