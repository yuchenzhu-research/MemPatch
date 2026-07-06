#!/usr/bin/env python3
"""Run a small sequential local Ollama smoke test for MemPatch-Bench final."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from mempatch.benchmark.generate import generate_raw_files
from mempatch.benchmark.leakage import audit_public_rows
from mempatch.benchmark.general_taxonomy import DECISIONS, FAILURE_MODES, MEMORY_OPERATIONS, MEMORY_STATUSES
from mempatch.benchmark.model_runner import canonical_response, extract_json_object
from mempatch.benchmark.release import export_release, read_jsonl
from mempatch.benchmark.score import score_row
from mempatch.benchmark.public_view import public_scenario_view
from mempatch.revision.runtime.ablation_projection import project_actions_without_dpa
from mempatch.revision.runtime.dpa_runtime import parse_actions
from mempatch.revision.runtime.learned_proposer import build_proposer_prompt
from mempatch.revision.runtime.revision_module import run_revision_module_on_scenario
from mempatch.revision.runtime.scenario_revision import build_scenario_revision_view
from tools.evaluation.server.methods import build_method_view
from mempatch_bench.runners.ollama_client import OllamaClient, OllamaError


EMPTY_RESPONSE = {
    "answer": "",
    "decision": None,
    "memory_operation": None,
    "memory_state": {},
    "evidence_event_ids": [],
    "failure_diagnosis": None,
    "followup_answer": "",
}


def collect_memory_ids(public_view: dict[str, Any]) -> list[str]:
    public_input = public_view.get("public_input", {})
    memory_ids = [
        memory["memory_id"]
        for memory in public_input.get("initial_memory") or public_input.get("initial_memories") or []
        if isinstance(memory, dict) and memory.get("memory_id")
    ]
    for event in public_input.get("event_trace") or public_input.get("events") or []:
        if not isinstance(event, dict):
            continue
        for memory_id in event.get("related_memory_ids") or []:
            if memory_id not in memory_ids:
                memory_ids.append(memory_id)
    return memory_ids


def build_benchmark_prompt(public_view: dict[str, Any]) -> str:
    memory_ids = collect_memory_ids(public_view)
    payload = {
        "instruction": (
            "Answer as strict JSON only matching the schema exactly. Do not use Markdown fences. "
            "Use only the visible scenario content. Do not use external knowledge. "
            "Use exact enum strings. Do not invent memory IDs or event IDs. "
            "Cite only minimal supporting event IDs. "
            "Choose exactly one lifecycle memory_operation for the durable memory action. "
            "CRITICAL: decision, memory_operation, and failure_diagnosis must be scalar strings, not lists. "
            "CRITICAL: every memory_state value must be exactly one allowed memory status string. "
            "Never output custom memory_state values such as background, ignored, irrelevant, active, inactive, unchanged, or valid. "
            "Decision order: refuse_due_to_policy, escalate, ask_clarification, mark_unresolved, use_current_memory."
        ),
        "allowed_enums": {
            "decision": list(DECISIONS),
            "memory_operation": list(MEMORY_OPERATIONS),
            "memory_status": list(MEMORY_STATUSES),
            "failure_diagnosis": list(FAILURE_MODES),
        },
        "required_output_schema": {
            "answer": "short final answer/action text (string)",
            "decision": "exactly one allowed decision string",
            "memory_operation": "exactly one allowed memory_operation string",
            "memory_state": {memory_id: "exactly one allowed memory_status string" for memory_id in memory_ids},
            "evidence_event_ids": "minimal list of event_id strings copied exactly from public_input.events",
            "failure_diagnosis": "exactly one allowed failure_diagnosis string",
            "followup_answer": "short answer to the visible followup_task after applying the memory operation",
        },
        **public_view,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


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


def normalize_scalar_lists(parsed: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(parsed)
    for key in ("decision", "memory_operation", "failure_diagnosis"):
        value = cleaned.get(key)
        if isinstance(value, list) and len(value) == 1:
            cleaned[key] = value[0]
    mem = cleaned.get("memory_state")
    if isinstance(mem, dict):
        cleaned["memory_state"] = {
            key: value[0] if isinstance(value, list) and len(value) == 1 else value
            for key, value in mem.items()
        }
    return cleaned


def safe_response(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = normalize_scalar_lists(extract_json_object(text))
        return canonical_response(parsed), None
    except Exception as exc:
        return dict(EMPTY_RESPONSE), f"{type(exc).__name__}: {exc}"


def restore_action_array(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.replace("```json", "").replace("```", "").strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def generation_kwargs(config: dict[str, Any], *, kind: str) -> dict[str, Any]:
    gen = config["generation"]
    return {
        "temperature": float(gen.get("temperature", 0.0)),
        "top_p": float(gen.get("top_p", 1.0)),
        "seed": gen.get("seed", 42),
        "num_ctx": int(gen.get("num_ctx", 8192)),
        "num_predict": int(gen.get("action_num_predict" if kind == "action" else "response_num_predict", 1024)),
        "keep_alive": str(gen.get("keep_alive", "5m")),
    }


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


def run_direct(
    *,
    scenario: dict[str, Any],
    model: str,
    split: str,
    client: OllamaClient,
    config: dict[str, Any],
) -> dict[str, Any]:
    public_view = public_scenario_view(scenario)
    method_view = build_method_view("frozen_direct", public_view, int(config.get("retrieval_k", 8)))
    generation = client.chat(model=model, prompt=build_benchmark_prompt(method_view), **generation_kwargs(config, kind="response"))
    response, parse_error = safe_response(generation.text)
    return {
        "scenario_id": scenario["scenario_id"],
        "model": model,
        "method": "direct_json",
        "split": split,
        "response": response,
        "raw_response": generation.text,
        "raw_generation": {
            "latency_seconds": generation.latency_seconds,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
        },
        "parse_error": parse_error,
    }


def run_mempatch(
    *,
    scenario: dict[str, Any],
    model: str,
    split: str,
    client: OllamaClient,
    config: dict[str, Any],
) -> dict[str, Any]:
    public_view = public_scenario_view(scenario)
    direct_view = build_method_view("frozen_direct", public_view, int(config.get("retrieval_k", 8)))
    response_generation = client.chat(
        model=model,
        prompt=build_benchmark_prompt(direct_view),
        **generation_kwargs(config, kind="response"),
    )
    raw_response, response_parse_error = safe_response(response_generation.text)
    revision_view = build_scenario_revision_view(scenario)
    action_generation = client.chat(
        model=model,
        prompt=build_proposer_prompt(revision_view),
        **generation_kwargs(config, kind="action"),
    )
    actions_text = restore_action_array(action_generation.text)
    action_parse_error: str | None = None
    try:
        parse_result = parse_actions(actions_text)
        guarded = run_revision_module_on_scenario(
            scenario,
            actions_text=actions_text,
            raw_response=raw_response,
            include_audit=True,
        )
        response = guarded["response"]
        dpa_audit = guarded.get("dpa_audit")
        no_guard = project_actions_without_dpa(
            view=revision_view,
            parse_result=parse_result,
            raw_response=raw_response,
            scenario_public_view=public_view,
        )
    except Exception as exc:
        response = dict(EMPTY_RESPONSE)
        dpa_audit = None
        no_guard = None
        action_parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "scenario_id": scenario["scenario_id"],
        "model": model,
        "method": "mempatch",
        "split": split,
        "response": response,
        "raw_response": response_generation.text,
        "raw_actions": action_generation.text,
        "actions_text": actions_text,
        "raw_generation": {
            "response_latency_seconds": response_generation.latency_seconds,
            "action_latency_seconds": action_generation.latency_seconds,
            "response_input_tokens": response_generation.input_tokens,
            "response_output_tokens": response_generation.output_tokens,
            "action_input_tokens": action_generation.input_tokens,
            "action_output_tokens": action_generation.output_tokens,
        },
        "parse_error": response_parse_error or action_parse_error,
        "response_parse_error": response_parse_error,
        "action_parse_error": action_parse_error,
        "dpa_audit": dpa_audit,
        "no_guard_response": no_guard,
    }


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def macro_f1(rows: list[dict[str, Any]], class_key: str, correct_key: str) -> float:
    by_class: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        cls = str(row.get(class_key) or "")
        by_class[cls].append(bool(row.get(correct_key)))
    if not by_class:
        return 0.0
    # With only correctness and gold class available, this is macro recall,
    # equivalent to macro F1 only when predicted class counts match gold counts.
    return sum(sum(vals) / len(vals) for vals in by_class.values() if vals) / len(by_class)


def aggregate(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(score_rows)
    if not n:
        return {
            "n": 0,
            "schema_valid_rate": 0.0,
            "exact_state_map": 0.0,
            "contract_valid_state_success": 0.0,
            "decision_macro_f1": 0.0,
            "evidence_f1": 0.0,
            "diagnosis_accuracy": 0.0,
            "strict_joint": 0.0,
            "unsafe_reuse_rate": 0.0,
            "parse_failure_rate": 0.0,
        }
    return {
        "n": n,
        "schema_valid_rate": sum(bool(r.get("schema_valid")) for r in score_rows) / n,
        "exact_state_map": sum(bool(r.get("exact_state_map")) for r in score_rows) / n,
        "contract_valid_state_success": sum(bool(r.get("schema_valid")) and bool(r.get("exact_state_map")) for r in score_rows) / n,
        "decision_macro_f1": macro_f1(score_rows, "decision_f1_class", "decision_correct"),
        "evidence_f1": sum(float(r.get("evidence_f1") or 0.0) for r in score_rows) / n,
        "diagnosis_accuracy": sum(bool(r.get("diagnosis_correct")) for r in score_rows) / n,
        "strict_joint": sum(bool(r.get("strict_joint")) for r in score_rows) / n,
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
        f"- Pytest preflight: {manifest.get('pytest_preflight')}",
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
        "| model | method | split | n | schema_valid_rate | exact_state_map | contract_valid_state_success | decision_macro_f1 | evidence_f1 | diagnosis_accuracy | strict_joint | unsafe_reuse_rate | parse_failure_rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {model} | {method} | {split} | {n} | {schema_valid_rate:.3f} | {exact_state_map:.3f} | "
            "{contract_valid_state_success:.3f} | {decision_macro_f1:.3f} | {evidence_f1:.3f} | "
            "{diagnosis_accuracy:.3f} | {strict_joint:.3f} | {unsafe_reuse_rate:.3f} | {parse_failure_rate:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "`decision_macro_f1` is the smoke-run class-balanced decision correctness available from deterministic score rows.",
        ]
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
    base_url = os.environ.get(config.get("ollama_base_url_env", "OLLAMA_BASE_URL")) or config.get("default_ollama_base_url", "http://localhost:11434")
    gen = config["generation"]
    client = OllamaClient(
        base_url=base_url,
        timeout_seconds=float(gen.get("timeout_seconds", 240)),
        retries=int(gen.get("retries", 1)),
        retry_sleep_seconds=float(gen.get("retry_sleep_seconds", 3)),
    )
    available = client.model_names()
    requested = models_override or list(config["models"])
    runnable = [model for model in requested if model in available]
    skipped = [model for model in requested if model not in available]
    if not runnable:
        raise RuntimeError(f"no requested Ollama models are available: {requested}")
    manifest: dict[str, Any] = {
        "started_at": utc_now(),
        "repository_sha": git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ollama_base_url": base_url,
        "models_requested": requested,
        "models_available": available,
        "models_skipped": skipped,
        "methods": list(config["methods"]),
        "benchmark": data_info,
        "model_failures": [],
        "jobs": [],
        "pytest_preflight": "run separately; see console output",
    }
    all_predictions: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for model in runnable:
        for method in config["methods"]:
            for split_conf in config["benchmark"]["splits"]:
                split = str(split_conf["name"])
                max_cases = int(max_cases_override or split_conf.get("max_cases", 20))
                scenarios = read_jsonl(Path(data_info["raw_paths"][split]))[:max_cases]
                labels = read_jsonl(Path(data_info["release_label_paths"][split]))
                pred_path = prediction_dir / f"{slug(model)}.{method}.{split}.predictions.jsonl"
                if resume and pred_path.exists():
                    existing_predictions = read_jsonl(pred_path)
                    if len(existing_predictions) >= len(scenarios):
                        predictions = existing_predictions[: len(scenarios)]
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
                        all_predictions.extend(predictions)
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
                                "prediction_path": str(pred_path),
                                "elapsed_seconds": 0.0,
                                "resumed": True,
                                "aggregate": agg,
                            }
                        )
                        print(f"Reused existing predictions model={model} method={method} split={split}", flush=True)
                        continue
                predictions: list[dict[str, Any]] = []
                job_start = time.perf_counter()
                print(f"Running model={model} method={method} split={split} cases={len(scenarios)}", flush=True)
                for index, scenario in enumerate(scenarios, start=1):
                    try:
                        if method == "direct_json":
                            row = run_direct(scenario=scenario, model=model, split=split, client=client, config=config)
                        elif method == "mempatch":
                            row = run_mempatch(scenario=scenario, model=model, split=split, client=client, config=config)
                        else:
                            raise ValueError(f"unknown method: {method}")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        row = empty_prediction(
                            scenario_id=str(scenario["scenario_id"]),
                            model=model,
                            method=method,
                            split=split,
                            error=error,
                        )
                        manifest["model_failures"].append(
                            {"model": model, "method": method, "split": split, "scenario_id": scenario["scenario_id"], "error": error}
                        )
                    predictions.append(row)
                    all_predictions.append(row)
                    print(f"  [{index}/{len(scenarios)}] {scenario['scenario_id']} parse_error={bool(row.get('parse_error'))}", flush=True)
                write_jsonl_rows(pred_path, predictions)
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
                        "prediction_path": str(pred_path),
                        "elapsed_seconds": round(time.perf_counter() - job_start, 3),
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
