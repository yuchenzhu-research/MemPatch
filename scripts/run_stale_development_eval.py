#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

from retracemem.adapters.stale_adapter import StaleAdapter
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.evaluation.manifest import RunConfiguration, RunManifest, compute_file_sha256
from retracemem.evaluation.progress import ProgressReporter, ProgressSnapshot
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.providers.capped import CappedProviderWrapper
from retracemem.providers.http_provider import HTTPLLMProvider

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_REFERENCE_ROOT = "reference/STALE"
DEFAULT_STAGE_A_PROMPT_VERSION = "evidence_edge_prediction_v1"
DIMENSIONS = (
    ("dim1_query", "SR"),
    ("dim2_query", "PR"),
    ("dim3_query", "IPA"),
)


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_safe_text(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_safe_text(v)}" for k, v in sorted(value.items()))
    return str(value)


def _sessions_for_sample(sample: dict[str, Any]) -> list[str]:
    sessions = sample.get("sessions") or sample.get("haystack_session") or []
    if not isinstance(sessions, list):
        return []
    return [text for item in sessions if (text := _safe_text(item).strip())]


def _queries_for_sample(sample: dict[str, Any]) -> list[tuple[str, str, str]]:
    probing = sample.get("probing_queries")
    if not isinstance(probing, dict):
        probing = {}
    queries: list[tuple[str, str, str]] = []
    for key, dimension in DIMENSIONS:
        query = str(probing.get(key) or sample.get(key) or "").strip()
        if query:
            queries.append((key, dimension, query))
    return queries


def _answer_key(sample: dict[str, Any], dimension: str) -> str:
    if dimension in {"SR", "IPA"}:
        return str(sample.get("new_memory") or sample.get("M_new") or "").strip()
    return str(sample.get("old_memory") or sample.get("M_old") or "").strip()


def _prompt(stage: str, sample: dict[str, Any], query: str, dimension: str) -> str:
    sessions = "\n\n".join(_sessions_for_sample(sample))
    if stage == "A":
        instruction = (
            "Use evidence-preserving effect-triggered authorization. Do not treat silence or topical adjacency "
            "as a reason to discard unrelated memories. Answer the query from the available session evidence."
        )
    else:
        instruction = "Directly judge and answer the query from the available session evidence."
    return (
        f"Development-only STALE query. Dimension: {dimension}.\n"
        f"Instruction: {instruction}\n"
        f"Sessions available to the agent:\n{sessions}\n\n"
        f"Query: {query}\n"
        "Return only a concise answer, without JSON."
    )


def _score(answer: str, expected: str) -> int:
    normalized_answer = " ".join(answer.lower().split())
    normalized_expected = " ".join(expected.lower().split())
    if not normalized_expected:
        return 0
    return int(normalized_expected in normalized_answer or normalized_answer in normalized_expected)


def _prompt_hash(stage: str) -> str:
    path = REPO_ROOT / "prompts" / ("retrace_llm" if stage == "A" else "directjudge") / (
        f"{DEFAULT_STAGE_A_PROMPT_VERSION}.txt" if stage == "A" else "direct_usability_v1.txt"
    )
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return ""


def _make_client(mode: str, cache_path: str, provider: CappedProviderWrapper | MockLLMProvider) -> CachedLLMClient:
    return CachedLLMClient(cache=JSONLCache(cache_path), provider_client=provider, cost_accountant=CostAccounting())


def _cost_summary(client: CachedLLMClient, cap: CappedProviderWrapper | None, *, is_live: bool) -> dict[str, Any]:
    cost = client.cost_accountant.to_dict()
    calls = cost.get("calls", {})
    tokens = cost.get("tokens", {})
    outbound = cap.calls_made if cap is not None else 0
    outbound_tokens = cap.tokens_used if cap is not None else 0
    return {
        "semantic_invocations": calls.get("total", 0),
        "outbound_network_calls": outbound if is_live else 0,
        "cache_hits": cost.get("cache_hits", 0),
        "cache_misses": cost.get("cache_misses", 0),
        "tokens": tokens,
        "tokens_from_outbound_calls": outbound_tokens if is_live else 0,
        "latency_ms": round(cost.get("latency_ms", 0.0), 2),
        "errors": cost.get("error_counts", {}),
    }


def _snapshot(stage: str, scenarios_done: int, scenarios_total: int, queries_done: int, queries_total: int, client: CachedLLMClient, cap: CappedProviderWrapper | None, max_calls: int, max_tokens: int, current_id: str, is_live: bool) -> ProgressSnapshot:
    summary = _cost_summary(client, cap, is_live=is_live)
    return ProgressSnapshot(
        phase="generating",
        stage=stage,
        scenarios_done=scenarios_done,
        scenarios_total=scenarios_total,
        queries_done=queries_done,
        queries_total=queries_total,
        semantic_invocations=summary["semantic_invocations"],
        outbound_network_calls=summary["outbound_network_calls"],
        max_calls=max_calls,
        cache_hits=summary["cache_hits"],
        cache_misses=summary["cache_misses"],
        tokens_from_outbound_calls=summary["tokens_from_outbound_calls"],
        max_tokens=max_tokens,
        current_id=current_id,
    )


def _run_stage(stage: str, samples: list[dict[str, Any]], client: CachedLLMClient, cap: CappedProviderWrapper | None, args: argparse.Namespace, reporter: ProgressReporter) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_queries = sum(len(_queries_for_sample(sample)) for sample in samples)
    done_queries = 0
    for scenario_index, sample in enumerate(samples, start=1):
        uid = str(sample.get("sample_id") or sample.get("uid") or scenario_index)
        for query_key, dimension, query in _queries_for_sample(sample):
            trace = client.generate(
                prompt=_prompt(stage, sample, query, dimension),
                model_id=args.model,
                provider=args.provider,
                prompt_template_hash=_prompt_hash(stage),
                response_schema_version="stale_development_v1",
                parser_version="plain_answer_v1",
                temperature=0.0,
                metadata={"stage": stage, "dimension": dimension, "sample_id": uid},
            )
            answer = trace.response or ""
            expected = _answer_key(sample, dimension)
            rows.append(
                {
                    "sample_id": uid,
                    "query_id": f"{uid}:{query_key}",
                    "dimension": dimension,
                    "stage": stage,
                    "status": trace.status,
                    "answer": answer,
                    "expected_answer_checksum": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
                    "score": _score(answer, expected),
                    "trace_id": trace.call_id,
                    "error": trace.error_message,
                }
            )
            done_queries += 1
        reporter.update(
            _snapshot(
                f"Stage {stage}",
                scenario_index,
                len(samples),
                done_queries,
                total_queries,
                client,
                cap,
                args.max_calls,
                args.max_tokens,
                uid,
                args.mode == "live-dev",
            )
        )
    if samples:
        reporter.phase(f"Stage {stage} complete.")
        reporter.update(
            _snapshot(
                f"Stage {stage}",
                len(samples),
                len(samples),
                done_queries,
                total_queries,
                client,
                cap,
                args.max_calls,
                args.max_tokens,
                str(samples[-1].get("sample_id") or samples[-1].get("uid") or len(samples)),
                args.mode == "live-dev",
            )
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {}
    for stage in ("A", "B"):
        stage_rows = [row for row in rows if row["stage"] == stage]
        dims: dict[str, dict[str, Any]] = {}
        for _key, dimension in DIMENSIONS:
            dim_rows = [row for row in stage_rows if row["dimension"] == dimension]
            total = len(dim_rows)
            correct = sum(int(row["score"]) for row in dim_rows)
            dims[dimension] = {"correct": correct, "total": total, "score": correct / total if total else None}
        total = len(stage_rows)
        correct = sum(int(row["score"]) for row in stage_rows)
        by_stage[stage] = {
            "queries": total,
            "correct": correct,
            "overall_score": correct / total if total else None,
            "dimensions": dims,
            "execution_errors": sum(1 for row in stage_rows if row["status"] != "success"),
            "parse_errors": 0,
        }
    return by_stage


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only STALE Stage A/B runner.")
    parser.add_argument("--mode", choices=("replay", "live-dev"), default="replay")
    parser.add_argument("--reference-root", default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--limit-scenarios", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/stale_development_eval")
    parser.add_argument("--provider", default=os.getenv("RETRACE_LIVE_PROVIDER", DEFAULT_PROVIDER))
    parser.add_argument("--model", default=os.getenv("RETRACE_LIVE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-calls", type=int, default=int(os.getenv("RETRACE_LIVE_MAX_CALLS", "1000")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("RETRACE_LIVE_MAX_TOKENS", "2000000")))
    parser.add_argument("--live-approved", action="store_true")
    parser.add_argument("--progress-mode", choices=("auto", "bar", "line", "off"), default="auto")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.mode == "live-dev" and not args.live_approved:
        raise SystemExit("Refusing live execution without --live-approved.")
    if args.mode == "live-dev" and ((output_dir / "stale_development_report.json").exists() or (output_dir / "stale_development_manifest.json").exists()):
        raise SystemExit("Refusing to overwrite existing live output directory; choose a fresh output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = StaleAdapter(args.reference_root)
    main_files = adapter.discover_main_files()
    if not main_files:
        raise SystemExit(f"No STALE MAIN files found under {args.reference_root}")
    source_file = main_files[0]
    samples = adapter.load_records(source_file)[: max(args.limit_scenarios, 0)]
    if not samples:
        raise SystemExit(f"No STALE samples loaded from {source_file}")

    total_queries = sum(len(_queries_for_sample(sample)) for sample in samples)
    log_file = args.log_file or None
    reporter = ProgressReporter(mode=args.progress_mode, every=args.progress_every, log_file=log_file)
    reporter.plan(
        f"STALE development-only run | scenarios={len(samples)} | queries={total_queries} per stage | "
        f"stages=A,B | provider={args.provider} | model={args.model} | max_calls={args.max_calls} | max_tokens={args.max_tokens}"
    )

    cap: CappedProviderWrapper | None = None
    if args.mode == "live-dev":
        provider = CappedProviderWrapper(HTTPLLMProvider(), args.max_calls, args.max_tokens)
        cap = provider
    else:
        provider = MockLLMProvider(default_response="mock development answer")

    cache_path = output_dir / f"stale_dev_{uuid.uuid4()}_cache.jsonl"
    client_a = _make_client(args.mode, str(cache_path), provider)
    client_b = _make_client(args.mode, str(cache_path), provider)
    started = time.time()
    reporter.phase("Generating Stage A responses...")
    rows = _run_stage("A", samples, client_a, cap, args, reporter)
    reporter.phase("Generating Stage B responses...")
    rows.extend(_run_stage("B", samples, client_b, cap, args, reporter))
    reporter.phase("Evaluating development responses...")
    aggregate = _aggregate(rows)
    cost = {
        "stage_a": _cost_summary(client_a, cap, is_live=args.mode == "live-dev"),
        "stage_b": _cost_summary(client_b, cap, is_live=args.mode == "live-dev"),
    }
    total_outbound = cap.calls_made if cap is not None and args.mode == "live-dev" else 0
    total_outbound_tokens = cap.tokens_used if cap is not None and args.mode == "live-dev" else 0
    report = {
        "_disclaimer": [
            "STALE END-TO-END DEVELOPMENT-SCALE RUN ONLY.",
            "NOT an official final benchmark result.",
            "No Memora run, no Stage C, no training.",
        ],
        "mode": args.mode,
        "provider": args.provider,
        "model": args.model,
        "stage_a_prompt_version": DEFAULT_STAGE_A_PROMPT_VERSION,
        "source_file": str(source_file),
        "source_checksum": compute_file_sha256(str(source_file)),
        "scenario_count": len(samples),
        "query_count_per_stage": total_queries,
        "aggregate": aggregate,
        "cost": cost,
        "cap_usage": {
            "outbound_network_calls": total_outbound,
            "tokens_from_outbound_calls": total_outbound_tokens,
            "max_calls": args.max_calls,
            "max_tokens": args.max_tokens,
        },
        "latency_seconds": round(time.time() - started, 3),
        "rows": rows,
    }
    report_path = output_dir / "stale_development_report.json"
    _write_json(report_path, report)
    manifest = RunManifest(
        config=RunConfiguration(
            run_id=f"stale-dev-{uuid.uuid4()}",
            stage_and_method_name="STALE-development-StageAB",
            provider_name=args.provider,
            model_id=args.model,
            temperature=0.0,
            prompt_hashes={"stage_a": _prompt_hash("A"), "stage_b": _prompt_hash("B")},
            parser_schema_versions={"stale_development": "v1"},
            cache_path=str(cache_path),
            dataset_checksum=compute_file_sha256(str(source_file)),
            comparison_regime="development-only-shared-stale-input",
            metadata={
                "source_file": str(source_file),
                "scenario_count": len(samples),
                "query_count_per_stage": total_queries,
                "progress_mode": args.progress_mode,
                "stage_a_prompt_version": DEFAULT_STAGE_A_PROMPT_VERSION,
            },
        ),
        aggregate_cost={"cost": cost, "cap_usage": report["cap_usage"]},
        instance_count=len(samples),
        output_path=str(report_path),
        metadata={"development_only": True},
    )
    manifest_path = output_dir / "stale_development_manifest.json"
    manifest.save(str(manifest_path))
    reporter.phase("Saving manifest and report...")
    reporter.done(f"Report written to {report_path}; manifest written to {manifest_path}")
    reporter.close()


if __name__ == "__main__":
    main()
