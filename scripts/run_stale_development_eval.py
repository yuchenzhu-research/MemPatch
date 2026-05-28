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
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.providers.http_provider import HTTPLLMProvider

DIMENSIONS = (("dim1_query", "SR"), ("dim2_query", "PR"), ("dim3_query", "IPA"))


def text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text_of(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {text_of(val)}" for key, val in value.items())
    return str(value)


def queries(sample: dict[str, Any]) -> list[tuple[str, str, str]]:
    probing = sample.get("probing_queries") if isinstance(sample.get("probing_queries"), dict) else {}
    found = []
    for key, dim in DIMENSIONS:
        query = str(probing.get(key) or sample.get(key) or "").strip()
        if query:
            found.append((key, dim, query))
    return found


def expected(sample: dict[str, Any], dim: str) -> str:
    if dim == "PR":
        return str(sample.get("M_old") or sample.get("old_memory") or "").strip()
    return str(sample.get("M_new") or sample.get("new_memory") or "").strip()


def prompt(stage: str, sample: dict[str, Any], query: str, dim: str) -> str:
    sessions = "\n\n".join(text_of(item) for item in sample.get("haystack_session", []) if text_of(item).strip())
    method = "Stage A ReTrace effect-triggered authorization" if stage == "A" else "Stage B DirectJudge baseline"
    return (
        f"STALE development adapter run. Method: {method}. Dimension: {dim}.\n"
        f"Use only the session evidence below. Do not mention hidden fields.\n"
        f"Sessions:\n{sessions}\n\nQuery: {query}\nConcise answer:"
    )


def score(answer: str, gold: str) -> int:
    a = " ".join(answer.lower().split())
    g = " ".join(gold.lower().split())
    return int(bool(g) and (g in a or a in g))


def make_client(mode: str, cache_path: str) -> CachedLLMClient:
    provider = HTTPLLMProvider() if mode == "live-dev" else MockLLMProvider(default_response="mock answer")
    return CachedLLMClient(JSONLCache(cache_path), provider, CostAccounting())


def cost(client: CachedLLMClient) -> dict[str, Any]:
    data = client.cost_accountant.to_dict()
    return {
        "semantic_invocations": data.get("calls", {}).get("total", 0),
        "cache_hits": data.get("cache_hits", 0),
        "cache_misses": data.get("cache_misses", 0),
        "tokens": data.get("tokens", {}),
        "latency_ms": round(data.get("latency_ms", 0.0), 2),
        "errors": data.get("error_counts", {}),
    }


def print_progress(stage: str, done_scenarios: int, total_scenarios: int, done_queries: int, total_queries: int, client: CachedLLMClient, start: float, uid: str, max_calls: int, max_tokens: int) -> None:
    c = cost(client)
    elapsed = time.time() - start
    print(
        f"[{stage}] scenarios {done_scenarios}/{total_scenarios} | queries {done_queries}/{total_queries} | "
        f"semantic {c['semantic_invocations']} | outbound {c['cache_misses']}/{max_calls} | "
        f"tokens {c['tokens'].get('total', 0)}/{max_tokens} | elapsed {elapsed:.1f}s | uid={str(uid)[:48]}",
        flush=True,
    )


def run_stage(stage: str, samples: list[dict[str, Any]], client: CachedLLMClient, args: argparse.Namespace, start: float) -> list[dict[str, Any]]:
    rows = []
    total_queries = sum(len(queries(sample)) for sample in samples)
    done_queries = 0
    for index, sample in enumerate(samples, start=1):
        uid = str(sample.get("uid") or sample.get("sample_id") or index)
        for key, dim, query in queries(sample):
            trace = client.generate(
                prompt=prompt(stage, sample, query, dim),
                model_id=args.model,
                provider=args.provider,
                temperature=0.0,
                response_schema_version="stale_adapter_v1",
                parser_version="plain_text_v1",
                prompt_template_hash=hashlib.sha256(stage.encode()).hexdigest(),
                metadata={"stage": stage, "dimension": dim, "uid": uid},
            )
            answer = trace.response or ""
            gold = expected(sample, dim)
            rows.append({"uid": uid, "query_id": f"{uid}:{key}", "dimension": dim, "stage": stage, "status": trace.status, "score": score(answer, gold), "answer": answer, "error": trace.error_message})
            done_queries += 1
        print_progress(f"Stage {stage}", index, len(samples), done_queries, total_queries, client, start, uid, args.max_calls, args.max_tokens)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for stage in ("A", "B"):
        stage_rows = [row for row in rows if row["stage"] == stage]
        dims = {}
        for _key, dim in DIMENSIONS:
            dim_rows = [row for row in stage_rows if row["dimension"] == dim]
            total = len(dim_rows)
            correct = sum(row["score"] for row in dim_rows)
            dims[dim] = {"correct": correct, "total": total, "score": correct / total if total else None}
        total = len(stage_rows)
        correct = sum(row["score"] for row in stage_rows)
        result[stage] = {"correct": correct, "total": total, "score": correct / total if total else None, "dimensions": dims, "errors": sum(1 for row in stage_rows if row["status"] != "success")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "live-dev"), default="replay")
    parser.add_argument("--live-approved", action="store_true")
    parser.add_argument("--reference-root", default="reference/STALE")
    parser.add_argument("--limit-scenarios", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/stale_gemini35flash_dev_30")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--max-calls", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=2000000)
    parser.add_argument("--progress-mode", default="line")
    args = parser.parse_args()
    if args.mode == "live-dev" and not args.live_approved:
        raise SystemExit("Refusing live execution without --live-approved")
    out = Path(args.output_dir)
    if args.mode == "live-dev" and ((out / "stale_development_report.json").exists() or (out / "stale_development_manifest.json").exists()):
        raise SystemExit("Refusing to overwrite existing live output directory")
    out.mkdir(parents=True, exist_ok=True)
    adapter = StaleAdapter(args.reference_root)
    files = adapter.discover_main_files()
    if not files:
        raise SystemExit("No STALE MAIN files found")
    source = files[0]
    samples = adapter.load_records(source)[: args.limit_scenarios]
    if not samples:
        raise SystemExit("No STALE samples loaded")
    total_queries = sum(len(queries(sample)) for sample in samples)
    print(f"[PLAN] STALE development adapter run | scenarios={len(samples)} | queries={total_queries} per stage | provider={args.provider} | model={args.model}", flush=True)
    start = time.time()
    cache_path = str(out / f"stale_cache_{uuid.uuid4()}.jsonl")
    client_a = make_client(args.mode, cache_path)
    client_b = make_client(args.mode, cache_path)
    rows = run_stage("A", samples, client_a, args, start)
    rows.extend(run_stage("B", samples, client_b, args, start))
    report = {"development_only": True, "official_result": False, "source_path": str(source), "source_checksum": compute_file_sha256(str(source)), "scenarios": len(samples), "queries_per_stage": total_queries, "provider": args.provider, "model": args.model, "aggregate": aggregate(rows), "cost": {"stage_a": cost(client_a), "stage_b": cost(client_b)}, "rows": rows}
    report_path = out / "stale_development_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = RunManifest(config=RunConfiguration(run_id=f"stale-dev-{uuid.uuid4()}", stage_and_method_name="STALE-development-adapter", provider_name=args.provider, model_id=args.model, temperature=0.0, cache_path=cache_path, dataset_checksum=report["source_checksum"], metadata={"development_only": True, "source_path": str(source), "scenarios": len(samples)}), aggregate_cost=report["cost"], instance_count=len(samples), output_path=str(report_path), metadata={"official_result": False})
    manifest_path = out / "stale_development_manifest.json"
    manifest.save(str(manifest_path))
    print(f"[DONE] Report written to {report_path}; manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
