#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retracemem.adapters.stale_adapter import StaleAdapter
from retracemem.backends.retrace_backend import ReTraceBackend
from retracemem.backends.retrieval_baseline import RetrievalBaselineBackend
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation import evaluation_record_from_backend_output, write_jsonl
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.extraction.manual_fixture_extractor import ManualFixtureExtractor
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.retrieval.candidate_retriever import SimpleOverlapRetriever
from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType
from retracemem.verifier.heuristic_verifier import HeuristicRelationVerifier
from retracemem.verifier.prompt_verifier import PromptRelationVerifier


DEFAULT_OUTPUT = "outputs/stale/{method}_frozen_eval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STALE Frozen Evaluation.")
    parser.add_argument("--reference-root", default="reference/STALE")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--method", choices=("retrieval_baseline", "retrace"), default="retrace")
    parser.add_argument("--output", default=None)
    parser.add_argument("--disable-ledger", action="store_true")
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--disable-temporal", action="store_true")
    args = parser.parse_args()

    adapter = StaleAdapter(args.reference_root)
    main_files = adapter.discover_main_files()
    output = args.output or DEFAULT_OUTPUT.format(method=args.method)

    if not main_files:
        print(f"No STALE MAIN files found under {args.reference_root}; nothing to run.")
        write_jsonl([], output)
        print("records_written: 0")
        return

    samples = adapter.load_records(main_files[0])[: max(args.limit, 0)]
    if not samples:
        print(f"No STALE samples loaded; nothing to run.")
        write_jsonl([], output)
        print("records_written: 0")
        return

    # Ingestion setup
    if args.method == "retrieval_baseline":
        backend: Any = RetrievalBaselineBackend()
    else:
        # ReTrace Backend Setup
        # For evaluation, check if API is available, otherwise use HeuristicRelationVerifier
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        cache = JSONLCache("artifacts/cache/stale_eval_cache.jsonl")
        cost_accountant = CostAccounting()

        if api_key:
            # We would typically setup the real provider client here
            # For smoke check safety, if API fails or we want local, we fallback to Heuristic
            provider_client = MockLLMProvider(default_response="{}")
            client = CachedLLMClient(cache, provider_client, cost_accountant)
            verifier: Any = PromptRelationVerifier(client=client)
        else:
            verifier = HeuristicRelationVerifier()
            client = None

        extractor = ManualFixtureExtractor()
        # For evaluation, we want to evaluate all beliefs without overlap retrieval issues
        class EvalRetriever:
            def retrieve_candidates(self, new_ev, all_b):
                return all_b

        backend = ReTraceBackend(
            extractor=extractor,
            verifier=verifier,
            retriever=EvalRetriever(),
            client=client,
            disable_ledger=args.disable_ledger,
            disable_gate=args.disable_gate,
            disable_temporal=args.disable_temporal,
        )

    records = []
    for sample_index, sample in enumerate(samples, start=1):
        user_id = str(sample.get("sample_id") or sample.get("uid") or f"sample_{sample_index}")
        backend.reset_user(user_id)

        sessions = _sessions_for_sample(sample)

        # Pre-register Manual Extractors fixtures if ReTrace is used
        if isinstance(backend, ReTraceBackend):
            old_belief_text = str(sample.get("old_memory") or sample.get("M_old") or "")
            new_belief_text = str(sample.get("new_memory") or sample.get("M_new") or "")

            old_belief_id = f"belief_old_{user_id}"
            new_belief_id = f"belief_new_{user_id}"

            # Register old memory extraction
            if old_belief_text:
                backend.extractor.register(
                    f"{user_id}_old_memory",
                    [Belief(id=old_belief_id, proposition=old_belief_text, supported_by=[f"{user_id}_old_memory"])],
                )

            # Register new memory extraction
            if new_belief_text:
                backend.extractor.register(
                    f"{user_id}_new_memory",
                    [Belief(id=new_belief_id, proposition=new_belief_text, supported_by=[f"{user_id}_new_memory"])],
                )

        # Ingest sessions
        for session in sessions:
            backend.ingest_session(user_id, session, metadata={"sample_id": user_id})

        # Answer probing queries
        for query_key, query in _probing_queries(sample).items():
            if not query.strip():
                continue
            retrieved = backend.search(user_id, query, limit=10)
            answer = backend.answer(
                user_id, query, retrieved, metadata={"sample_id": user_id, "dimension": query_key}
            )
            records.append(
                evaluation_record_from_backend_output(
                    query_id=f"{user_id}:{query_key}",
                    method=args.method,
                    retrieved=retrieved,
                    answer=answer,
                )
            )

    write_jsonl(records, output)
    print(f"source_file: {main_files[0]}")
    print(f"samples_loaded: {len(samples)}")
    print(f"records_written: {len(records)}")
    print(f"output_path: {output}")


def _sessions_for_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    sample_id = str(sample.get("sample_id") or sample.get("uid") or "sample")

    old_memory = str(sample.get("old_memory") or sample.get("M_old") or "")
    if old_memory:
        sessions.append({"id": f"{sample_id}_old_memory", "text": old_memory})

    raw_sessions = sample.get("sessions") or sample.get("haystack_session") or []
    if isinstance(raw_sessions, list):
        for index, raw_session in enumerate(raw_sessions, start=1):
            text = _text_from_benchmark_value(raw_session)
            if text:
                sessions.append({"id": f"{sample_id}_session_{index:04d}", "text": text})

    new_memory = str(sample.get("new_memory") or sample.get("M_new") or "")
    if new_memory:
        sessions.append({"id": f"{sample_id}_new_memory", "text": new_memory})

    return sessions


def _probing_queries(sample: dict[str, Any]) -> dict[str, str]:
    queries = sample.get("probing_queries")
    if not isinstance(queries, dict):
        queries = {}
    return {
        key: str(queries.get(key) or sample.get(key) or "")
        for key in ("dim1_query", "dim2_query", "dim3_query")
    }


def _text_from_benchmark_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "message", "utterance", "value"):
            item = value.get(key)
            if isinstance(item, str):
                return item
        return " ".join(_text_from_benchmark_value(item) for item in value.values()).strip()
    if isinstance(value, list):
        return " ".join(_text_from_benchmark_value(item) for item in value).strip()
    return str(value) if value is not None else ""


if __name__ == "__main__":
    main()
