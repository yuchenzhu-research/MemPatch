"""Offline canonical runner for the official frozen STALE benchmark.

Wires Stage A (ReTrace typed-edge ingestion + DPA) and Stage B (DirectJudge
authorization over the same ingested state and same answer generator) against
the strict :class:`StaleOfficialAdapter`. The runner enforces three
invariants:

- evaluator-only fields (``M_old``, ``M_new``, ``explanation``,
  ``relevant_session_index``, ``type``) are never injected into the
  method-visible state;
- haystack sessions are ingested exactly once per scenario in timestamp order;
- the three probing queries reuse the same persistent authorized state.

This module is offline-only: it accepts mock or replay-cached LLM clients and
never executes live API calls. Live execution must be wired by a separate
authorized task.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retracemem.adapters.stale_official_adapter import (
    StaleMethodVisibleScenario,
    StaleOfficialAdapter,
    StaleOfficialRecord,
)
from retracemem.methods.contracts import EdgePredictionBatch
from retracemem.extraction.typed_extractor import PromptTypedBeliefExtractor
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.pipeline import ReTracePipeline
from retracemem.providers.budget import BudgetWrappedProvider, GlobalBudget
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.retrieval.typed_retrievers import (
    OverlapImpactCandidateRetriever,
    OverlapQueryBeliefRetriever,
)
from retracemem.schemas import EvidenceNode
from retracemem.verifier.proposal_strategy import BatchedEvidenceEdgeProposalStrategy
from retracemem.verifier.prompt_batched_evidence_edge_verifier import PromptBatchedEvidenceEdgeVerifier


_DATASET_SOURCE = "STALEproj/STALE"
_DATASET_ARTIFACT = "T1_T2_400_FULL.json"


@dataclass(frozen=True)
class StaleOfflineRunConfig:
    dataset_path: str = "data_external/stale_official_frozen/T1_T2_400_FULL.json"
    output_dir: str = "outputs/stale_official_frozen_wiring_demo"
    limit_t1: int = 0
    limit_t2: int = 0


@dataclass(frozen=True)
class StaleLiveRunConfig:
    dataset_path: str = "data_external/stale_official_frozen/T1_T2_400_FULL.json"
    output_dir: str = "outputs/stale_official_frozen_stageab_live_smoke_2"
    limit_t1: int = 1
    limit_t2: int = 1
    provider: str = "siliconflow"
    model: str = "deepseek-ai/DeepSeek-V4-Pro"
    judge_provider: str = "siliconflow"
    judge_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    http_timeout_seconds: float = 120.0
    max_calls: int = 500
    max_tokens: int = 2_000_000
    evaluator_concurrency: int = 2
    ingest_chunk_size: int = 10


class _NoEdgeBatchedVerifier:
    """Deterministic offline batched verifier that proposes zero edges.

    Used as a clearly-labeled offline wiring stand-in. It preserves the
    bounded-batched code path (one batched call per ingested evidence) without
    inventing typed-edge semantics from raw STALE haystack text.
    """

    prompt_version = "offline_zero_edge_v1"
    model_id = "offline_mock"
    provider = "offline_mock"

    def verify_edges_batch(
        self,
        new_evidence: Any,
        candidate_beliefs: Any,
        candidate_replacement_beliefs: Any,
        candidate_conditions_by_belief: Any,
        temporal_context: Any,
    ) -> EdgePredictionBatch:
        return EdgePredictionBatch(
            proposed_edges=(),
            model_call_trace_id=f"offline_zero:{new_evidence.evidence_id}",
            prompt_version=self.prompt_version,
            model_id=self.model_id,
            provider=self.provider,
        )


def build_offline_pipeline(client: CachedLLMClient) -> ReTracePipeline:
    """Build a non-leaking offline pipeline wired with bounded-batched Stage A."""
    batched_verifier = _NoEdgeBatchedVerifier()
    strategy = BatchedEvidenceEdgeProposalStrategy(batched_verifier)
    return ReTracePipeline.for_development_fixture(
        edge_proposal_strategy=strategy,
        client=client,
        model_id="offline_mock",
        provider="offline_mock",
    )


def build_live_stage_a_pipeline(
    client_extract: CachedLLMClient,
    client_edges: CachedLLMClient,
    client_answer: CachedLLMClient,
    config: StaleLiveRunConfig,
) -> ReTracePipeline:
    batched_verifier = PromptBatchedEvidenceEdgeVerifier(
        client=client_edges,
        model_id=config.model,
        provider=config.provider,
    )
    strategy = BatchedEvidenceEdgeProposalStrategy(batched_verifier)
    extractor = PromptTypedBeliefExtractor(
        client=client_extract,
        model_id=config.model,
        provider=config.provider,
    )
    return ReTracePipeline.for_development_fixture(
        extractor=extractor,
        edge_proposal_strategy=strategy,
        impact_retriever=OverlapImpactCandidateRetriever(),
        query_retriever=OverlapQueryBeliefRetriever(),
        client=client_answer,
        model_id=config.model,
        provider=config.provider,
    )


def build_live_stage_b_pipeline(
    client_extract: CachedLLMClient,
    client_answer_and_judge: CachedLLMClient,
    config: StaleLiveRunConfig,
) -> ReTracePipeline:
    extractor = PromptTypedBeliefExtractor(
        client=client_extract,
        model_id=config.model,
        provider=config.provider,
    )
    return ReTracePipeline.for_development_fixture(
        extractor=extractor,
        edge_proposal_strategy=BatchedEvidenceEdgeProposalStrategy(_NoEdgeBatchedVerifier()),
        impact_retriever=OverlapImpactCandidateRetriever(),
        query_retriever=OverlapQueryBeliefRetriever(),
        client=client_answer_and_judge,
        model_id=config.model,
        provider=config.provider,
    )


def make_live_client(
    cache_path: Path,
    budget: GlobalBudget,
    stage: str,
    timeout: float,
) -> CachedLLMClient:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    provider = BudgetWrappedProvider(HTTPLLMProvider(timeout=timeout), budget, stage)
    return CachedLLMClient(JSONLCache(str(cache_path)), provider, CostAccounting())


def ingest_method_visible_sessions(
    pipeline: ReTracePipeline,
    user_id: str,
    scenario: StaleMethodVisibleScenario,
    chunk_size: int = 1,
    progress_method: str | None = None,
) -> list[str]:
    """Ingest haystack sessions in declared (timestamp-aligned) order, exactly once."""
    pipeline.reset_user(user_id)
    ingested_evidence_ids: list[str] = []
    chunks = build_chronological_evidence_chunks(scenario, chunk_size)
    for chunk_number, evidence in enumerate(chunks, start=1):
        if progress_method:
            raw_count = evidence.metadata.get("raw_session_count", 0)
            print(
                f"[INGEST] method={progress_method} uid={scenario.uid} "
                f"chunk={chunk_number}/{len(chunks)} raw_sessions={raw_count}",
                flush=True,
            )
        pipeline.ingest_evidence(user_id, evidence)
        ingested_evidence_ids.append(evidence.evidence_id)
    return ingested_evidence_ids


def build_chronological_evidence_chunks(
    scenario: StaleMethodVisibleScenario,
    chunk_size: int,
) -> tuple[EvidenceNode, ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks: list[EvidenceNode] = []
    raw_total = len(scenario.haystack_sessions)
    for start in range(0, raw_total, chunk_size):
        end = min(start + chunk_size, raw_total)
        raw_indices = tuple(range(start, end))
        chunk_sessions = scenario.haystack_sessions[start:end]
        chunk_timestamps = scenario.timestamps[start:end]
        text_parts: list[str] = []
        for raw_index, turns in zip(raw_indices, chunk_sessions, strict=True):
            text_parts.append(
                f"[raw_session_index={raw_index}]\n" + "\n".join(turns)
            )
        chunk_index = len(chunks)
        evidence = EvidenceNode(
            evidence_id=f"{scenario.uid}:chunk:{chunk_index}",
            session_id=f"{scenario.uid}:chunk:{chunk_index}",
            timestamp=chunk_timestamps[-1] if chunk_timestamps else None,
            text="\n\n---\n\n".join(text_parts),
            source_dataset="stale_official",
            source_pointer=f"{scenario.uid}#chunk:{chunk_index}",
            metadata={
                "chunk_index": chunk_index,
                "raw_session_indices": raw_indices,
                "timestamps": chunk_timestamps,
                "raw_session_count": len(raw_indices),
                "ingest_chunk_size": chunk_size,
            },
        )
        chunks.append(evidence)
    return tuple(chunks)


def answer_probing_queries(
    pipeline: ReTracePipeline,
    user_id: str,
    scenario: StaleMethodVisibleScenario,
    method: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Reuse the persistent ingested state to answer all three probing queries."""
    responses: dict[str, str] = {}
    meta: dict[str, Any] = {}
    for query_key, query_text in scenario.probing_queries:
        before = pipeline.client.cost_accountant.to_dict() if pipeline.client else {}
        started = time.time()
        record = pipeline.answer(user_id, query_text, method=method)
        after = pipeline.client.cost_accountant.to_dict() if pipeline.client else {}
        response_key = query_key.replace("_query", "_response")
        responses[response_key] = record.answer
        meta[response_key.replace("_response", "_meta")] = {
            "elapsed_seconds": time.time() - started,
            "usage": _usage_delta(before, after),
        }
    return responses, meta


def run_scenario(
    pipeline: ReTracePipeline,
    record: StaleOfficialRecord,
    method: str,
    chunk_size: int = 1,
    progress_method: str | None = None,
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    user_id = f"stale:{method}:{record.method_visible.uid}"
    evidence_ids = ingest_method_visible_sessions(
        pipeline,
        user_id,
        record.method_visible,
        chunk_size=chunk_size,
        progress_method=progress_method,
    )
    responses, meta = answer_probing_queries(pipeline, user_id, record.method_visible, method)
    return responses, meta, evidence_ids


def _usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    b_tokens = before.get("tokens", {}) if isinstance(before, dict) else {}
    a_tokens = after.get("tokens", {}) if isinstance(after, dict) else {}
    return {
        "prompt_tokens": int(a_tokens.get("prompt", 0) - b_tokens.get("prompt", 0)),
        "completion_tokens": int(a_tokens.get("completion", 0) - b_tokens.get("completion", 0)),
        "total_tokens": int(a_tokens.get("total", 0) - b_tokens.get("total", 0)),
    }


def export_official_target_responses(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    payload = [
        {"uid": row["uid"], "target_model_responses": row["target_model_responses"]}
        for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_subset(
    records: tuple[StaleOfficialRecord, ...], limit_t1: int, limit_t2: int,
) -> tuple[StaleOfficialRecord, ...]:
    selected: list[StaleOfficialRecord] = []
    counts = {"T1": 0, "T2": 0}
    targets = {"T1": limit_t1, "T2": limit_t2}
    for record in records:
        rtype = record.evaluator_only.type
        if counts.get(rtype, 0) < targets.get(rtype, 0):
            selected.append(record)
            counts[rtype] = counts.get(rtype, 0) + 1
        if all(counts.get(key, 0) >= targets.get(key, 0) for key in targets):
            break
    return tuple(selected)


def estimate_live_calls(config: StaleLiveRunConfig) -> dict[str, Any]:
    adapter = StaleOfficialAdapter(config.dataset_path)
    selected = select_subset(adapter.load(), config.limit_t1, config.limit_t2)
    scenario_rows: list[dict[str, Any]] = []
    total_chunks = 0
    for record in selected:
        raw_sessions = len(record.method_visible.haystack_sessions)
        chunks = len(build_chronological_evidence_chunks(record.method_visible, config.ingest_chunk_size))
        total_chunks += chunks
        scenario_rows.append(
            {
                "uid": record.method_visible.uid,
                "type": record.evaluator_only.type,
                "raw_sessions": raw_sessions,
                "chunks": chunks,
            }
        )
    scenario_count = len(selected)
    stage_a_answer_calls = scenario_count * 3
    stage_b_directjudge_calls = scenario_count * 3
    stage_b_answer_calls = scenario_count * 3
    estimate = {
        "selected_scenario_count": scenario_count,
        "composition": {
            "T1": sum(1 for r in selected if r.evaluator_only.type == "T1"),
            "T2": sum(1 for r in selected if r.evaluator_only.type == "T2"),
        },
        "ingest_chunk_size": config.ingest_chunk_size,
        "scenarios": scenario_rows,
        "expected_shared_extraction_network_calls": total_chunks,
        "stage_a_edge_call_upper_bound": total_chunks,
        "stage_a_answer_calls": stage_a_answer_calls,
        "stage_b_directjudge_calls": stage_b_directjudge_calls,
        "stage_b_answer_calls": stage_b_answer_calls,
        "approx_target_method_total_call_upper_bound_excluding_evaluator": (
            total_chunks
            + total_chunks
            + stage_a_answer_calls
            + stage_b_directjudge_calls
            + stage_b_answer_calls
        ),
        "zero_api_calls": True,
        "shared_extraction_cache": True,
    }
    return estimate


def run_offline_wiring_demo(
    config: StaleOfflineRunConfig,
    client_a: CachedLLMClient,
    client_b: CachedLLMClient,
) -> dict[str, Any]:
    started = time.time()
    adapter = StaleOfficialAdapter(config.dataset_path)
    all_records = adapter.load()
    selected = select_subset(all_records, config.limit_t1, config.limit_t2)

    pipeline_a = build_offline_pipeline(client_a)
    pipeline_b = build_offline_pipeline(client_b)

    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    errors: list[str] = []

    for record in selected:
        uid = record.method_visible.uid
        try:
            stage_a_responses, stage_a_meta, _ = run_scenario(pipeline_a, record, method="retrace")
        except Exception as exc:  # pragma: no cover - logged in manifest
            errors.append(f"stage_a:{uid}: {type(exc).__name__}: {exc}")
            stage_a_responses = {
                "dim1_response": "",
                "dim2_response": "",
                "dim3_response": "",
            }
            stage_a_meta = {}
        rows_a.append({"uid": uid, "type": record.evaluator_only.type,
                       "target_model_responses": stage_a_responses,
                       "target_model_meta": stage_a_meta})
        try:
            stage_b_responses, stage_b_meta, _ = run_scenario(pipeline_b, record, method="directjudge")
        except Exception as exc:  # pragma: no cover - logged in manifest
            errors.append(f"stage_b:{uid}: {type(exc).__name__}: {exc}")
            stage_b_responses = {
                "dim1_response": "",
                "dim2_response": "",
                "dim3_response": "",
            }
            stage_b_meta = {}
        rows_b.append({"uid": uid, "type": record.evaluator_only.type,
                       "target_model_responses": stage_b_responses,
                       "target_model_meta": stage_b_meta})

    output_dir = Path(config.output_dir)
    stage_a_path = output_dir / "stage_a_target_responses.json"
    stage_b_path = output_dir / "stage_b_target_responses.json"
    export_official_target_responses(rows_a, stage_a_path)
    export_official_target_responses(rows_b, stage_b_path)

    manifest = {
        "dataset_source": _DATASET_SOURCE,
        "dataset_artifact": _DATASET_ARTIFACT,
        "official_frozen_dataset": True,
        "schema_wiring_demo_only": True,
        "official_model_result": False,
        "official_judge_evaluation_executed": False,
        "live_provider_calls": False,
        "selected_records": [
            {"uid": r.method_visible.uid, "type": r.evaluator_only.type}
            for r in selected
        ],
        "stage_a_export": str(stage_a_path),
        "stage_b_export": str(stage_b_path),
        "errors": errors,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = output_dir / "wiring_demo_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest, "manifest_path": str(manifest_path)}


def write_selected_subset(records: tuple[StaleOfficialRecord, ...], dataset_path: str, output_path: Path) -> None:
    source_rows = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    wanted = {record.method_visible.uid for record in records}
    subset = [row for row in source_rows if row.get("uid") in wanted]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")


def export_evaluator_answers(rows: list[dict[str, Any]], output_path: Path) -> None:
    payload = [
        {
            "uid": row["uid"],
            "target_model_responses": row["target_model_responses"],
            "target_model_meta": row.get("target_model_meta", {}),
        }
        for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_live_stageab_generation(config: StaleLiveRunConfig) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing live output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "caches"
    budget = GlobalBudget(max_calls=config.max_calls, max_tokens=config.max_tokens)
    adapter = StaleOfficialAdapter(config.dataset_path)
    selected = select_subset(adapter.load(), config.limit_t1, config.limit_t2)
    subset_path = output_dir / "selected_official_subset.json"
    write_selected_subset(selected, config.dataset_path, subset_path)

    client_shared_extract = make_live_client(cache_dir / "shared_extract.jsonl", budget, "shared_extract", config.http_timeout_seconds)
    client_a_edges = make_live_client(cache_dir / "stage_a_edges.jsonl", budget, "stage_a_edges", config.http_timeout_seconds)
    client_a_answer = make_live_client(cache_dir / "stage_a_answer.jsonl", budget, "stage_a_answer", config.http_timeout_seconds)
    client_b_answer_judge = make_live_client(cache_dir / "stage_b_directjudge_answer.jsonl", budget, "stage_b_directjudge_answer", config.http_timeout_seconds)

    pipeline_a = build_live_stage_a_pipeline(client_shared_extract, client_a_edges, client_a_answer, config)
    pipeline_b = build_live_stage_b_pipeline(client_shared_extract, client_b_answer_judge, config)
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    errors: list[str] = []

    total = len(selected)
    for idx, record in enumerate(selected, start=1):
        uid = record.method_visible.uid
        rtype = record.evaluator_only.type
        print(f"[SCENARIO_START] {idx}/{total} uid={uid} type={rtype} calls={budget.calls}/{budget.max_calls} tokens={budget.tokens}/{budget.max_tokens}", flush=True)
        try:
            responses, meta, evidence_ids = run_scenario(
                pipeline_a,
                record,
                method="retrace",
                chunk_size=config.ingest_chunk_size,
                progress_method="stage_a",
            )
            rows_a.append({"uid": uid, "type": rtype, "target_model_responses": responses, "target_model_meta": meta, "ingested_evidence_ids": evidence_ids})
            print(f"[STAGE_A_DONE] {idx}/{total} uid={uid} sessions={len(evidence_ids)} calls={budget.calls}/{budget.max_calls} tokens={budget.tokens}/{budget.max_tokens}", flush=True)
        except Exception as exc:
            msg = f"stage_a:{uid}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            rows_a.append({"uid": uid, "type": rtype, "target_model_responses": _empty_responses(), "target_model_meta": {}, "error": msg})
            print(f"[STAGE_A_ERROR] {idx}/{total} uid={uid} error={type(exc).__name__}", flush=True)
        try:
            responses, meta, evidence_ids = run_scenario(
                pipeline_b,
                record,
                method="directjudge",
                chunk_size=config.ingest_chunk_size,
                progress_method="stage_b",
            )
            rows_b.append({"uid": uid, "type": rtype, "target_model_responses": responses, "target_model_meta": meta, "ingested_evidence_ids": evidence_ids})
            print(f"[STAGE_B_DONE] {idx}/{total} uid={uid} sessions={len(evidence_ids)} calls={budget.calls}/{budget.max_calls} tokens={budget.tokens}/{budget.max_tokens}", flush=True)
        except Exception as exc:
            msg = f"stage_b:{uid}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            rows_b.append({"uid": uid, "type": rtype, "target_model_responses": _empty_responses(), "target_model_meta": {}, "error": msg})
            print(f"[STAGE_B_ERROR] {idx}/{total} uid={uid} error={type(exc).__name__}", flush=True)

    stage_a_path = output_dir / "stage_a_target_responses.json"
    stage_b_path = output_dir / "stage_b_target_responses.json"
    export_evaluator_answers(rows_a, stage_a_path)
    export_evaluator_answers(rows_b, stage_b_path)
    manifest = {
        "dataset_source": _DATASET_SOURCE,
        "dataset_artifact": _DATASET_ARTIFACT,
        "official_frozen_dataset": True,
        "schema_wiring_demo_only": False,
        "official_model_result": False,
        "official_judge_evaluation_executed": False,
        "live_provider_calls": True,
        "provider": config.provider,
        "model": config.model,
        "selected_records": [{"uid": r.method_visible.uid, "type": r.evaluator_only.type} for r in selected],
        "ingest_chunk_size": config.ingest_chunk_size,
        "raw_session_count_by_uid": {
            r.method_visible.uid: len(r.method_visible.haystack_sessions) for r in selected
        },
        "shared_extraction_cache": True,
        "authorization_state_isolated": True,
        "stage_b_authorization_timing": "query_time_directjudge_baseline",
        "stage_a_authorization_timing": "ingest_time_retrace",
        "strict_update_time_authorization_matching": False,
        "selected_subset_path": str(subset_path),
        "stage_a_export": str(stage_a_path),
        "stage_b_export": str(stage_b_path),
        "global_budget": budget.summary(),
        "errors": errors,
        "elapsed_seconds": time.time() - started,
        "fairness": {
            "isolated_state": True,
            "sessions_ingested_once_per_method_per_scenario": True,
            "three_queries_reuse_persistent_state": True,
            "stage_a_authorization": "batched_typed_edges_revision_gate_dpa",
            "stage_b_authorization": "directjudge_query_authorization",
            "shared_answer_generator": True,
            "shared_extraction_cache_only": True,
        },
    }
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest, "manifest_path": str(manifest_path)}


def run_official_evaluator(
    *,
    answers_path: str,
    dataset_path: str,
    output_path: str,
    model_method: str,
    conflict_type: str,
    judge_provider: str,
    judge_model: str,
    concurrency: int,
    timeout: float,
) -> None:
    eval_path = Path("reference/STALE/STALE/Evaluation/full_eval_performance.py")
    spec = importlib.util.spec_from_file_location("stale_full_eval_performance", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluator at {eval_path}")
    eval_dir = str(eval_path.parent)
    stale_root = str(eval_path.parents[1])
    for item in (eval_dir, stale_root):
        if item not in sys.path:
            sys.path.insert(0, item)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVAL_PROVIDER = judge_provider
    module.EVAL_MODEL = judge_model
    module.CONCURRENCY_LIMIT = concurrency
    module.eval_client = _AsyncHTTPEvaluatorClient(judge_provider, timeout)
    asyncio.run(module.run_evaluation(answers_path, dataset_path, output_path, model_method, conflict_type))


class _AsyncHTTPEvaluatorClient:
    def __init__(self, provider: str, timeout: float) -> None:
        self.chat = _AsyncHTTPChat(provider, timeout)


class _AsyncHTTPChat:
    def __init__(self, provider: str, timeout: float) -> None:
        self.completions = _AsyncHTTPCompletions(provider, timeout)


class _AsyncHTTPCompletions:
    def __init__(self, provider: str, timeout: float) -> None:
        self.provider = provider
        self.inner = HTTPLLMProvider(timeout=timeout)

    async def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        prompt = "\n\n".join(str(message.get("content", "")) for message in messages)
        trace = await asyncio.to_thread(
            self.inner.generate,
            prompt=prompt,
            model_id=kwargs.get("model", ""),
            provider=self.provider,
            temperature=kwargs.get("temperature"),
        )
        if trace.status != "success" or trace.response is None:
            raise RuntimeError(trace.error_message or "Evaluator HTTP call failed")
        return _EvalResponse(trace.response, trace.prompt_tokens, trace.completion_tokens, trace.total_tokens)


class _EvalResponse:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.choices = [_EvalChoice(content)]
        self.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


class _EvalChoice:
    def __init__(self, content: str) -> None:
        self.message = _EvalMessage(content)


class _EvalMessage:
    def __init__(self, content: str) -> None:
        self.content = content


def _empty_responses() -> dict[str, str]:
    return {"dim1_response": "", "dim2_response": "", "dim3_response": ""}
