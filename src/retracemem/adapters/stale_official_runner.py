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

import json
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
from retracemem.pipeline import ReTracePipeline
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import EvidenceNode
from retracemem.verifier.proposal_strategy import BatchedEvidenceEdgeProposalStrategy


_DATASET_SOURCE = "STALEproj/STALE"
_DATASET_ARTIFACT = "T1_T2_400_FULL.json"


@dataclass(frozen=True)
class StaleOfflineRunConfig:
    dataset_path: str = "data_external/stale_official_frozen/T1_T2_400_FULL.json"
    output_dir: str = "outputs/stale_official_frozen_wiring_demo"
    limit_t1: int = 0
    limit_t2: int = 0


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


def ingest_method_visible_sessions(
    pipeline: ReTracePipeline,
    user_id: str,
    scenario: StaleMethodVisibleScenario,
) -> list[str]:
    """Ingest haystack sessions in declared (timestamp-aligned) order, exactly once."""
    pipeline.reset_user(user_id)
    ingested_evidence_ids: list[str] = []
    for index, (turns, ts) in enumerate(
        zip(scenario.haystack_sessions, scenario.timestamps, strict=True)
    ):
        evidence_id = f"{scenario.uid}:session:{index}"
        evidence = EvidenceNode(
            evidence_id=evidence_id,
            session_id=f"{scenario.uid}:s{index}",
            timestamp=ts,
            text="\n".join(turns),
            source_dataset="stale_official",
            source_pointer=f"{scenario.uid}#{index}",
        )
        pipeline.ingest_evidence(user_id, evidence)
        ingested_evidence_ids.append(evidence_id)
    return ingested_evidence_ids


def answer_probing_queries(
    pipeline: ReTracePipeline,
    user_id: str,
    scenario: StaleMethodVisibleScenario,
    method: str,
) -> dict[str, str]:
    """Reuse the persistent ingested state to answer all three probing queries."""
    responses: dict[str, str] = {}
    for query_key, query_text in scenario.probing_queries:
        record = pipeline.answer(user_id, query_text, method=method)
        response_key = query_key.replace("_query", "_response")
        responses[response_key] = record.answer
    return responses


def run_scenario(
    pipeline: ReTracePipeline,
    record: StaleOfficialRecord,
    method: str,
) -> dict[str, str]:
    user_id = f"stale:{method}:{record.method_visible.uid}"
    ingest_method_visible_sessions(pipeline, user_id, record.method_visible)
    return answer_probing_queries(pipeline, user_id, record.method_visible, method)


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
            stage_a_responses = run_scenario(pipeline_a, record, method="retrace")
        except Exception as exc:  # pragma: no cover - logged in manifest
            errors.append(f"stage_a:{uid}: {type(exc).__name__}: {exc}")
            stage_a_responses = {
                "dim1_response": "",
                "dim2_response": "",
                "dim3_response": "",
            }
        rows_a.append({"uid": uid, "type": record.evaluator_only.type,
                       "target_model_responses": stage_a_responses})
        try:
            stage_b_responses = run_scenario(pipeline_b, record, method="directjudge")
        except Exception as exc:  # pragma: no cover - logged in manifest
            errors.append(f"stage_b:{uid}: {type(exc).__name__}: {exc}")
            stage_b_responses = {
                "dim1_response": "",
                "dim2_response": "",
                "dim3_response": "",
            }
        rows_b.append({"uid": uid, "type": record.evaluator_only.type,
                       "target_model_responses": stage_b_responses})

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
