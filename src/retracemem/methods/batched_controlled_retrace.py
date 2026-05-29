"""BatchedControlledReTraceLLM: Stage A scalable batched authorization runner.

Uses a single semantic-model call to predict typed edges for all candidate
beliefs, then admits each through RevisionGate and runs DPA identically to
the per-belief ``ControlledReTraceLLM`` reference path.
"""
from __future__ import annotations

from typing import Any
from retracemem.authorization import authorize, EvidenceProposalBatch
from retracemem.methods.contracts import ControlledMethodResult, SharedCandidateView
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.verifier.prompt_batched_evidence_edge_verifier import PromptBatchedEvidenceEdgeVerifier


def cost_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    tokens_before = before.get("tokens", {})
    tokens_after = after.get("tokens", {})
    calls_before = before.get("calls", {})
    calls_after = after.get("calls", {})
    return {
        "latency_ms": after.get("latency_ms", 0.0) - before.get("latency_ms", 0.0),
        "tokens": {
            "prompt": tokens_after.get("prompt", 0) - tokens_before.get("prompt", 0),
            "completion": tokens_after.get("completion", 0) - tokens_before.get("completion", 0),
            "total": tokens_after.get("total", 0) - tokens_before.get("total", 0),
        },
        "calls": {
            key: calls_after.get(key, 0) - calls_before.get(key, 0)
            for key in set(calls_after) | set(calls_before)
        },
        "cache_hits": after.get("cache_hits", 0) - before.get("cache_hits", 0),
        "cache_misses": after.get("cache_misses", 0) - before.get("cache_misses", 0),
    }


class BatchedControlledReTraceLLM:
    """Stage A batched runner: one model call for all beliefs + RevisionGate + DPA.

    Does NOT modify the existing per-belief ControlledReTraceLLM.
    """

    def __init__(
        self,
        edge_verifier: PromptBatchedEvidenceEdgeVerifier,
        client: CachedLLMClient,
    ) -> None:
        self.edge_verifier = edge_verifier
        self.client = client

    def run(self, view: SharedCandidateView) -> ControlledMethodResult:
        """Execute batched Stage A on the fixed view."""
        cost_before = self.client.cost_accountant.to_dict()

        batch = self.edge_verifier.verify_edges_batch(
            new_evidence=view.new_evidence,
            candidate_beliefs=view.candidate_beliefs,
            candidate_replacement_beliefs=view.candidate_replacement_beliefs,
            candidate_conditions_by_belief=view.candidate_conditions_by_belief,
            temporal_context=view.evidence_context,
        )

        execution = authorize(
            view,
            (
                EvidenceProposalBatch(
                    edges=batch.proposed_edges,
                    model_call_trace_id=batch.model_call_trace_id,
                    metadata=batch.metadata,
                ),
            ),
            audit_metadata={
                "execution_mode": "batched_local_edges_v1",
                "prompt_version": self.edge_verifier.prompt_version,
                "prompt_hash": self.edge_verifier._template_hash,
                "model_id": self.edge_verifier.model_id,
                "provider": self.edge_verifier.provider,
                "model_revision_or_api_version": self.edge_verifier.model_revision_or_api_version,
                "batch_candidate_belief_count": len(view.candidate_beliefs),
                "batch_model_call_trace_id": batch.model_call_trace_id,
            },
        )
        cost_after = self.client.cost_accountant.to_dict()

        return ControlledMethodResult(
            method_name="retrace_llm_batched_controlled",
            instance_id=view.instance_id,
            query_id=view.query_id,
            authorized_belief_ids=execution.authorized_belief_ids,
            excluded_belief_ids=execution.excluded_belief_ids,
            model_call_trace_ids=execution.trace.get("model_call_trace_ids", ()),
            cost=cost_delta(cost_before, cost_after),
            provenance=execution.trace,
        )

