"""ControlledReTraceLLM: Stage A primary-controlled attribution runner.

Consumes a fixed ``SharedCandidateView`` and returns authorization
decisions using the full typed DPA pipeline without running extraction,
induction, or retrieval.
"""
from __future__ import annotations

from typing import Any
from retracemem.authorization import authorize, EvidenceProposalBatch
from retracemem.methods.contracts import ControlledMethodResult, SharedCandidateView
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier


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


class ControlledReTraceLLM:
    """Stage A primary-controlled runner: evidence-edge prediction + DPA.

    Does NOT run extraction, requirement induction, or retrieval.
    Builds an isolated typed graph from the fixed SharedCandidateView inputs.
    """

    def __init__(
        self,
        edge_verifier: PromptEvidenceEdgeVerifier,
        client: CachedLLMClient,
    ) -> None:
        self.edge_verifier = edge_verifier
        self.client = client

    def run(self, view: SharedCandidateView, *, bypass_gate: bool = False) -> ControlledMethodResult:
        """Execute controlled Stage A on the fixed view."""
        cost_before = self.client.cost_accountant.to_dict()

        proposal_batches: list[EvidenceProposalBatch] = []
        for belief in view.candidate_beliefs:
            belief_conditions = tuple(
                c
                for bid, conds in view.candidate_conditions_by_belief
                if bid == belief.belief_id
                for c in conds
            )
            batch = self.edge_verifier.verify_edges_with_trace(
                new_evidence=view.new_evidence,
                candidate_belief=belief,
                candidate_replacement_beliefs=view.candidate_replacement_beliefs,
                candidate_conditions=belief_conditions,
                temporal_context=view.evidence_context,
            )
            proposal_batches.append(
                EvidenceProposalBatch(
                    edges=batch.proposed_edges,
                    model_call_trace_id=batch.model_call_trace_id,
                    source_belief_id=belief.belief_id,
                )
            )

        execution = authorize(
            view,
            tuple(proposal_batches),
            bypass_gate=bypass_gate,
            audit_metadata={
                "prompt_version": self.edge_verifier.prompt_version,
                "prompt_hash": self.edge_verifier._template_hash,
                "model_id": self.edge_verifier.model_id,
                "provider": self.edge_verifier.provider,
                "model_revision_or_api_version": self.edge_verifier.model_revision_or_api_version,
                "bypass_gate": bypass_gate,
            },
        )
        cost_after = self.client.cost_accountant.to_dict()

        method_name = "retrace_llm_controlled_ablation" if bypass_gate else "retrace_llm_controlled"
        return ControlledMethodResult(
            method_name=method_name,
            instance_id=view.instance_id,
            query_id=view.query_id,
            authorized_belief_ids=execution.authorized_belief_ids,
            excluded_belief_ids=execution.excluded_belief_ids,
            model_call_trace_ids=execution.trace.get("model_call_trace_ids", ()),
            cost=cost_delta(cost_before, cost_after),
            provenance=execution.trace,
        )
