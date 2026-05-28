"""BatchedControlledReTraceLLM: Stage A scalable batched authorization runner.

Uses a single semantic-model call to predict typed edges for all candidate
beliefs, then admits each through RevisionGate and runs DPA identically to
the per-belief ``ControlledReTraceLLM`` reference path.

Critical invariant: for the same accepted edge set, batched and per-belief
paths yield identical Gate/DPA authorization results.
"""
from __future__ import annotations

from retracemem.methods.authorization_executor import (
    ProposedEvidenceEdges,
    cost_delta,
    execute_authorization,
)
from retracemem.methods.contracts import ControlledMethodResult, SharedCandidateView
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.verifier.prompt_batched_evidence_edge_verifier import PromptBatchedEvidenceEdgeVerifier


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

        execution = execute_authorization(
            view,
            (
                ProposedEvidenceEdges(
                    edges=batch.proposed_edges,
                    model_call_trace_id=batch.model_call_trace_id,
                    metadata=batch.metadata,
                ),
            ),
            base_provenance={
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
            model_call_trace_ids=execution.model_call_trace_ids,
            cost=cost_delta(cost_before, cost_after),
            provenance=execution.provenance,
        )
