"""ControlledReTraceLLM: Stage A primary-controlled attribution runner.

Consumes a fixed ``SharedCandidateView`` and returns authorization
decisions using the full typed DPA pipeline without running extraction,
induction, or retrieval.

Execution pipeline:
    SharedCandidateView
    → PromptEvidenceEdgeVerifier.verify_edges_with_trace (edge prediction)
    → isolated typed graph (fresh EpisodeLedger + BeliefStore)
    → RevisionGate (structural admission)
    → DefeatPathAuthorizationAlgorithm
    → ControlledMethodResult (full provenance)

AB-1A.5 auditability guarantees:
    - Every verifier invocation preserves its model_call_trace_id, including
      zero-edge invocations.
    - Rejected fixed DependencyEdge anchors fail immediately and loudly.
    - Rejected EvidenceEdge proposals are recorded in provenance with their
      gate rejection reason; they do not enter the store.
    - Admitted fixed anchors are recorded in provenance.
    - metadata fields are non-semantic and MUST NOT be consumed.
"""
from __future__ import annotations

from retracemem.methods.authorization_executor import (
    ProposedEvidenceEdges,
    cost_delta,
    execute_authorization,
)
from retracemem.methods.contracts import ControlledMethodResult, SharedCandidateView
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier


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

    def run(self, view: SharedCandidateView) -> ControlledMethodResult:
        """Execute controlled Stage A on the fixed view."""
        cost_before = self.client.cost_accountant.to_dict()

        proposal_batches: list[ProposedEvidenceEdges] = []
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
                ProposedEvidenceEdges(
                    edges=batch.proposed_edges,
                    model_call_trace_id=batch.model_call_trace_id,
                    source_belief_id=belief.belief_id,
                )
            )

        execution = execute_authorization(
            view,
            tuple(proposal_batches),
            base_provenance={
                "prompt_version": self.edge_verifier.prompt_version,
                "prompt_hash": self.edge_verifier._template_hash,
                "model_id": self.edge_verifier.model_id,
                "provider": self.edge_verifier.provider,
                "model_revision_or_api_version": self.edge_verifier.model_revision_or_api_version,
            },
        )
        cost_after = self.client.cost_accountant.to_dict()

        return ControlledMethodResult(
            method_name="retrace_llm_controlled",
            instance_id=view.instance_id,
            query_id=view.query_id,
            authorized_belief_ids=execution.authorized_belief_ids,
            excluded_belief_ids=execution.excluded_belief_ids,
            model_call_trace_ids=execution.model_call_trace_ids,
            cost=cost_delta(cost_before, cost_after),
            provenance=execution.provenance,
        )
