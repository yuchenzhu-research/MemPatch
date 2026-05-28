"""BatchedControlledReTraceLLM: Stage A scalable batched authorization runner.

Uses a single semantic-model call to predict typed edges for all candidate
beliefs, then admits each through RevisionGate and runs DPA identically to
the per-belief ``ControlledReTraceLLM`` reference path.

Critical invariant: for the same accepted edge set, batched and per-belief
paths yield identical Gate/DPA authorization results.
"""
from __future__ import annotations

from typing import Any

from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.methods.contracts import ControlledMethodResult, SharedCandidateView
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import AuthorizationStatus
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm
from retracemem.tms.gate import RevisionGate
from retracemem.verifier.prompt_batched_evidence_edge_verifier import PromptBatchedEvidenceEdgeVerifier


def _cost_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
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
            k: calls_after.get(k, 0) - calls_before.get(k, 0)
            for k in set(calls_after) | set(calls_before)
        },
        "cache_hits": after.get("cache_hits", 0) - before.get("cache_hits", 0),
        "cache_misses": after.get("cache_misses", 0) - before.get("cache_misses", 0),
    }


_STATUS_MAP = {
    AuthorizationStatus.AUTHORIZED: "AUTHORIZED",
    AuthorizationStatus.BLOCKED: "BLOCKED",
    AuthorizationStatus.SUPERSEDED: "SUPERSEDED",
    AuthorizationStatus.UNRESOLVED: "UNRESOLVED",
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

        ledger = EpisodeLedger()
        store = BeliefStore()
        gate = RevisionGate()

        for ev in view.evidence_context:
            ledger.append(ev)

        for b in view.candidate_beliefs:
            store.add_belief(b)
        for b in view.candidate_replacement_beliefs:
            if not store.has_belief(b.belief_id):
                store.add_belief(b)

        for _bid, conds in view.candidate_conditions_by_belief:
            for c in conds:
                if not store.has_condition(c.condition_id):
                    store.add_condition(c)

        admitted_anchors: list[dict[str, Any]] = []
        for _bid, deps in view.dependency_edges_by_belief:
            for dep in deps:
                decision = gate.admit_dependency_edge(dep, store)
                if not decision.admitted:
                    raise ValueError(
                        f"Fixed supplied DependencyEdge '{dep.edge_id}' "
                        f"rejected by RevisionGate: {decision.reason}"
                    )
                store.add_dependency_edge(dep)
                admitted_anchors.append({
                    "edge_id": dep.edge_id,
                    "belief_id": dep.belief_id,
                    "condition_id": dep.condition_id,
                })

        batch = self.edge_verifier.verify_edges_batch(
            new_evidence=view.new_evidence,
            candidate_beliefs=view.candidate_beliefs,
            candidate_replacement_beliefs=view.candidate_replacement_beliefs,
            candidate_conditions_by_belief=view.candidate_conditions_by_belief,
            temporal_context=view.evidence_context,
        )

        edge_proposals: list[dict[str, Any]] = []
        for edge in batch.proposed_edges:
            decision = gate.admit_evidence_edge(edge, store)
            edge_proposals.append({
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type.value,
                "target_id": edge.target_id,
                "admitted": decision.admitted,
                "gate_reason": decision.reason,
                "model_call_trace_id": batch.model_call_trace_id,
            })
            if decision.admitted:
                store.add_evidence_edge(edge)

        dpa = DefeatPathAuthorizationAlgorithm(store, ledger)
        authorized_ids: list[str] = []
        excluded_ids: list[str] = []
        fine_grained: dict[str, str] = {}
        defeat_paths: list[dict[str, Any]] = []

        for belief in view.candidate_beliefs:
            trace = dpa.authorize(
                belief.belief_id,
                as_of_evidence_id=view.new_evidence.evidence_id,
                query_id=view.query_id,
            )
            status_str = _STATUS_MAP.get(trace.status, trace.status.value)
            fine_grained[belief.belief_id] = status_str

            if trace.status == AuthorizationStatus.AUTHORIZED:
                authorized_ids.append(belief.belief_id)
            else:
                excluded_ids.append(belief.belief_id)
                if trace.accepted_defeat_path is not None:
                    defeat_paths.append({
                        "belief_id": belief.belief_id,
                        "path_type": trace.accepted_defeat_path.path_type.value,
                        "path_id": trace.accepted_defeat_path.path_id,
                        "evidence_edge_ids": list(trace.accepted_defeat_path.supporting_evidence_edge_ids),
                        "dependency_edge_ids": list(trace.accepted_defeat_path.supporting_dependency_edge_ids),
                        "replacement_belief_id": trace.accepted_defeat_path.replacement_belief_id,
                    })

        cost_after = self.client.cost_accountant.to_dict()
        instance_cost = _cost_delta(cost_before, cost_after)

        return ControlledMethodResult(
            method_name="retrace_llm_batched_controlled",
            instance_id=view.instance_id,
            query_id=view.query_id,
            authorized_belief_ids=tuple(authorized_ids),
            excluded_belief_ids=tuple(excluded_ids),
            model_call_trace_ids=(batch.model_call_trace_id,),
            cost=instance_cost,
            provenance={
                "execution_mode": "batched_local_edges_v1",
                "view_fingerprint": view.view_fingerprint,
                "fine_grained_statuses": fine_grained,
                "defeat_paths": defeat_paths,
                "admitted_fixed_anchors": admitted_anchors,
                "edge_proposals": edge_proposals,
                "prompt_version": self.edge_verifier.prompt_version,
                "prompt_hash": self.edge_verifier._template_hash,
                "model_id": self.edge_verifier.model_id,
                "provider": self.edge_verifier.provider,
                "model_revision_or_api_version": self.edge_verifier.model_revision_or_api_version,
                "batch_candidate_belief_count": len(view.candidate_beliefs),
                "batch_model_call_trace_id": batch.model_call_trace_id,
            },
        )
