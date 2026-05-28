"""ControlledReTraceLLM: Stage A primary-controlled attribution runner.

Consumes a fixed ``SharedCandidateView`` and returns authorization
decisions using the full typed DPA pipeline without running extraction,
induction, or retrieval.

Execution pipeline:
    SharedCandidateView
    → PromptEvidenceEdgeVerifier (evidence-edge prediction only)
    → isolated typed graph (fresh EpisodeLedger + BeliefStore)
    → RevisionGate (structural admission)
    → DefeatPathAuthorizationAlgorithm
    → ControlledMethodResult
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
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier


def _cost_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute per-instance cost by subtracting cumulative snapshots."""
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
        if view.new_evidence is None:
            raise ValueError(
                "ControlledReTraceLLM requires SharedCandidateView.new_evidence"
            )

        cost_before = self.client.cost_accountant.to_dict()

        # Build isolated typed graph
        ledger = EpisodeLedger()
        store = BeliefStore()
        gate = RevisionGate()

        # 1. Append all evidence to the ledger
        for ev in view.evidence_context:
            ledger.append(ev)

        # 2. Add all candidate beliefs and replacement beliefs to the store
        for b in view.candidate_beliefs:
            store.add_belief(b)
        for b in view.candidate_replacement_beliefs:
            if not store.has_belief(b.belief_id):
                store.add_belief(b)

        # 3. Add all conditions to the store
        for _bid, conds in view.candidate_conditions_by_belief:
            for c in conds:
                if not store.has_condition(c.condition_id):
                    store.add_condition(c)

        # 4. Admit supplied dependency edges through the gate
        for _bid, deps in view.dependency_edges_by_belief:
            for dep in deps:
                decision = gate.admit_dependency_edge(dep, store)
                if decision.admitted:
                    store.add_dependency_edge(dep)

        # 5. For each candidate belief, invoke edge verifier and admit results
        all_trace_ids: list[str] = []
        for belief in view.candidate_beliefs:
            belief_conditions = tuple(
                c
                for bid, conds in view.candidate_conditions_by_belief
                if bid == belief.belief_id
                for c in conds
            )
            proposed_edges = self.edge_verifier.verify_edges(
                new_evidence=view.new_evidence,
                candidate_belief=belief,
                candidate_replacement_beliefs=view.candidate_replacement_beliefs,
                candidate_conditions=belief_conditions,
                temporal_context=view.evidence_context,
            )
            for edge in proposed_edges:
                decision = gate.admit_evidence_edge(edge, store)
                if decision.admitted:
                    store.add_evidence_edge(edge)
                if edge.model_call_trace_id and edge.model_call_trace_id not in all_trace_ids:
                    all_trace_ids.append(edge.model_call_trace_id)

        # 6. Run DPA for each candidate belief
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

        # 7. Compute per-instance cost
        cost_after = self.client.cost_accountant.to_dict()
        instance_cost = _cost_delta(cost_before, cost_after)

        return ControlledMethodResult(
            method_name="retrace_llm_controlled",
            instance_id=view.instance_id,
            query_id=view.query_id,
            authorized_belief_ids=tuple(authorized_ids),
            excluded_belief_ids=tuple(excluded_ids),
            model_call_trace_ids=tuple(all_trace_ids),
            cost=instance_cost,
            provenance={
                "view_fingerprint": view.view_fingerprint,
                "fine_grained_statuses": fine_grained,
                "defeat_paths": defeat_paths,
            },
        )
