from __future__ import annotations

from typing import Any
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
)
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm
from retracemem.extraction.typed_extractor import TypedBeliefExtractor, ManualTypedBeliefExtractor
from retracemem.verifier.contracts import RequirementInducer, EvidenceEdgeVerifier
from retracemem.verifier.requirement_inducer import HeuristicRequirementInducer
from retracemem.verifier.evidence_edge_verifier import HeuristicEvidenceEdgeVerifier
from retracemem.retrieval.typed_retrievers import (
    ImpactCandidateRetriever,
    QueryBeliefRetriever,
    ManualImpactCandidateRetriever,
    ManualQueryBeliefRetriever,
)
from retracemem.tms.gate import RevisionGate
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.generation.basis_builder import BasisBuilder


class ReTraceBackend:
    """End-to-End ReTrace backend utilizing canonical typed components."""

    def __init__(
        self,
        extractor: TypedBeliefExtractor | None = None,
        inducer: RequirementInducer | None = None,
        edge_verifier: EvidenceEdgeVerifier | None = None,
        impact_retriever: ImpactCandidateRetriever | None = None,
        query_retriever: QueryBeliefRetriever | None = None,
        client: CachedLLMClient | None = None,
        model_id: str = "gemini-pro",
        provider: str = "google",
        disable_ledger: bool = False,
        disable_gate: bool = False,
        disable_temporal: bool = False,
    ) -> None:
        self.ledgers: dict[str, EpisodeLedger] = {}
        self.stores: dict[str, BeliefStore] = {}
        self.extractor = extractor or ManualTypedBeliefExtractor()
        self.inducer = inducer or HeuristicRequirementInducer()
        self.edge_verifier = edge_verifier or HeuristicEvidenceEdgeVerifier()
        self.impact_retriever = impact_retriever or ManualImpactCandidateRetriever()
        self.query_retriever = query_retriever or ManualQueryBeliefRetriever()
        self.gate = RevisionGate()
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.disable_ledger = disable_ledger
        self.disable_gate = disable_gate
        self.disable_temporal = disable_temporal

    def reset_user(self, user_id: str) -> None:
        self.ledgers[user_id] = EpisodeLedger()
        self.stores[user_id] = BeliefStore()

    def ingest_session(
        self,
        user_id: str,
        session: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        del metadata
        self._ensure_user(user_id)

        if not session:
            return

        ledger = self.ledgers[user_id]
        ev_id = session.get("id") or session.get("evidence_id") or session.get("session_id") or f"ev-{len(ledger)}"
        session_id = session.get("session_id") or user_id
        timestamp = session.get("timestamp") or None
        text = session.get("text") or session.get("content") or ""
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        source_dataset = session.get("source_dataset") or session.get("source_id") or "ingest"
        source_pointer = session.get("source_pointer") or "ingest_pointer"
        is_raw_source = session.get("is_raw_source", True)
        ev_metadata = session.get("metadata") or {}

        evidence = EvidenceNode(
            evidence_id=ev_id,
            session_id=session_id,
            timestamp=timestamp,
            text=text,
            source_dataset=source_dataset,
            source_pointer=source_pointer,
            is_raw_source=is_raw_source,
            metadata=ev_metadata,
        )

        self.ingest_evidence(user_id, evidence)

    def ingest_evidence(
        self,
        user_id: str,
        evidence: EvidenceNode,
    ) -> list[EvidenceEdge]:
        self._ensure_user(user_id)

        ledger = self.ledgers[user_id]
        store = self.stores[user_id]

        ledger.append(evidence)

        prior_beliefs = tuple(store.all_beliefs())

        # Extract candidates
        raw_new_beliefs = self.extractor.extract(evidence, scope_id=user_id)
        new_beliefs: list[BeliefNode] = []
        for belief in raw_new_beliefs:
            from dataclasses import replace
            meta = dict(belief.metadata) if belief.metadata else {}
            meta["scope_id"] = user_id
            updated_belief = replace(belief, metadata=meta)
            new_beliefs.append(updated_belief)

        for belief in new_beliefs:
            store.add_belief(belief)

        # Induce requirements
        for belief in new_beliefs:
            proposals = self.inducer.induce_requirements(belief, (evidence,))
            for proposal in proposals:
                cond = proposal.condition
                if not store.has_condition(cond.condition_id):
                    store.add_condition(cond)
                if not store.has_dependency_edge(proposal.dependency_edge.edge_id):
                    gate_decision = self.gate.admit_dependency_edge(proposal.dependency_edge, store)
                    if gate_decision.admitted:
                        store.add_dependency_edge(proposal.dependency_edge)

        # Impact candidate retrieval
        impact_candidates = self.impact_retriever.retrieve_impacts(
            new_evidence=evidence,
            prior_beliefs=prior_beliefs,
            store=store,
        )

        temporal_context = tuple(ledger.all())
        admitted_edges: list[EvidenceEdge] = []

        # Verify edges
        for candidate in impact_candidates:
            evidence_edges = self.edge_verifier.verify_edges(
                new_evidence=evidence,
                candidate_belief=candidate.belief,
                candidate_replacement_beliefs=tuple(new_beliefs),
                candidate_conditions=candidate.conditions,
                temporal_context=temporal_context,
            )
            for edge in evidence_edges:
                gate_decision = self.gate.admit_evidence_edge(edge, store)
                if gate_decision.admitted:
                    if not store.has_evidence_edge(edge.edge_id):
                        store.add_evidence_edge(edge)
                        admitted_edges.append(edge)

        return admitted_edges

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del metadata
        self._ensure_user(user_id)

        store = self.stores[user_id]
        ledger = self.ledgers[user_id]

        engine = DefeatPathAuthorizationAlgorithm(store, ledger)
        builder = BasisBuilder(self.query_retriever, engine)
        result = builder.build(
            query=query,
            beliefs=tuple(store.all_beliefs()),
            limit=limit,
        )
        return result["authorized_basis"]

    def answer(
        self,
        user_id: str,
        query: str,
        retrieved: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del user_id, metadata

        if self.client is not None:
            context = "\n".join(f"- {item.get('proposition') or item.get('text', '')}" for item in retrieved)
            prompt = f"Answer the user's query using the authorized beliefs provided below.\n\nAuthorized Beliefs:\n{context}\n\nQuery: {query}\n\nAnswer:"
            try:
                trace = self.client.generate(
                    prompt=prompt,
                    model_id=self.model_id,
                    provider=self.provider,
                    temperature=0.0,
                )
                if trace.status == "success" and trace.response:
                    return trace.response
            except Exception:
                pass

        context = "\n".join(item.get("proposition") or item.get("text", "") for item in retrieved)
        return f"Query: {query}\nAuthorized basis:\n{context}"

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self.ledgers:
            self.reset_user(user_id)
