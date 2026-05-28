from __future__ import annotations

from typing import Any
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    EvidenceEdge,
)
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm
from retracemem.extraction.typed_extractor import TypedBeliefExtractor, ManualTypedBeliefExtractor
from retracemem.verifier.contracts import RequirementInducer, EvidenceEdgeVerifier, BatchedEvidenceEdgeVerifier
from retracemem.verifier.proposal_strategy import (
    BatchedEvidenceEdgeProposalStrategy,
    EvidenceEdgeProposalStrategy,
    PerBeliefEvidenceEdgeProposalStrategy,
)
from retracemem.verifier.requirement_inducer import HeuristicRequirementInducer
from retracemem.verifier.evidence_edge_verifier import HeuristicEvidenceEdgeVerifier
from retracemem.retrieval.typed_retrievers import (
    ImpactCandidateRetriever,
    QueryBeliefRetriever,
    ManualImpactCandidateRetriever,
    ManualQueryBeliefRetriever,
)
from retracemem.tms.gate import RevisionGate
from retracemem.generation.basis_builder import BasisBuilder


class ReTraceBackend:
    """End-to-End ReTrace backend utilizing canonical typed components."""

    def __init__(
        self,
        extractor: TypedBeliefExtractor | None = None,
        inducer: RequirementInducer | None = None,
        edge_verifier: EvidenceEdgeVerifier | None = None,
        batched_edge_verifier: BatchedEvidenceEdgeVerifier | None = None,
        edge_proposal_strategy: EvidenceEdgeProposalStrategy | None = None,
        impact_retriever: ImpactCandidateRetriever | None = None,
        query_retriever: QueryBeliefRetriever | None = None,
        max_batch_beliefs: int = 8,
        client: Any | None = None,
        disable_ledger: bool = False,
        disable_gate: bool = False,
        disable_temporal: bool = False,
    ) -> None:
        if client is not None:
            raise ValueError(
                "API-backed answer generation belongs to a later Stage A wrapper, not the Wave 2 typed backend"
            )
        if disable_ledger or disable_gate or disable_temporal:
            raise ValueError("Wave 2 typed backend does not support disable_ledger/disable_gate/disable_temporal ablations")
        required_components = {
            "extractor": extractor,
            "inducer": inducer,
            "edge_proposal_strategy": edge_proposal_strategy or edge_verifier or batched_edge_verifier,
            "impact_retriever": impact_retriever,
            "query_retriever": query_retriever,
        }
        missing = [name for name, value in required_components.items() if value is None]
        if missing:
            raise ValueError(
                "ReTraceBackend requires explicit typed components; use ReTraceBackend.for_development_fixture() "
                f"only for deterministic development tests. Missing: {', '.join(missing)}"
            )
        self.ledgers: dict[str, EpisodeLedger] = {}
        self.stores: dict[str, BeliefStore] = {}
        self.extractor = extractor
        self.inducer = inducer
        if edge_proposal_strategy is not None:
            self.edge_proposal_strategy = edge_proposal_strategy
        elif batched_edge_verifier is not None:
            self.edge_proposal_strategy = BatchedEvidenceEdgeProposalStrategy(
                batched_edge_verifier,
                max_batch_beliefs=max_batch_beliefs,
            )
        elif edge_verifier is not None:
            self.edge_proposal_strategy = PerBeliefEvidenceEdgeProposalStrategy(edge_verifier)
        else:
            raise ValueError("ReTraceBackend requires an evidence-edge proposal strategy or verifier")
        self.impact_retriever = impact_retriever
        self.query_retriever = query_retriever
        if max_batch_beliefs < 1:
            raise ValueError("max_batch_beliefs must be >= 1")
        self.max_batch_beliefs = max_batch_beliefs
        self.last_ingest_stats: dict[str, Any] = {}
        self.gate = RevisionGate()

    @classmethod
    def for_development_fixture(
        cls,
        extractor: TypedBeliefExtractor | None = None,
        inducer: RequirementInducer | None = None,
        edge_verifier: EvidenceEdgeVerifier | None = None,
        batched_edge_verifier: BatchedEvidenceEdgeVerifier | None = None,
        edge_proposal_strategy: EvidenceEdgeProposalStrategy | None = None,
        impact_retriever: ImpactCandidateRetriever | None = None,
        query_retriever: QueryBeliefRetriever | None = None,
        **kwargs: Any,
    ) -> ReTraceBackend:
        """Development-only fixture backend; forbidden for paper main-result runners."""
        return cls(
            extractor=extractor or ManualTypedBeliefExtractor(),
            inducer=inducer or HeuristicRequirementInducer(),
            edge_verifier=edge_verifier or HeuristicEvidenceEdgeVerifier(),
            batched_edge_verifier=batched_edge_verifier,
            edge_proposal_strategy=edge_proposal_strategy,
            impact_retriever=impact_retriever or ManualImpactCandidateRetriever(),
            query_retriever=query_retriever or ManualQueryBeliefRetriever(),
            **kwargs,
        )

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

        proposal_result = self.edge_proposal_strategy.propose_edges(
            new_evidence=evidence,
            impact_candidates=tuple(impact_candidates),
            candidate_replacement_beliefs=tuple(new_beliefs),
            temporal_context=temporal_context,
        )
        self._admit_evidence_edges(proposal_result.edges, store, admitted_edges)

        self.last_ingest_stats = {
            "candidate_count": len(impact_candidates),
            "batch_count": proposal_result.batch_count,
            "max_batch_beliefs": self.max_batch_beliefs,
            "verifier_calls": proposal_result.verifier_calls,
            "evidence_chars": len(evidence.text),
            "latency_ms": proposal_result.latency_ms,
            "execution_mode": proposal_result.execution_mode,
        }

        return admitted_edges

    def _admit_evidence_edges(
        self,
        evidence_edges: tuple[EvidenceEdge, ...] | list[EvidenceEdge],
        store: BeliefStore,
        admitted_edges: list[EvidenceEdge],
    ) -> None:
        for edge in evidence_edges:
            gate_decision = self.gate.admit_evidence_edge(edge, store)
            if gate_decision.admitted:
                if not store.has_evidence_edge(edge.edge_id):
                    store.add_evidence_edge(edge)
                    admitted_edges.append(edge)

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        return result

    def answer(
        self,
        user_id: str,
        query: str,
        retrieved: list[dict[str, Any]] | dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del user_id, metadata

        retrieved_items = retrieved.get("authorized_basis", []) if isinstance(retrieved, dict) else retrieved

        context = "\n".join(item.get("proposition") or item.get("text", "") for item in retrieved_items)
        return f"Query: {query}\nAuthorized basis:\n{context}"

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self.ledgers:
            self.reset_user(user_id)
