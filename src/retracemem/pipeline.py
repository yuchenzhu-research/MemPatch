from __future__ import annotations

from dataclasses import replace
from typing import Any

from retracemem.evaluation.records import evaluation_record_from_backend_output
from retracemem.backends.retrace_backend import ReTraceBackend
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    EvidenceEdge,
    EvaluationRecord,
)
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.extraction.typed_extractor import TypedBeliefExtractor
from retracemem.verifier.contracts import BatchedEvidenceEdgeVerifier, RequirementInducer, EvidenceEdgeVerifier
from retracemem.verifier.proposal_strategy import EvidenceEdgeProposalStrategy
from retracemem.retrieval.typed_retrievers import ImpactCandidateRetriever, QueryBeliefRetriever
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm


class ReTracePipeline:
    """Local evidence-preserving belief revision pipeline routing to typed backend."""

    def __init__(
        self,
        backend: ReTraceBackend | None = None,
        *,
        extractor: TypedBeliefExtractor | None = None,
        inducer: RequirementInducer | None = None,
        edge_verifier: EvidenceEdgeVerifier | None = None,
        batched_edge_verifier: BatchedEvidenceEdgeVerifier | None = None,
        edge_proposal_strategy: EvidenceEdgeProposalStrategy | None = None,
        impact_retriever: ImpactCandidateRetriever | None = None,
        query_retriever: QueryBeliefRetriever | None = None,
        client: CachedLLMClient | None = None,
        model_id: str = "gemini-pro",
        provider: str = "google",
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        
        if backend is not None:
            self.backend = backend
        else:
            components = {
                "extractor": extractor,
                "inducer": inducer,
                "edge_verifier": edge_proposal_strategy or edge_verifier or batched_edge_verifier,
                "impact_retriever": impact_retriever,
                "query_retriever": query_retriever,
            }
            missing = [k for k, v in components.items() if v is None]
            if missing:
                raise ValueError(
                    "ReTracePipeline requires either an explicit backend or all five typed components; "
                    "use ReTracePipeline.for_development_fixture() only for deterministic development tests. "
                    f"Missing: {', '.join(missing)}"
                )
            self.backend = ReTraceBackend(
                extractor=extractor,
                inducer=inducer,
                edge_verifier=edge_verifier,
                batched_edge_verifier=batched_edge_verifier,
                edge_proposal_strategy=edge_proposal_strategy,
                impact_retriever=impact_retriever,
                query_retriever=query_retriever,
            )

        if self.client is not None:
            from retracemem.generation.answer_generator import PromptAnswerGenerator
            self.answer_generator = PromptAnswerGenerator(client=self.client)
        else:
            self.answer_generator = None

    @classmethod
    def for_development_fixture(
        cls,
        *,
        extractor: TypedBeliefExtractor | None = None,
        inducer: RequirementInducer | None = None,
        edge_verifier: EvidenceEdgeVerifier | None = None,
        batched_edge_verifier: BatchedEvidenceEdgeVerifier | None = None,
        edge_proposal_strategy: EvidenceEdgeProposalStrategy | None = None,
        impact_retriever: ImpactCandidateRetriever | None = None,
        query_retriever: QueryBeliefRetriever | None = None,
        client: CachedLLMClient | None = None,
        model_id: str = "mock",
        provider: str = "mock",
    ) -> ReTracePipeline:
        """Development-only deterministic fixture pipeline; forbidden for paper main-result runners."""
        backend = ReTraceBackend.for_development_fixture(
            extractor=extractor,
            inducer=inducer,
            edge_verifier=edge_verifier,
            batched_edge_verifier=batched_edge_verifier,
            edge_proposal_strategy=edge_proposal_strategy,
            impact_retriever=impact_retriever,
            query_retriever=query_retriever,
        )
        return cls(backend=backend, client=client, model_id=model_id, provider=provider)

    @property
    def stores(self) -> dict[str, BeliefStore]:
        return self.backend.stores

    @property
    def ledgers(self) -> dict[str, EpisodeLedger]:
        return self.backend.ledgers

    def reset_user(self, user_id: str) -> None:
        self.backend.reset_user(user_id)

    def add_belief(self, user_id: str, belief: BeliefNode) -> None:
        self.backend._ensure_user(user_id)
        meta = dict(belief.metadata) if belief.metadata else {}
        meta["scope_id"] = user_id
        updated_belief = replace(belief, metadata=meta)
        self.backend.stores[user_id].add_belief(updated_belief)

    def ingest_evidence(self, user_id: str, evidence: EvidenceNode) -> list[EvidenceEdge]:
        return self.backend.ingest_evidence(user_id, evidence)

    def authorized_basis(self, user_id: str, query: str, limit: int = 10, method: str = "retrace") -> list[dict[str, Any]]:
        self.backend._ensure_user(user_id)
        store = self.backend.stores[user_id]
        ledger = self.backend.ledgers[user_id]

        if method == "retrace":
            return self.backend.search(user_id, query, limit=limit)["authorized_basis"]
        
        elif method == "directjudge":
            if self.client is None:
                raise ValueError("CachedLLMClient is required to run directjudge authorized_basis.")
            
            from retracemem.methods.contracts import SharedCandidateView
            from retracemem.methods.directjudge import DirectJudgeLLM

            beliefs = tuple(store.all_beliefs())
            candidate_beliefs = self.backend.query_retriever.retrieve_for_query(query, beliefs, limit=limit)
            
            new_evidence = ledger.all()[-1] if len(ledger) > 0 else EvidenceNode(
                evidence_id="empty", session_id=user_id, timestamp=None, text="", source_dataset="empty", source_pointer="empty"
            )

            candidate_conditions = []
            candidate_deps = []
            for b in candidate_beliefs:
                dep_edges = store.dependencies_of(b.belief_id)
                conds = []
                for edge in dep_edges:
                    if store.has_condition(edge.condition_id):
                        conds.append(store.get_condition(edge.condition_id))
                candidate_conditions.append((b.belief_id, tuple(conds)))
                candidate_deps.append((b.belief_id, tuple(dep_edges)))

            view = SharedCandidateView(
                instance_id=user_id,
                query_id=f"q_{user_id}",
                query=query,
                evidence_context=tuple(ledger.all()),
                new_evidence=new_evidence,
                candidate_beliefs=tuple(candidate_beliefs),
                candidate_replacement_beliefs=(),
                candidate_conditions_by_belief=tuple(candidate_conditions),
                dependency_edges_by_belief=tuple(candidate_deps),
            )

            judge = DirectJudgeLLM(client=self.client, model_id=self.model_id, provider=self.provider)
            method_result = judge.judge(view)

            authorized_basis = []
            for b in candidate_beliefs:
                if b.belief_id in method_result.authorized_belief_ids:
                    authorized_basis.append({
                        "belief_id": b.belief_id,
                        "proposition": b.proposition,
                        "source_evidence_ids": list(b.source_evidence_ids),
                        "authorization_status": "AUTHORIZED",
                    })
            return authorized_basis
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def answer(self, user_id: str, query: str, limit: int = 10, method: str = "retrace") -> EvaluationRecord:
        self.backend._ensure_user(user_id)
        store = self.backend.stores[user_id]
        ledger = self.backend.ledgers[user_id]

        if method == "retrace":
            search_result = self.backend.search(user_id, query, limit=limit)
            basis = search_result["authorized_basis"]
            excluded = search_result.get("excluded", [])
        elif method == "directjudge":
            basis = self.authorized_basis(user_id, query, limit=limit, method=method)
            basis_ids = {item["belief_id"] for item in basis}
            candidate_beliefs = self.backend.query_retriever.retrieve_for_query(
                query, tuple(store.all_beliefs()), limit=limit
            )
            excluded = []
            for b in candidate_beliefs:
                if b.belief_id not in basis_ids:
                    excluded.append({
                        "belief_id": b.belief_id,
                        "status": "NOT_USABLE",
                        "accepted_defeat_path": None,
                    })
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

        blocked = self._excluded_to_blocked(excluded, store)

        if self.answer_generator is not None:
            answer_text = self.answer_generator.generate_answer(
                query=query,
                authorized_basis=basis,
                model_id=self.model_id,
                provider=self.provider,
            )
        else:
            context = "\n".join(item.get("proposition") or item.get("text", "") for item in basis)
            answer_text = f"Query: {query}\nAuthorized basis:\n{context}"

        record = evaluation_record_from_backend_output(
            query_id=f"{user_id}:{query}",
            method=f"{method}_pipeline",
            retrieved=[self._evidence_record(evidence) for evidence in ledger.all()],
            candidate_beliefs=[self._belief_record(belief) for belief in store.all_beliefs()],
            authorized_basis=basis,
            blocked_beliefs=blocked,
            answer=answer_text,
        )
        return replace(record, authorized_basis=basis)

    @staticmethod
    def _excluded_to_blocked(excluded: list[dict[str, Any]], store: BeliefStore) -> list[dict[str, Any]]:
        blocked: list[dict[str, Any]] = []
        for item in excluded:
            belief_id = item["belief_id"]
            text = store.get_belief(belief_id).proposition if store.has_belief(belief_id) else ""
            defeat_path = item.get("accepted_defeat_path")
            blocked.append({
                "belief_id": belief_id,
                "text": text,
                "reason": item["status"],
                "justification_path": [defeat_path["path_id"]] if defeat_path else [],
            })
        return blocked

    @staticmethod
    def _belief_record(belief: BeliefNode) -> dict[str, Any]:
        return {
            "belief_id": belief.belief_id,
            "text": belief.proposition,
            "supported_by": list(belief.source_evidence_ids),
            "status": "authorized",
            "metadata": dict(belief.metadata),
        }

    @staticmethod
    def _evidence_record(evidence: EvidenceNode) -> dict[str, Any]:
        return {
            "evidence_id": evidence.evidence_id,
            "timestamp": evidence.timestamp,
            "text": evidence.text,
            "source_id": evidence.source_dataset,
            "metadata": dict(evidence.metadata),
        }
