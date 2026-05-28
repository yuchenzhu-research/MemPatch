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
from retracemem.verifier.contracts import RequirementInducer, EvidenceEdgeVerifier
from retracemem.retrieval.typed_retrievers import ImpactCandidateRetriever, QueryBeliefRetriever


class ReTracePipeline:
    """Local evidence-preserving belief revision pipeline routing to typed backend."""

    def __init__(
        self,
        backend: ReTraceBackend | None = None,
        *,
        extractor: TypedBeliefExtractor | None = None,
        inducer: RequirementInducer | None = None,
        edge_verifier: EvidenceEdgeVerifier | None = None,
        impact_retriever: ImpactCandidateRetriever | None = None,
        query_retriever: QueryBeliefRetriever | None = None,
    ) -> None:
        if backend is not None:
            self.backend = backend
        else:
            components = {
                "extractor": extractor,
                "inducer": inducer,
                "edge_verifier": edge_verifier,
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
                impact_retriever=impact_retriever,
                query_retriever=query_retriever,
            )

    @classmethod
    def for_development_fixture(
        cls,
        *,
        extractor: TypedBeliefExtractor | None = None,
        inducer: RequirementInducer | None = None,
        edge_verifier: EvidenceEdgeVerifier | None = None,
        impact_retriever: ImpactCandidateRetriever | None = None,
        query_retriever: QueryBeliefRetriever | None = None,
    ) -> ReTracePipeline:
        """Development-only deterministic fixture pipeline; forbidden for paper main-result runners."""
        backend = ReTraceBackend.for_development_fixture(
            extractor=extractor,
            inducer=inducer,
            edge_verifier=edge_verifier,
            impact_retriever=impact_retriever,
            query_retriever=query_retriever,
        )
        return cls(backend=backend)

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

    def authorized_basis(self, user_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.backend.search(user_id, query, limit=limit)["authorized_basis"]

    def answer(self, user_id: str, query: str, limit: int = 10) -> EvaluationRecord:
        self.backend._ensure_user(user_id)
        store = self.backend.stores[user_id]
        ledger = self.backend.ledgers[user_id]

        search_result = self.backend.search(user_id, query, limit=limit)
        basis = search_result["authorized_basis"]
        blocked = self._excluded_to_blocked(search_result.get("excluded", []), store)

        context = "\n".join(item.get("proposition") or item.get("text", "") for item in basis)
        answer_text = f"Query: {query}\nAuthorized basis:\n{context}"

        record = evaluation_record_from_backend_output(
            query_id=f"{user_id}:{query}",
            method="retrace_pipeline",
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
