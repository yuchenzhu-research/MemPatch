from __future__ import annotations

from dataclasses import replace
from typing import Any

from retracemem.evaluation.records import evaluation_record_from_backend_output
from retracemem.generation.basis_builder import BasisBuilder
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.schemas import Belief, EpisodicEvidence, EvaluationRecord, RelationPrediction, RelationType
from retracemem.tms.authorization import AuthorizationEngine
from retracemem.tms.gate import RevisionGate
from retracemem.verifier.base import RelationVerifier

try:
    from retracemem.verifier.heuristic_verifier import HeuristicRelationVerifier
except ImportError:  # pragma: no cover - another worker may add this module later.
    HeuristicRelationVerifier = None  # type: ignore[assignment]


class ReTracePipeline:
    """Local evidence-preserving belief revision pipeline."""

    def __init__(self, verifier: RelationVerifier | None = None) -> None:
        self.verifier = verifier if verifier is not None else self._default_verifier()
        self.ledgers: dict[str, EpisodeLedger] = {}
        self.stores: dict[str, BeliefStore] = {}
        self.gate = RevisionGate()

    def reset_user(self, user_id: str) -> None:
        self.ledgers[user_id] = EpisodeLedger()
        self.stores[user_id] = BeliefStore()

    def add_belief(self, user_id: str, belief: Belief) -> None:
        self._ensure_user(user_id)
        self.stores[user_id].add_belief(belief)

    def ingest_evidence(self, user_id: str, evidence: EpisodicEvidence) -> list[RelationPrediction]:
        self._ensure_user(user_id)
        ledger = self.ledgers[user_id]
        store = self.stores[user_id]
        ledger.append(evidence)

        accepted: list[RelationPrediction] = []
        for belief in list(store.all_beliefs()):
            prediction = self.verifier.verify(
                new_evidence=evidence,
                candidate_belief=belief,
                context={
                    "user_id": user_id,
                    "evidence_ids": ledger.ids(),
                    "relation_count": len(store.all_relations()),
                },
            )
            normalized = self._normalize_prediction(prediction, evidence, belief)
            if not self.gate.accept_local_relation(normalized):
                continue
            self._ensure_target_belief(store, normalized, evidence)
            store.add_relation(normalized)
            accepted.append(normalized)
        return accepted

    def authorized_basis(self, user_id: str, query: str, limit: int = 10) -> list[dict[str, str]]:
        self._ensure_user(user_id)
        return BasisBuilder(self.stores[user_id]).build(query=query, limit=limit)

    def answer(self, user_id: str, query: str, limit: int = 10) -> EvaluationRecord:
        self._ensure_user(user_id)
        store = self.stores[user_id]
        basis = self.authorized_basis(user_id, query, limit=limit)
        blocked = self._blocked_beliefs(store)
        context = "\n".join(item["text"] for item in basis)
        answer_text = f"Query: {query}\nAuthorized basis:\n{context}"

        record = evaluation_record_from_backend_output(
            query_id=f"{user_id}:{query}",
            method="retrace_pipeline",
            retrieved=[self._evidence_record(evidence) for evidence in self.ledgers[user_id].all()],
            candidate_beliefs=[self._belief_record(belief) for belief in store.all_beliefs()],
            authorized_basis=basis,
            blocked_beliefs=blocked,
            answer=answer_text,
        )
        return replace(record, authorized_basis=basis)

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self.ledgers:
            self.reset_user(user_id)

    @staticmethod
    def _normalize_prediction(
        prediction: RelationPrediction,
        evidence: EpisodicEvidence,
        belief: Belief,
    ) -> RelationPrediction:
        if not isinstance(prediction, RelationPrediction):
            raise TypeError("verifier.verify must return RelationPrediction")
        return replace(
            prediction,
            evidence_id=prediction.evidence_id or evidence.id,
            belief_id=prediction.belief_id or belief.id,
        )

    @staticmethod
    def _ensure_target_belief(
        store: BeliefStore,
        prediction: RelationPrediction,
        evidence: EpisodicEvidence,
    ) -> None:
        if prediction.relation != RelationType.SUPERSEDE:
            return
        if not prediction.target_belief_id or prediction.target_belief_id == prediction.belief_id:
            return
        if store.has_belief(prediction.target_belief_id):
            return
        store.add_belief(
            Belief(
                id=prediction.target_belief_id,
                proposition=evidence.text,
                supported_by=[evidence.id],
                metadata={"derived_from_relation": prediction.belief_id},
            )
        )

    @staticmethod
    def _blocked_beliefs(store: BeliefStore) -> list[dict[str, Any]]:
        engine = AuthorizationEngine(store)
        blocked: list[dict[str, Any]] = []
        for belief in store.all_beliefs():
            decision = engine.decide(belief)
            if decision.authorized:
                continue
            blocked.append(
                {
                    "belief_id": belief.id,
                    "text": belief.proposition,
                    "reason": decision.reason,
                    "justification_path": decision.justification_path,
                }
            )
        return blocked

    @staticmethod
    def _belief_record(belief: Belief) -> dict[str, Any]:
        return {
            "belief_id": belief.id,
            "text": belief.proposition,
            "supported_by": list(belief.supported_by),
            "status": belief.status.value,
            "metadata": dict(belief.metadata),
        }

    @staticmethod
    def _evidence_record(evidence: EpisodicEvidence) -> dict[str, Any]:
        return {
            "evidence_id": evidence.id,
            "timestamp": evidence.timestamp,
            "text": evidence.text,
            "source_id": evidence.source_id,
            "metadata": dict(evidence.metadata),
        }

    @staticmethod
    def _default_verifier() -> RelationVerifier:
        if HeuristicRelationVerifier is not None:
            return HeuristicRelationVerifier()
        return _UncertainVerifier()


class _UncertainVerifier:
    """Fallback used only when the heuristic verifier module is unavailable."""

    def verify(
        self,
        new_evidence: EpisodicEvidence,
        candidate_belief: Belief,
        context: dict[str, object] | None = None,
    ) -> RelationPrediction:
        del context
        return RelationPrediction(
            relation=RelationType.UNCERTAIN,
            evidence_id=new_evidence.id,
            belief_id=candidate_belief.id,
            rationale="No relation verifier is installed; defaulting to uncertain.",
            confidence=0.0,
        )
