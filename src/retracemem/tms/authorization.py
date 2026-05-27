from __future__ import annotations

from retracemem.memory.belief_store import BeliefStore
from retracemem.schemas import AuthorizationDecision, Belief, BeliefStatus, RelationType
from retracemem.tms.gate import RevisionGate


class AuthorizationEngine:
    """Compute whether a belief may govern current answers."""

    def __init__(self, store: BeliefStore) -> None:
        self.store = store
        self.gate = RevisionGate()

    def decide(self, belief: Belief, at_time: str | None = None) -> AuthorizationDecision:
        del at_time
        status_decision = self._decision_from_status(belief)
        if status_decision is not None:
            return status_decision

        relations = [
            relation
            for relation in self.store.relations_for_belief(belief.id)
            if self.gate.accept_local_relation(relation)
        ]

        for relation in relations:
            if relation.relation == RelationType.SUPERSEDE and relation.belief_id == belief.id:
                return AuthorizationDecision(
                    belief_id=belief.id,
                    authorized=False,
                    reason="superseded",
                    justification_path=[self._relation_ref(relation)],
                )

        for relation in relations:
            if relation.relation == RelationType.BLOCK and relation.belief_id == belief.id:
                return AuthorizationDecision(
                    belief_id=belief.id,
                    authorized=False,
                    reason="blocked",
                    justification_path=[self._relation_ref(relation)],
                )

        for relation in relations:
            if relation.relation == RelationType.UNCERTAIN and relation.belief_id == belief.id:
                return AuthorizationDecision(
                    belief_id=belief.id,
                    authorized=False,
                    reason="uncertain",
                    justification_path=[self._relation_ref(relation)],
                )

        return AuthorizationDecision(
            belief_id=belief.id,
            authorized=True,
            reason="supported",
            justification_path=list(belief.supported_by),
        )

    @staticmethod
    def _decision_from_status(belief: Belief) -> AuthorizationDecision | None:
        if belief.status == BeliefStatus.AUTHORIZED:
            return None
        if belief.status == BeliefStatus.BLOCKED:
            return AuthorizationDecision(
                belief_id=belief.id,
                authorized=False,
                reason="blocked",
                justification_path=list(belief.supported_by),
            )
        if belief.status == BeliefStatus.UNRESOLVED:
            return AuthorizationDecision(
                belief_id=belief.id,
                authorized=False,
                reason="unresolved",
                justification_path=list(belief.supported_by),
            )
        if belief.status == BeliefStatus.HISTORICAL:
            return AuthorizationDecision(
                belief_id=belief.id,
                authorized=False,
                reason="historical",
                justification_path=list(belief.supported_by),
            )
        return None

    @staticmethod
    def _relation_ref(relation: object) -> str:
        evidence_id = getattr(relation, "evidence_id", None) or "unknown_evidence"
        belief_id = getattr(relation, "belief_id", None) or "unknown_belief"
        relation_type = getattr(relation, "relation", None)
        relation_name = getattr(relation_type, "value", str(relation_type))
        return f"{evidence_id}:{relation_name}:{belief_id}"
