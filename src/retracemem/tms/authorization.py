from __future__ import annotations

from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.memory.temporal_validity import TemporalValidity
from retracemem.schemas import AuthorizationDecision, Belief, BeliefStatus, RelationPrediction, RelationType
from retracemem.tms.gate import RevisionGate


class AuthorizationEngine:
    """Compute whether a belief may govern current answers using temporal state machine."""

    def __init__(
        self,
        store: BeliefStore,
        ledger: EpisodeLedger | None = None,
        disable_gate: bool = False,
        disable_temporal: bool = False,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.gate = RevisionGate()
        self.disable_gate = disable_gate
        self.disable_temporal = disable_temporal

    def _ensure_mock_ledger(self) -> None:
        if self.ledger is not None:
            return
        from retracemem.memory.episode_ledger import EpisodeLedger
        from retracemem.schemas import EpisodicEvidence
        self.ledger = EpisodeLedger()
        for idx, rel in enumerate(self.store.all_relations()):
            if rel.evidence_id and rel.evidence_id not in self.ledger:
                ts = getattr(rel, "valid_from", None) or f"2026-05-27T00:{idx:02d}:00Z"
                try:
                    mock_ev = EpisodicEvidence(
                        id=rel.evidence_id,
                        timestamp=ts,
                        text=f"Mock evidence {rel.evidence_id}",
                        source_id="mock",
                    )
                    self.ledger.append(mock_ev)
                except Exception:
                    pass

    def _is_before_cutoff(
        self, ev_id: str, cutoff_ts: str | None, cutoff_ev_id: str | None, evidence_order: dict[str, int]
    ) -> bool:
        if not ev_id:
            return True
        if ev_id not in evidence_order:
            return False

        ev = self.ledger.get(ev_id)
        ev_ts = ev.timestamp or ""
        ev_idx = evidence_order[ev_id]

        if cutoff_ev_id and cutoff_ev_id in evidence_order:
            limit_ev = self.ledger.get(cutoff_ev_id)
            limit_ts = limit_ev.timestamp or ""
            limit_idx = evidence_order[cutoff_ev_id]
            if (ev_ts, ev_idx) > (limit_ts, limit_idx):
                return False

        if cutoff_ts:
            if ev_ts > cutoff_ts:
                return False

        return True

    def _is_cond_blocked_ablated(
        self, cond_name: str, valid_rels: list[RelationPrediction], at_evidence_id: str | None
    ) -> bool:
        if self.disable_temporal:
            return any(r.condition == cond_name and r.relation == RelationType.BLOCK for r in valid_rels)
        temp_validity = TemporalValidity(self.ledger)
        return temp_validity.is_condition_blocked(cond_name, valid_rels, at_evidence_id=at_evidence_id)

    def decide(
        self,
        belief: Belief,
        at_time: str | None = None,
        at_evidence_id: str | None = None,
    ) -> AuthorizationDecision:
        self._ensure_mock_ledger()

        assert self.ledger is not None
        evidence_order = {ev.id: idx for idx, ev in enumerate(self.ledger.all())}

        cutoff_ts = None
        cutoff_ev_id = at_evidence_id
        if at_time:
            if at_time in evidence_order:
                cutoff_ev_id = at_time
            else:
                cutoff_ts = at_time

        # Filter relations with optional ablation gate/temporal
        valid_rels: list[RelationPrediction] = []
        for r in self.store.all_relations():
            if not self.disable_gate:
                if not self.gate.accept_local_relation(r):
                    continue
            else:
                if r.relation == RelationType.NONE:
                    continue

            if not self.disable_temporal:
                if r.evidence_id and not self._is_before_cutoff(
                    r.evidence_id, cutoff_ts, cutoff_ev_id, evidence_order
                ):
                    continue
            valid_rels.append(r)

        # Check explicit status
        status_decision = self._decision_from_status(belief)
        if status_decision is not None:
            return status_decision

        # 1. Supersede
        supersede_rels = [
            r for r in valid_rels if r.relation == RelationType.SUPERSEDE and r.belief_id == belief.id
        ]
        if supersede_rels:
            if self.disable_temporal:
                latest_supersede = supersede_rels[-1]
            else:
                temp_validity = TemporalValidity(self.ledger)
                sorted_supersedes = temp_validity.get_chronological_relations(supersede_rels)
                latest_supersede = sorted_supersedes[-1]
            return AuthorizationDecision(
                belief_id=belief.id,
                authorized=False,
                reason="superseded",
                justification_path=[self._relation_ref(latest_supersede)],
            )

        # 2. Uncertain
        uncertain_rels = [
            r for r in valid_rels if r.relation == RelationType.UNCERTAIN and r.belief_id == belief.id
        ]
        if uncertain_rels:
            if self.disable_temporal:
                latest_unc = uncertain_rels[-1]
            else:
                temp_validity = TemporalValidity(self.ledger)
                sorted_uncs = temp_validity.get_chronological_relations(uncertain_rels)
                latest_unc = sorted_uncs[-1]
            return AuthorizationDecision(
                belief_id=belief.id,
                authorized=False,
                reason="uncertain",
                justification_path=[self._relation_ref(latest_unc)],
            )

        # 3. Direct Block
        direct_block_rels = [
            r for r in valid_rels if r.relation == RelationType.BLOCK and r.belief_id == belief.id
        ]
        if direct_block_rels:
            temp_validity = TemporalValidity(self.ledger)
            for d_rel in direct_block_rels:
                if d_rel.condition:
                    if self._is_cond_blocked_ablated(d_rel.condition, valid_rels, at_evidence_id=cutoff_ev_id):
                        return AuthorizationDecision(
                            belief_id=belief.id,
                            authorized=False,
                            reason="blocked",
                            justification_path=[self._relation_ref(d_rel)],
                        )
                else:
                    if self.disable_temporal:
                        direct_blocks = [
                            r
                            for r in valid_rels
                            if r.belief_id == belief.id and not r.condition and r.relation == RelationType.BLOCK
                        ]
                        if direct_blocks:
                            return AuthorizationDecision(
                                belief_id=belief.id,
                                authorized=False,
                                reason="blocked",
                                justification_path=[self._relation_ref(direct_blocks[-1])],
                            )
                    else:
                        direct_status_rels = [
                            r
                            for r in valid_rels
                            if r.belief_id == belief.id
                            and not r.condition
                            and r.relation in {RelationType.BLOCK, RelationType.SUPPORT}
                        ]
                        sorted_direct = temp_validity.get_chronological_relations(direct_status_rels)
                        if sorted_direct and sorted_direct[-1].relation == RelationType.BLOCK:
                            return AuthorizationDecision(
                                belief_id=belief.id,
                                authorized=False,
                                reason="blocked",
                                justification_path=[self._relation_ref(sorted_direct[-1])],
                            )

        # 4. Prerequisite Condition Block (REQUIRED_BY)
        required_relations = [
            r for r in valid_rels if r.relation == RelationType.REQUIRED_BY and r.belief_id == belief.id
        ]
        for req_rel in required_relations:
            cond_name = req_rel.condition
            if not cond_name:
                continue
            if self._is_cond_blocked_ablated(cond_name, valid_rels, at_evidence_id=cutoff_ev_id):
                cond_blocks = [
                    r for r in valid_rels if r.condition == cond_name and r.relation == RelationType.BLOCK
                ]
                if self.disable_temporal:
                    latest_block = cond_blocks[-1] if cond_blocks else req_rel
                else:
                    temp_validity = TemporalValidity(self.ledger)
                    sorted_blocks = temp_validity.get_chronological_relations(cond_blocks)
                    latest_block = sorted_blocks[-1] if sorted_blocks else req_rel

                return AuthorizationDecision(
                    belief_id=belief.id,
                    authorized=False,
                    reason="blocked",
                    justification_path=[self._relation_ref(latest_block)],
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
