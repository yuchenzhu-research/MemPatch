from __future__ import annotations

from typing import Any
from retracemem.schemas import RelationPrediction, RelationType
from retracemem.memory.episode_ledger import EpisodeLedger


class TemporalValidity:
    """Manages temporal validity of conditions and beliefs based on evidence ledger order."""

    def __init__(self, ledger: EpisodeLedger) -> None:
        self.ledger = ledger

    def get_chronological_relations(self, relations: list[RelationPrediction]) -> list[RelationPrediction]:
        """Sorts relations chronologically based on their associated evidence timestamps/order in ledger."""
        evidences = self.ledger.all()
        evidence_order = {ev.id: idx for idx, ev in enumerate(evidences)}

        def get_key(rel: RelationPrediction) -> tuple[str, int]:
            if not rel.evidence_id:
                return ("", 0)
            try:
                ev = self.ledger.get(rel.evidence_id)
                ts = ev.timestamp or ""
                idx = evidence_order.get(rel.evidence_id, 999999)
                return (ts, idx)
            except KeyError:
                return ("", 999999)

        return sorted(relations, key=get_key)

    def is_condition_blocked(
        self, condition: str, relations: list[RelationPrediction], at_evidence_id: str | None = None
    ) -> bool:
        """Determines if a condition is blocked at a specific logical time."""
        relevant = [
            r
            for r in relations
            if r.condition == condition and r.relation in {RelationType.BLOCK, RelationType.SUPPORT}
        ]

        if at_evidence_id:
            evidences = self.ledger.all()
            evidence_order = {ev.id: idx for idx, ev in enumerate(evidences)}
            if at_evidence_id not in evidence_order:
                # If the cutoff evidence is not in ledger, we default to no cutoff filtering
                pass
            else:
                cutoff_idx = evidence_order[at_evidence_id]
                cutoff_ev = self.ledger.get(at_evidence_id)
                cutoff_ts = cutoff_ev.timestamp or ""

                filtered_relevant = []
                for r in relevant:
                    if not r.evidence_id or r.evidence_id not in evidence_order:
                        continue
                    r_ev = self.ledger.get(r.evidence_id)
                    r_ts = r_ev.timestamp or ""
                    r_idx = evidence_order[r.evidence_id]
                    if (r_ts, r_idx) <= (cutoff_ts, cutoff_idx):
                        filtered_relevant.append(r)
                relevant = filtered_relevant

        sorted_relevant = self.get_chronological_relations(relevant)

        if not sorted_relevant:
            return False

        latest_rel = sorted_relevant[-1]
        return latest_rel.relation == RelationType.BLOCK
