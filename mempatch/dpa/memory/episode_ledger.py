from __future__ import annotations

from mempatch.dpa.schemas import EvidenceNode


class EpisodeLedger:
    """Append-only evidence ledger."""

    def __init__(self) -> None:
        self._items: dict[str, EvidenceNode] = {}

    def append(self, evidence: EvidenceNode) -> None:
        if not evidence.evidence_id:
            raise ValueError("evidence_id is required")
        if evidence.evidence_id in self._items:
            raise ValueError(f"evidence already exists: {evidence.evidence_id}")
        self._items[evidence.evidence_id] = evidence

    def get(self, evidence_id: str) -> EvidenceNode:
        if evidence_id not in self._items:
            raise KeyError(f"unknown evidence: {evidence_id}")
        return self._items[evidence_id]

    def all(self) -> list[EvidenceNode]:
        return list(self._items.values())

    def ids(self) -> list[str]:
        return list(self._items)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def __len__(self) -> int:
        return len(self._items)

