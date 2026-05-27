from __future__ import annotations

from retracemem.schemas import EpisodicEvidence


class EpisodeLedger:
    """Append-only evidence ledger."""

    def __init__(self) -> None:
        self._items: dict[str, EpisodicEvidence] = {}

    def append(self, evidence: EpisodicEvidence) -> None:
        if not evidence.id:
            raise ValueError("evidence id is required")
        if evidence.id in self._items:
            raise ValueError(f"evidence already exists: {evidence.id}")
        self._items[evidence.id] = evidence

    def get(self, evidence_id: str) -> EpisodicEvidence:
        if evidence_id not in self._items:
            raise KeyError(f"unknown evidence: {evidence_id}")
        return self._items[evidence_id]

    def all(self) -> list[EpisodicEvidence]:
        return list(self._items.values())

    def ids(self) -> list[str]:
        return list(self._items)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def __len__(self) -> int:
        return len(self._items)
