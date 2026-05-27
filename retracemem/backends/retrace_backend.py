from __future__ import annotations

from typing import Any

from retracemem.generation.basis_builder import BasisBuilder
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger


class ReTraceBackend:
    """Minimal local backend shell for ReTrace experiments."""

    def __init__(self) -> None:
        self.ledgers: dict[str, EpisodeLedger] = {}
        self.stores: dict[str, BeliefStore] = {}

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
        # The real ingestion path will call belief extraction and verification.
        self.ledgers[user_id]
        self.stores[user_id]
        if not session:
            return

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del metadata
        self._ensure_user(user_id)
        basis = BasisBuilder(self.stores[user_id]).build(query=query, limit=limit)
        return basis

    def answer(
        self,
        user_id: str,
        query: str,
        retrieved: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del user_id, metadata
        context = "\n".join(item.get("text", "") for item in retrieved)
        return f"Query: {query}\nAuthorized basis:\n{context}"

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self.ledgers:
            self.reset_user(user_id)
