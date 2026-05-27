from __future__ import annotations

from retracemem.memory.belief_store import BeliefStore
from retracemem.tms.authorization import AuthorizationEngine


class BasisBuilder:
    """Build a query-time authorized current basis."""

    def __init__(self, store: BeliefStore) -> None:
        self.store = store
        self.authorization = AuthorizationEngine(store)

    def build(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        del query
        if limit <= 0:
            return []

        basis: list[dict[str, str]] = []
        for belief in self.store.all_beliefs():
            decision = self.authorization.decide(belief)
            if decision.authorized:
                basis.append({"belief_id": belief.id, "text": belief.proposition})
            if len(basis) >= limit:
                break
        return basis
