from __future__ import annotations

from typing import Any, Protocol


class MemoryBackend(Protocol):
    """Benchmark-neutral memory system interface.

    The shape intentionally stays close to Memora's BaseMemorySystem while
    remaining usable for STALE/CUPMem-style sample runners.
    """

    def reset_user(self, user_id: str) -> None:
        ...

    def ingest_session(
        self,
        user_id: str,
        session: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ...

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def answer(
        self,
        user_id: str,
        query: str,
        retrieved: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        ...
