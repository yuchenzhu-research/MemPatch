from __future__ import annotations

from typing import Any

from retracemem.extraction.base import BeliefExtractor
from retracemem.extraction.manual_fixture_extractor import ManualFixtureExtractor
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.retrieval.candidate_retriever import CandidateRelationRetriever, SimpleOverlapRetriever
from retracemem.schemas import EpisodicEvidence
from retracemem.tms.authorization import AuthorizationEngine
from retracemem.verifier.base import RelationVerifier


class ReTraceBackend:
    """End-to-End ReTrace backend with support for ablation configurations."""

    def __init__(
        self,
        extractor: BeliefExtractor | None = None,
        verifier: RelationVerifier | None = None,
        retriever: CandidateRelationRetriever | None = None,
        client: CachedLLMClient | None = None,
        model_id: str = "gemini-pro",
        provider: str = "google",
        disable_ledger: bool = False,
        disable_gate: bool = False,
        disable_temporal: bool = False,
    ) -> None:
        self.ledgers: dict[str, EpisodeLedger] = {}
        self.stores: dict[str, BeliefStore] = {}
        self.extractor = extractor or ManualFixtureExtractor()
        self.verifier = verifier
        self.retriever = retriever or SimpleOverlapRetriever()
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.disable_ledger = disable_ledger
        self.disable_gate = disable_gate
        self.disable_temporal = disable_temporal

        if self.verifier is None and self.client is not None:
            from retracemem.verifier.prompt_verifier import PromptRelationVerifier

            self.verifier = PromptRelationVerifier(
                self.client, model_id=self.model_id, provider=self.provider
            )

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

        if not session:
            return

        ledger = self.ledgers[user_id]
        store = self.stores[user_id]

        ev_id = session.get("id") or session.get("evidence_id") or session.get("session_id") or f"ev-{len(ledger)}"
        timestamp = session.get("timestamp") or ""
        text = session.get("text") or session.get("content") or ""
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        source_id = session.get("source_id") or "ingest"

        evidence = EpisodicEvidence(
            id=ev_id,
            timestamp=timestamp,
            text=text,
            source_id=source_id,
            metadata=session.get("metadata", {}),
        )
        # In ablation "disable_ledger", we still keep ledger instance locally
        # for API compatibility, but we might ignore it during query decision.
        ledger.append(evidence)

        extracted_beliefs = self.extractor.extract(evidence)
        for belief in extracted_beliefs:
            store.add_belief(belief)

        if self.verifier is not None:
            candidates = self.retriever.retrieve_candidates(evidence, store.all_beliefs())
            for candidate in candidates:
                if candidate.id in {b.id for b in extracted_beliefs}:
                    continue

                prediction = self.verifier.verify(evidence, candidate)
                store.add_relation(prediction)

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del metadata
        self._ensure_user(user_id)

        store = self.stores[user_id]
        ledger = None if self.disable_ledger else self.ledgers[user_id]

        engine = AuthorizationEngine(
            store,
            ledger,
            disable_gate=self.disable_gate,
            disable_temporal=self.disable_temporal,
        )
        basis: list[dict[str, Any]] = []

        for belief in store.all_beliefs():
            dec = engine.decide(belief)
            if dec.authorized:
                basis.append({"belief_id": belief.id, "text": belief.proposition})
            if len(basis) >= limit:
                break
        return basis

    def answer(
        self,
        user_id: str,
        query: str,
        retrieved: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del user_id, metadata

        if self.client is not None:
            context = "\n".join(f"- {item.get('text', '')}" for item in retrieved)
            prompt = f"Answer the user's query using the authorized beliefs provided below.\n\nAuthorized Beliefs:\n{context}\n\nQuery: {query}\n\nAnswer:"
            try:
                trace = self.client.generate(
                    prompt=prompt,
                    model_id=self.model_id,
                    provider=self.provider,
                    temperature=0.0,
                )
                if trace.status == "success" and trace.response:
                    return trace.response
            except Exception:
                pass

        context = "\n".join(item.get("text", "") for item in retrieved)
        return f"Query: {query}\nAuthorized basis:\n{context}"

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self.ledgers:
            self.reset_user(user_id)
