from __future__ import annotations

from typing import Iterable

from mempatch.dpa.memory.episode_ledger import EpisodeLedger
from mempatch.dpa.schemas import DefeatPath, EvidenceEdge


class MissingEvidenceError(KeyError):
    """Raised when an evidence edge references an ``evidence_id`` that is not
    present in the ``EpisodeLedger``.

    The DPA runtime must not fabricate a mock ledger
    or invent timestamps; a missing temporal anchor is a hard error, not a
    silent fallback.
    """


class TemporalValidity:
    """Deterministic temporal ordering for typed evidence edges.

    The recency key for an ``EvidenceEdge`` is the triple

        (evidence_timestamp, ledger_index, edge_id)

    drawn from the ``EpisodeLedger``. ``edge_id`` participates as the final
    tie-break so the algorithm is fully deterministic when two edges share
    the same evidence atom.

    No mock ledger is ever fabricated. If an ``EvidenceEdge`` references an
    ``evidence_id`` that the ledger does not know about, a
    ``MissingEvidenceError`` is raised so the caller has to repair the
    upstream ingestion rather than DPA inventing a timestamp.
    """

    # Sentinel used internally when an evidence atom has no timestamp at
    # all. Strings sort before any ISO-8601 timestamp, so a missing
    # timestamp is treated as the oldest evidence rather than fabricated.
    _EMPTY_TS = ""

    def __init__(self, ledger: EpisodeLedger) -> None:
        self.ledger = ledger
        self._index: dict[str, int] = {
            ev.evidence_id: idx for idx, ev in enumerate(ledger.all())
        }

    # ------------------------------------------------------------------
    # Single-evidence accessors
    # ------------------------------------------------------------------

    def evidence_index(self, evidence_id: str) -> int:
        if evidence_id not in self._index:
            raise MissingEvidenceError(
                f"evidence_id {evidence_id!r} is not present in the ledger; "
                "DPA refuses to fabricate a temporal position"
            )
        return self._index[evidence_id]

    def evidence_timestamp(self, evidence_id: str) -> str:
        # ``evidence_index`` raises MissingEvidenceError if absent.
        self.evidence_index(evidence_id)
        return self.ledger.get(evidence_id).timestamp or self._EMPTY_TS

    # ------------------------------------------------------------------
    # EvidenceEdge ordering
    # ------------------------------------------------------------------

    def edge_recency_key(self, edge: EvidenceEdge) -> tuple[str, int, str]:
        """``(timestamp, ledger_index, edge_id)`` triple, fully deterministic."""
        return (
            self.evidence_timestamp(edge.evidence_id),
            self.evidence_index(edge.evidence_id),
            edge.edge_id,
        )

    def edges_valid_at(
        self,
        edges: Iterable[EvidenceEdge],
        as_of_time: str | None = None,
        as_of_evidence_id: str | None = None,
    ) -> list[EvidenceEdge]:
        """Filter ``edges`` to those that are temporally valid at the cutoff.

        Both ``as_of_time`` (an ISO-8601 string) and ``as_of_evidence_id``
        (an explicit ledger position) are accepted. If both are provided
        the edge must satisfy *both* cutoffs (logical AND); this lets a
        caller pin a query to an exact ledger position while still
        rejecting edges whose recorded timestamp drifts past the wall-clock
        cutoff.

        An edge with ``valid_until`` strictly less than the cutoff
        timestamp is excluded; ``valid_from`` strictly greater than the
        cutoff timestamp is also excluded. ``valid_from`` / ``valid_until``
        are interpreted as ISO-8601 strings and compared lexicographically.
        """
        result: list[EvidenceEdge] = []
        cutoff_ts: str | None = None
        cutoff_pos: tuple[str, int] | None = None
        if as_of_evidence_id is not None:
            cutoff_pos = (
                self.evidence_timestamp(as_of_evidence_id),
                self.evidence_index(as_of_evidence_id),
            )
        if as_of_time is not None:
            cutoff_ts = as_of_time

        for edge in edges:
            ts = self.evidence_timestamp(edge.evidence_id)
            idx = self.evidence_index(edge.evidence_id)
            # Cutoff on ledger position.
            if cutoff_pos is not None and (ts, idx) > cutoff_pos:
                continue
            # Cutoff on wall-clock timestamp.
            if cutoff_ts is not None and ts > cutoff_ts:
                continue
            # Edge-intrinsic validity window.
            if edge.valid_from is not None:
                horizon = cutoff_ts if cutoff_ts is not None else ts
                if edge.valid_from > horizon:
                    continue
            if edge.valid_until is not None:
                horizon = cutoff_ts if cutoff_ts is not None else ts
                if edge.valid_until < horizon:
                    continue
            result.append(edge)
        return result

    def latest_edge(self, edges: Iterable[EvidenceEdge]) -> EvidenceEdge | None:
        """Deterministic ``max`` by ``edge_recency_key``; ``None`` if empty."""
        materialized = list(edges)
        if not materialized:
            return None
        return max(materialized, key=self.edge_recency_key)

    # ------------------------------------------------------------------
    # DefeatPath ordering
    # ------------------------------------------------------------------

    def path_recency_key(
        self,
        path: DefeatPath,
        edge_lookup: dict[str, EvidenceEdge],
    ) -> tuple[str, int, str]:
        """Recency key for a defeat path: the key of its most recent
        supporting evidence edge.

        ``edge_lookup`` maps ``edge_id`` to the corresponding
        ``EvidenceEdge`` (typically ``BeliefStore._evidence_edges`` view).
        The path is ranked by its most recent supporting evidence edge so
        that competing defeat paths are ordered by the freshest defeating
        evidence rather than by an arbitrary insertion order.
        """
        if not path.supporting_evidence_edge_ids:
            # A path with no supporting evidence edges is logically the
            # oldest; this is used only for defensive ranking and should
            # not arise from a well-formed DPA construction.
            return (self._EMPTY_TS, -1, path.path_id)
        keys = [
            self.edge_recency_key(edge_lookup[edge_id])
            for edge_id in path.supporting_evidence_edge_ids
        ]
        return max(keys)

    def latest_path(
        self,
        paths: Iterable[DefeatPath],
        edge_lookup: dict[str, EvidenceEdge],
    ) -> DefeatPath | None:
        materialized = list(paths)
        if not materialized:
            return None
        return max(materialized, key=lambda p: self.path_recency_key(p, edge_lookup))
