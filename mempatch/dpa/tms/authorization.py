from __future__ import annotations

from mempatch.dpa.memory.belief_store import BeliefStore
from mempatch.dpa.memory.episode_ledger import EpisodeLedger
from mempatch.dpa.memory.temporal_validity import TemporalValidity
from mempatch.dpa.schemas import (
    AuthorizationStatus,
    AuthorizationTrace,
    DefeatPath,
    DefeatPathType,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
)


class DefeatPathAuthorizationAlgorithm:
    """Deterministic Defeat-Path Authorization (DPA) over a typed graph.

    The algorithm consumes only admitted typed edges from a ``BeliefStore``
    and the ordering provided by ``TemporalValidity`` over an
    ``EpisodeLedger``. It never invokes verifiers and never inspects flat
    ``RelationPrediction`` objects.

    **Precedence**:

        SUPERSEDES  >  PREREQUISITE_BLOCK  >  UNRESOLVED_UNCERTAIN  >  AUTHORIZED

    **Critical semantic clarification**:

        ``AUTHORIZED`` means the belief is *eligible* to participate in the
        current authorized basis. It does **not** mean the system has
        newly verified that the belief is presently true. Therefore:

        - ``RELEASES`` clears a blocker and may make an older belief
          eligible again. ``RELEASES`` never creates a new assertion that
          the belief is currently true.
        - A later ``SUPERSEDES`` edge still defeats an old belief even if
          an older blocker was previously released.

    **Tie-breaking** is fully deterministic by
    ``(evidence_timestamp, ledger_index, edge_id)``; see
    ``TemporalValidity.edge_recency_key``.

    No mock ledger is ever fabricated. If an edge references an evidence
    atom that the ledger does not know about, the underlying
    ``TemporalValidity`` raises ``MissingEvidenceError`` rather than
    inventing a position.
    """

    def __init__(self, store: BeliefStore, ledger: EpisodeLedger) -> None:
        self.store = store
        self.ledger = ledger
        self.temporal = TemporalValidity(ledger)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def authorize(
        self,
        belief_id: str,
        as_of_time: str | None = None,
        as_of_evidence_id: str | None = None,
        query_id: str | None = None,
    ) -> AuthorizationTrace:
        """Compute the ``AuthorizationTrace`` for ``belief_id`` at the cutoff.

        ``as_of_time`` and ``as_of_evidence_id`` are both accepted and
        combined as a logical AND (see ``TemporalValidity.edges_valid_at``).
        ``query_id`` is recorded on the resulting trace for downstream
        provenance hydration but does not influence the algorithm.
        """
        belief = self.store.get_belief(belief_id)
        trace_id = self._trace_id(belief_id, as_of_time, as_of_evidence_id, query_id)

        # 1. Direct supersession (highest precedence).
        super_trace = self._check_supersession(
            belief_id=belief_id,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
            trace_id=trace_id,
            query_id=query_id,
        )
        if super_trace is not None:
            return super_trace

        # 2. Conditional defeat through an accepted REQUIRES dependency
        #    whose latest evidence update is BLOCKS.
        blocked_trace = self._check_prerequisite_block(
            belief_id=belief_id,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
            trace_id=trace_id,
            query_id=query_id,
        )
        if blocked_trace is not None:
            return blocked_trace

        # 3. Belief-level uncertainty, possibly cleared by a later REAFFIRMS.
        unresolved_trace = self._check_belief_status(
            belief_id=belief_id,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
            trace_id=trace_id,
            query_id=query_id,
        )
        if unresolved_trace is not None:
            return unresolved_trace

        # 4. No accepted defeat path: authorized (eligibility only).
        return AuthorizationTrace(
            trace_id=trace_id,
            belief_id=belief_id,
            status=AuthorizationStatus.AUTHORIZED,
            accepted_defeat_path=None,
            considered_defeat_paths=(),
            supporting_evidence_ids=tuple(belief.source_evidence_ids),
            query_id=query_id,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )

    # ------------------------------------------------------------------
    # Step 1: SUPERSEDES
    # ------------------------------------------------------------------

    def _check_supersession(
        self,
        *,
        belief_id: str,
        as_of_time: str | None,
        as_of_evidence_id: str | None,
        trace_id: str,
        query_id: str | None,
    ) -> AuthorizationTrace | None:
        belief_edges = self.store.evidence_edges_for_belief(belief_id)
        super_edges = [e for e in belief_edges if e.edge_type == EvidenceEdgeType.SUPERSEDES]
        valid_super = self.temporal.edges_valid_at(
            super_edges,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )
        if not valid_super:
            return None
        latest = self.temporal.latest_edge(valid_super)
        assert latest is not None  # non-empty list guarantees a winner
        path = DefeatPath(
            path_id=f"path_super_{trace_id}",
            path_type=DefeatPathType.DIRECT_SUPERSEDE,
            target_belief_id=belief_id,
            supporting_dependency_edge_ids=(),
            supporting_evidence_edge_ids=(latest.edge_id,),
            replacement_belief_id=latest.replacement_belief_id,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )
        return AuthorizationTrace(
            trace_id=trace_id,
            belief_id=belief_id,
            status=AuthorizationStatus.SUPERSEDED,
            accepted_defeat_path=path,
            considered_defeat_paths=(path,),
            supporting_evidence_ids=(),
            query_id=query_id,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )

    # ------------------------------------------------------------------
    # Step 2: PREREQUISITE_BLOCK
    # ------------------------------------------------------------------

    def _check_prerequisite_block(
        self,
        *,
        belief_id: str,
        as_of_time: str | None,
        as_of_evidence_id: str | None,
        trace_id: str,
        query_id: str | None,
    ) -> AuthorizationTrace | None:
        dependencies = self.store.dependencies_of(belief_id)
        if not dependencies:
            return None

        edge_lookup: dict[str, EvidenceEdge] = {
            e.edge_id: e for e in self.store.all_evidence_edges()
        }
        blocking_paths: list[DefeatPath] = []

        for dep in dependencies:
            cond_edges = self.store.evidence_edges_for_condition(dep.condition_id)
            # Only BLOCKS / RELEASES updates participate in the latest-edge
            # contest for a condition. Other edge types cannot target a
            # condition under the schema, but we filter defensively.
            cond_updates = [
                e
                for e in cond_edges
                if e.edge_type in (EvidenceEdgeType.BLOCKS, EvidenceEdgeType.RELEASES)
            ]
            valid_updates = self.temporal.edges_valid_at(
                cond_updates,
                as_of_time=as_of_time,
                as_of_evidence_id=as_of_evidence_id,
            )
            latest = self.temporal.latest_edge(valid_updates)
            if latest is None:
                continue
            if latest.edge_type != EvidenceEdgeType.BLOCKS:
                # ``RELEASES`` clears the prior blocker; it does not
                # itself assert truth, but for this dependency there is
                # no active defeating evidence.
                continue
            blocking_paths.append(
                self._build_prerequisite_block_path(
                    dependency=dep,
                    blocking_edge=latest,
                    trace_id=trace_id,
                    as_of_time=as_of_time,
                    as_of_evidence_id=as_of_evidence_id,
                )
            )

        if not blocking_paths:
            return None

        chosen = self.temporal.latest_path(blocking_paths, edge_lookup)
        assert chosen is not None  # non-empty list guarantees a winner
        return AuthorizationTrace(
            trace_id=trace_id,
            belief_id=belief_id,
            status=AuthorizationStatus.BLOCKED,
            accepted_defeat_path=chosen,
            considered_defeat_paths=tuple(blocking_paths),
            supporting_evidence_ids=(),
            query_id=query_id,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )

    def _build_prerequisite_block_path(
        self,
        *,
        dependency: DependencyEdge,
        blocking_edge: EvidenceEdge,
        trace_id: str,
        as_of_time: str | None,
        as_of_evidence_id: str | None,
    ) -> DefeatPath:
        return DefeatPath(
            path_id=f"path_block_{trace_id}_{dependency.edge_id}",
            path_type=DefeatPathType.PREREQUISITE_BLOCK,
            target_belief_id=dependency.belief_id,
            supporting_dependency_edge_ids=(dependency.edge_id,),
            supporting_evidence_edge_ids=(blocking_edge.edge_id,),
            replacement_belief_id=None,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )

    # ------------------------------------------------------------------
    # Step 3: UNCERTAIN vs REAFFIRMS
    # ------------------------------------------------------------------

    def _check_belief_status(
        self,
        *,
        belief_id: str,
        as_of_time: str | None,
        as_of_evidence_id: str | None,
        trace_id: str,
        query_id: str | None,
    ) -> AuthorizationTrace | None:
        belief_edges = self.store.evidence_edges_for_belief(belief_id)
        status_edges = [
            e
            for e in belief_edges
            if e.edge_type in (EvidenceEdgeType.UNCERTAIN, EvidenceEdgeType.REAFFIRMS)
        ]
        valid_status = self.temporal.edges_valid_at(
            status_edges,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )
        latest = self.temporal.latest_edge(valid_status)
        if latest is None or latest.edge_type != EvidenceEdgeType.UNCERTAIN:
            # No status edge, or the latest one is REAFFIRMS (which clears
            # any prior UNCERTAIN). Either way, this stage does not defeat
            # the belief.
            return None
        path = DefeatPath(
            path_id=f"path_unresolved_{trace_id}",
            path_type=DefeatPathType.UNRESOLVED_UNCERTAIN,
            target_belief_id=belief_id,
            supporting_dependency_edge_ids=(),
            supporting_evidence_edge_ids=(latest.edge_id,),
            replacement_belief_id=None,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )
        return AuthorizationTrace(
            trace_id=trace_id,
            belief_id=belief_id,
            status=AuthorizationStatus.UNRESOLVED,
            accepted_defeat_path=path,
            considered_defeat_paths=(path,),
            supporting_evidence_ids=(),
            query_id=query_id,
            as_of_time=as_of_time,
            as_of_evidence_id=as_of_evidence_id,
        )

    # ------------------------------------------------------------------
    # Trace-id construction
    # ------------------------------------------------------------------

    @staticmethod
    def _trace_id(
        belief_id: str,
        as_of_time: str | None,
        as_of_evidence_id: str | None,
        query_id: str | None,
    ) -> str:
        """Deterministic, human-readable trace id.

        Format: ``trace::<belief>::<time>::<ev>::<query>`` with ``-`` as a
        placeholder for absent components. The same inputs always produce
        the same trace id, which keeps cassette-style test data stable.
        """
        return "trace::{b}::{t}::{e}::{q}".format(
            b=belief_id,
            t=as_of_time or "-",
            e=as_of_evidence_id or "-",
            q=query_id or "-",
        )


# Legacy import alias. ``mempatch.dpa.tms.__init__`` re-exports the name
# ``AuthorizationEngine`` so downstream packages keep importing the package
# without ImportError at module load time. The alias only preserves the *name*;
# the underlying behavior is the new typed-graph DPA.
AuthorizationEngine = DefeatPathAuthorizationAlgorithm
