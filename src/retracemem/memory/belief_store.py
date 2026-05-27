from __future__ import annotations

from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
)


class BeliefStore:
    """Typed graph store for the Defeat-Path Authorization runtime (Wave 1A).

    Holds four indexed collections per the refactor plan amendment A6:

    - ``beliefs``           : ``BeliefNode`` keyed by ``belief_id``.
    - ``conditions``        : ``ConditionNode`` keyed by ``condition_id``.
    - ``dependency_edges``  : ``DependencyEdge`` (REQUIRES) keyed by ``edge_id``.
    - ``evidence_edges``    : ``EvidenceEdge`` (BLOCKS / RELEASES /
      SUPERSEDES / REAFFIRMS / UNCERTAIN) keyed by ``edge_id``.

    No flat ``RelationPrediction`` is stored. The legacy
    ``add_relation`` / ``relations_for_belief`` / ``all_relations`` API is
    deliberately removed; callers that still use it will fail loudly at
    runtime, which is the intended transitional breakage on the
    ``refactor/defeat-path-core`` branch.

    Important: ``AUTHORIZED`` here means a belief is **eligible** to
    participate in the current authorized basis. The store does not assert
    that any belief is currently true; the typed graph only records evidence
    and the structural dependencies among beliefs and conditions.
    """

    def __init__(self) -> None:
        # Primary indexes.
        self._beliefs: dict[str, BeliefNode] = {}
        self._conditions: dict[str, ConditionNode] = {}
        self._dependency_edges: dict[str, DependencyEdge] = {}
        self._evidence_edges: dict[str, EvidenceEdge] = {}

        # Secondary indexes maintained alongside the primary collections so
        # that DPA can run in time proportional to the number of edges that
        # actually touch the queried belief instead of scanning the whole
        # graph on every authorization.
        self._dep_edges_by_belief: dict[str, list[str]] = {}
        self._dep_edges_by_condition: dict[str, list[str]] = {}
        self._ev_edges_by_condition: dict[str, list[str]] = {}
        self._ev_edges_by_belief: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # BeliefNode operations
    # ------------------------------------------------------------------

    def add_belief(self, belief: BeliefNode) -> None:
        if not belief.belief_id:
            raise ValueError("belief_id is required")
        if not belief.proposition:
            raise ValueError("belief proposition is required")
        if belief.belief_id in self._beliefs:
            raise ValueError(f"belief already exists: {belief.belief_id}")
        self._beliefs[belief.belief_id] = belief

    def get_belief(self, belief_id: str) -> BeliefNode:
        if belief_id not in self._beliefs:
            raise KeyError(f"unknown belief: {belief_id}")
        return self._beliefs[belief_id]

    def has_belief(self, belief_id: str) -> bool:
        return belief_id in self._beliefs

    def all_beliefs(self) -> list[BeliefNode]:
        return list(self._beliefs.values())

    # ------------------------------------------------------------------
    # ConditionNode operations
    # ------------------------------------------------------------------

    def add_condition(self, condition: ConditionNode) -> None:
        if not condition.condition_id:
            raise ValueError("condition_id is required")
        if not condition.scope_id:
            # Amendment A7: identity is namespaced by scope_id; an empty
            # scope_id would silently merge across users.
            raise ValueError("ConditionNode requires a non-empty scope_id (amendment A7)")
        if not condition.text:
            raise ValueError("ConditionNode requires non-empty text")
        if condition.condition_id in self._conditions:
            raise ValueError(f"condition already exists: {condition.condition_id}")
        self._conditions[condition.condition_id] = condition

    def get_condition(self, condition_id: str) -> ConditionNode:
        if condition_id not in self._conditions:
            raise KeyError(f"unknown condition: {condition_id}")
        return self._conditions[condition_id]

    def has_condition(self, condition_id: str) -> bool:
        return condition_id in self._conditions

    def all_conditions(self) -> list[ConditionNode]:
        return list(self._conditions.values())

    # ------------------------------------------------------------------
    # DependencyEdge operations (REQUIRES)
    # ------------------------------------------------------------------

    def add_dependency_edge(self, edge: DependencyEdge) -> None:
        if not edge.edge_id:
            raise ValueError("dependency edge_id is required")
        if edge.edge_id in self._dependency_edges:
            raise ValueError(f"dependency edge already exists: {edge.edge_id}")
        if edge.edge_type != "REQUIRES":
            # The gate is the sole authority on edge-type policy, but the
            # store enforces the schema-level invariant from amendment A2.
            raise ValueError(
                f"DependencyEdge.edge_type must be 'REQUIRES' (got {edge.edge_type!r})"
            )
        if edge.belief_id not in self._beliefs:
            raise KeyError(f"unknown belief on dependency edge: {edge.belief_id}")
        if edge.condition_id not in self._conditions:
            raise KeyError(f"unknown condition on dependency edge: {edge.condition_id}")
        self._dependency_edges[edge.edge_id] = edge
        self._dep_edges_by_belief.setdefault(edge.belief_id, []).append(edge.edge_id)
        self._dep_edges_by_condition.setdefault(edge.condition_id, []).append(edge.edge_id)

    def get_dependency_edge(self, edge_id: str) -> DependencyEdge:
        if edge_id not in self._dependency_edges:
            raise KeyError(f"unknown dependency edge: {edge_id}")
        return self._dependency_edges[edge_id]

    def has_dependency_edge(self, edge_id: str) -> bool:
        return edge_id in self._dependency_edges

    def all_dependency_edges(self) -> list[DependencyEdge]:
        return list(self._dependency_edges.values())

    # ------------------------------------------------------------------
    # EvidenceEdge operations
    # ------------------------------------------------------------------

    def add_evidence_edge(self, edge: EvidenceEdge) -> None:
        if not edge.edge_id:
            raise ValueError("evidence edge_id is required")
        if edge.edge_id in self._evidence_edges:
            raise ValueError(f"evidence edge already exists: {edge.edge_id}")
        if edge.target_kind == "condition":
            if edge.target_id not in self._conditions:
                raise KeyError(
                    f"evidence edge targets unknown condition: {edge.target_id}"
                )
            self._ev_edges_by_condition.setdefault(edge.target_id, []).append(edge.edge_id)
        elif edge.target_kind == "belief":
            if edge.target_id not in self._beliefs:
                raise KeyError(
                    f"evidence edge targets unknown belief: {edge.target_id}"
                )
            self._ev_edges_by_belief.setdefault(edge.target_id, []).append(edge.edge_id)
        else:
            raise ValueError(
                f"EvidenceEdge.target_kind must be 'condition' or 'belief' "
                f"(got {edge.target_kind!r})"
            )
        self._evidence_edges[edge.edge_id] = edge

    def get_evidence_edge(self, edge_id: str) -> EvidenceEdge:
        if edge_id not in self._evidence_edges:
            raise KeyError(f"unknown evidence edge: {edge_id}")
        return self._evidence_edges[edge_id]

    def has_evidence_edge(self, edge_id: str) -> bool:
        return edge_id in self._evidence_edges

    def all_evidence_edges(self) -> list[EvidenceEdge]:
        return list(self._evidence_edges.values())

    # ------------------------------------------------------------------
    # Indexed accessors required by Wave 1A spec
    # ------------------------------------------------------------------

    def dependencies_of(self, belief_id: str) -> list[DependencyEdge]:
        """Accepted ``REQUIRES`` dependency edges anchored at ``belief_id``."""
        edge_ids = self._dep_edges_by_belief.get(belief_id, ())
        return [self._dependency_edges[eid] for eid in edge_ids]

    def evidence_edges_for_condition(self, condition_id: str) -> list[EvidenceEdge]:
        """Evidence edges (BLOCKS / RELEASES) targeting ``condition_id``."""
        edge_ids = self._ev_edges_by_condition.get(condition_id, ())
        return [self._evidence_edges[eid] for eid in edge_ids]

    def evidence_edges_for_belief(self, belief_id: str) -> list[EvidenceEdge]:
        """Evidence edges (SUPERSEDES / REAFFIRMS / UNCERTAIN) targeting ``belief_id``."""
        edge_ids = self._ev_edges_by_belief.get(belief_id, ())
        return [self._evidence_edges[eid] for eid in edge_ids]

    def supporting_evidence_for_belief(self, belief_id: str) -> tuple[str, ...]:
        """The originating evidence ids recorded on the belief node itself.

        These are the immutable provenance pointers established when the
        belief was extracted; they are not modified by later evidence
        updates.
        """
        return tuple(self._beliefs[belief_id].source_evidence_ids)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._beliefs)
