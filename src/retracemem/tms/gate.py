from __future__ import annotations

from dataclasses import dataclass

from retracemem.memory.belief_store import BeliefStore
from retracemem.schemas import (
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
)


@dataclass(frozen=True)
class GateDecision:
    """Outcome of structural admission for a single edge.

    The gate never decides authorization. It only certifies that an edge is
    well-formed with respect to the typed graph schema in
    ``retracemem.schemas`` and to the current contents of ``BeliefStore``.

    ``reason`` is a stable machine-readable token (snake_case) so that
    tests and downstream tooling can assert on it without parsing prose.
    """

    admitted: bool
    reason: str


# Lookup of allowed (edge_type, target_kind) pairs for evidence edges.
# This makes the BLOCKS/RELEASES-target-condition and
# SUPERSEDES/REAFFIRMS/UNCERTAIN-target-belief constraint declarative
# rather than scattered through ``if`` branches.
_EVIDENCE_TARGET_KIND: dict[EvidenceEdgeType, str] = {
    EvidenceEdgeType.BLOCKS: "condition",
    EvidenceEdgeType.RELEASES: "condition",
    EvidenceEdgeType.SUPERSEDES: "belief",
    EvidenceEdgeType.REAFFIRMS: "belief",
    EvidenceEdgeType.UNCERTAIN: "belief",
}


class RevisionGate:
    """Structural admission gate for typed graph edges.

    The gate validates *structural well-formedness* of a candidate
    ``DependencyEdge`` or ``EvidenceEdge`` against the typed-graph schema
    and against the current ``BeliefStore`` membership. It deliberately
    does **not** judge defeat semantics; that is the responsibility of
    ``DefeatPathAuthorizationAlgorithm``.

    Important distinction:

      A well-formed ``BLOCKS(e, c)`` edge may exist in the graph but it
      does not block arbitrary beliefs. It defeats belief ``b`` only
      through an accepted ``REQUIRES(b, c)`` dependency edge. The gate
      admits the BLOCKS edge as structurally valid; DPA enforces the
      anchoring requirement.

      ``SUPERSEDES`` edges must carry a
    ``replacement_belief_id`` that exists in the store.

      ``REQUIRES`` is the only allowed
    ``DependencyEdge.edge_type``.
    """

    # ------------------------------------------------------------------
    # DependencyEdge admission
    # ------------------------------------------------------------------

    def admit_dependency_edge(
        self,
        edge: DependencyEdge,
        store: BeliefStore,
    ) -> GateDecision:
        if not edge.edge_id:
            return GateDecision(False, "empty_edge_id")
        if edge.edge_type != "REQUIRES":
            return GateDecision(False, "dependency_edge_type_not_requires")
        if not edge.belief_id:
            return GateDecision(False, "missing_belief_id")
        if not edge.condition_id:
            return GateDecision(False, "missing_condition_id")
        if not store.has_belief(edge.belief_id):
            return GateDecision(False, "unknown_belief")
        if not store.has_condition(edge.condition_id):
            return GateDecision(False, "unknown_condition")
        if not edge.inducer:
            # Provenance is first-class. An anonymous
            # dependency edge cannot enter the graph.
            return GateDecision(False, "missing_inducer_provenance")
        return GateDecision(True, "ok")

    # ------------------------------------------------------------------
    # EvidenceEdge admission
    # ------------------------------------------------------------------

    def admit_evidence_edge(
        self,
        edge: EvidenceEdge,
        store: BeliefStore,
    ) -> GateDecision:
        if not edge.edge_id:
            return GateDecision(False, "empty_edge_id")
        if not edge.evidence_id:
            return GateDecision(False, "missing_evidence_id")
        if not edge.verifier:
            # Provenance is first-class.
            return GateDecision(False, "missing_verifier_provenance")

        expected_kind = _EVIDENCE_TARGET_KIND.get(edge.edge_type)
        if expected_kind is None:
            return GateDecision(False, "unknown_evidence_edge_type")
        if edge.target_kind != expected_kind:
            return GateDecision(False, f"target_kind_mismatch_for_{edge.edge_type.value.lower()}")

        if edge.target_kind == "condition":
            if not store.has_condition(edge.target_id):
                return GateDecision(False, "unknown_condition_target")
        elif edge.target_kind == "belief":
            if not store.has_belief(edge.target_id):
                return GateDecision(False, "unknown_belief_target")

        # SUPERSEDES must carry a replacement that exists.
        if edge.edge_type == EvidenceEdgeType.SUPERSEDES:
            if edge.replacement_belief_id is None:
                return GateDecision(False, "supersedes_missing_replacement_belief_id")
            if edge.replacement_belief_id == edge.target_id:
                return GateDecision(False, "supersedes_replacement_equals_target")
            if not store.has_belief(edge.replacement_belief_id):
                return GateDecision(False, "supersedes_unknown_replacement_belief")
        else:
            # For non-SUPERSEDES edges, ``replacement_belief_id`` must be
            # absent; carrying one would be a category error per the
            # schema annotation.
            if edge.replacement_belief_id is not None:
                return GateDecision(False, "replacement_belief_id_only_valid_for_supersedes")

        return GateDecision(True, "ok")
