from __future__ import annotations

import re

from typing import Iterable

from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)


_TOPIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("mobility", "bike", "bikes", "bicycle", "bicycles", "cycling", "commute", "leg", "cast"),
    ("availability", "available", "schedule", "call", "calls", "meeting", "meetings"),
    ("address", "location", "lives", "moved", "relocated", "delivery", "office", "email", "contact"),
)


class ManualEvidenceEdgeVerifier:
    """Return pre-specified typed evidence edges for development fixtures."""

    name = "manual_evidence_edge_verifier"

    def __init__(
        self,
        edges: tuple[EvidenceEdge, ...]
        | dict[tuple[str, str], Iterable[EvidenceEdge]]
        | None = None,
    ) -> None:
        self._edges_by_evidence_and_belief: dict[tuple[str, str], list[EvidenceEdge]] = {}
        if isinstance(edges, dict):
            for (_evidence_id, belief_id), fixture_edges in edges.items():
                for edge in fixture_edges:
                    self.register(edge, belief_id=belief_id)
        else:
            for edge in edges or ():
                self.register(edge)

    def register(self, edge: EvidenceEdge, belief_id: str | None = None) -> None:
        _validate_edge(edge)
        key = (edge.evidence_id, belief_id or edge.target_id)
        self._edges_by_evidence_and_belief.setdefault(key, []).append(edge)

    def verify_edges(
        self,
        new_evidence: EvidenceNode,
        candidate_belief: BeliefNode,
        candidate_conditions: tuple[ConditionNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> list[EvidenceEdge]:
        direct_key = (new_evidence.evidence_id, candidate_belief.belief_id)
        condition_ids = tuple(condition.condition_id for condition in candidate_conditions)
        temporal_ids = tuple(evidence.evidence_id for evidence in temporal_context)
        return [
            self._with_context_metadata(edge, condition_ids, temporal_ids)
            for edge in self._edges_by_evidence_and_belief.get(direct_key, ())
        ]

    @staticmethod
    def _with_context_metadata(
        edge: EvidenceEdge,
        condition_ids: tuple[str, ...],
        temporal_ids: tuple[str, ...],
    ) -> EvidenceEdge:
        from dataclasses import replace

        return replace(
            edge,
            metadata={
                **dict(edge.metadata),
                "candidate_condition_ids": condition_ids,
                "temporal_context_ids": temporal_ids,
            },
        )


class HeuristicEvidenceEdgeVerifier:
    """Deterministic offline typed edge verifier for Wave 1B development."""

    name = "heuristic_evidence_edge_verifier"

    _BLOCK_TERMS = (
        "broke",
        "broken",
        "cast",
        "injury",
        "injured",
        "surgery",
        "cannot",
        "can't",
        "must avoid",
        "suspended",
    )
    _RELEASE_TERMS = (
        "cleared",
        "recovered",
        "recovery complete",
        "cast removed",
        "can resume",
        "allowed to resume",
        "no longer restricted",
    )
    _SUPERSEDE_TERMS = (
        "moved to",
        "relocated to",
        "now lives",
        "now uses",
        "changed",
        "replaced",
        "new primary",
        "preferred delivery location to",
    )
    _REAFFIRM_TERMS = ("still", "continues to", "continue to", "remains", "again")
    _UNCERTAIN_TERMS = (
        "not sure",
        "unclear",
        "maybe",
        "might have changed",
        "may have changed",
        "not confirmed",
        "no details",
    )
    _LOCATION_TERMS = ("address", "lives", "office", "delivery", "location", "email", "contact")

    def verify_edges(
        self,
        new_evidence: EvidenceNode,
        candidate_belief: BeliefNode,
        candidate_conditions: tuple[ConditionNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> list[EvidenceEdge]:
        evidence_text = _normalize(new_evidence.text)
        belief_text = _normalize(candidate_belief.proposition)
        condition_matches = self._matching_conditions(evidence_text, belief_text, candidate_conditions)
        temporal_ids = tuple(e.evidence_id for e in temporal_context)
        candidate_condition_ids = tuple(condition.condition_id for condition in candidate_conditions)

        edges: list[EvidenceEdge] = []

        if condition_matches and _contains_any(evidence_text, self._RELEASE_TERMS):
            if self._temporal_context_mentions_prior_block(temporal_context):
                for condition in condition_matches:
                    edges.append(
                        self._condition_edge(
                            edge_type=EvidenceEdgeType.RELEASES,
                            evidence=new_evidence,
                            condition=condition,
                            confidence=0.76,
                            rationale="Evidence releases a previously blocked candidate condition.",
                            temporal_ids=temporal_ids,
                            candidate_condition_ids=candidate_condition_ids,
                        )
                    )
            return edges

        if condition_matches and _contains_any(evidence_text, self._BLOCK_TERMS):
            for condition in condition_matches:
                edges.append(
                    self._condition_edge(
                        edge_type=EvidenceEdgeType.BLOCKS,
                        evidence=new_evidence,
                        condition=condition,
                        confidence=0.78,
                        rationale="Evidence blocks a candidate condition required by the belief.",
                        temporal_ids=temporal_ids,
                        candidate_condition_ids=candidate_condition_ids,
                    )
                )
            return edges

        if self._looks_like_supersession(evidence_text, belief_text):
            return [
                EvidenceEdge(
                    edge_id=f"ev:{new_evidence.evidence_id}:SUPERSEDES:{candidate_belief.belief_id}",
                    edge_type=EvidenceEdgeType.SUPERSEDES,
                    evidence_id=new_evidence.evidence_id,
                    target_kind="belief",
                    target_id=candidate_belief.belief_id,
                    verifier=self.name,
                    replacement_belief_id=f"replacement:{candidate_belief.belief_id}:{new_evidence.evidence_id}",
                    confidence=0.75,
                    rationale="Evidence appears to replace the prior belief.",
                    span=_short_span(new_evidence.text),
                    metadata={
                        "candidate_condition_ids": candidate_condition_ids,
                        "temporal_context_ids": temporal_ids,
                    },
                )
            ]

        if _contains_any(evidence_text, self._UNCERTAIN_TERMS) and _topic_overlap(evidence_text, belief_text):
            return [
                EvidenceEdge(
                    edge_id=f"ev:{new_evidence.evidence_id}:UNCERTAIN:{candidate_belief.belief_id}",
                    edge_type=EvidenceEdgeType.UNCERTAIN,
                    evidence_id=new_evidence.evidence_id,
                    target_kind="belief",
                    target_id=candidate_belief.belief_id,
                    verifier=self.name,
                    confidence=0.62,
                    rationale="Evidence makes the candidate belief uncertain without a replacement.",
                    span=_short_span(new_evidence.text),
                    metadata={
                        "candidate_condition_ids": candidate_condition_ids,
                        "temporal_context_ids": temporal_ids,
                    },
                )
            ]

        if _contains_any(evidence_text, self._REAFFIRM_TERMS) and _topic_overlap(evidence_text, belief_text):
            return [
                EvidenceEdge(
                    edge_id=f"ev:{new_evidence.evidence_id}:REAFFIRMS:{candidate_belief.belief_id}",
                    edge_type=EvidenceEdgeType.REAFFIRMS,
                    evidence_id=new_evidence.evidence_id,
                    target_kind="belief",
                    target_id=candidate_belief.belief_id,
                    verifier=self.name,
                    confidence=0.68,
                    rationale="Evidence reaffirms the candidate belief.",
                    span=_short_span(new_evidence.text),
                    metadata={
                        "candidate_condition_ids": candidate_condition_ids,
                        "temporal_context_ids": temporal_ids,
                    },
                )
            ]

        return []

    def _matching_conditions(
        self,
        evidence_text: str,
        belief_text: str,
        candidate_conditions: tuple[ConditionNode, ...],
    ) -> tuple[ConditionNode, ...]:
        matches: list[ConditionNode] = []
        for condition in candidate_conditions:
            condition_text = _normalize(condition.text)
            if _topic_overlap(evidence_text, condition_text) or _topic_overlap(belief_text, condition_text):
                matches.append(condition)
        return tuple(matches)

    @staticmethod
    def _temporal_context_mentions_prior_block(temporal_context: tuple[EvidenceNode, ...]) -> bool:
        if not temporal_context:
            return False
        return any(
            _contains_any(_normalize(evidence.text), HeuristicEvidenceEdgeVerifier._BLOCK_TERMS)
            or evidence.metadata.get("edge_type") == EvidenceEdgeType.BLOCKS.value
            for evidence in temporal_context
        )

    def _looks_like_supersession(self, evidence_text: str, belief_text: str) -> bool:
        return _contains_any(evidence_text, self._SUPERSEDE_TERMS) and (
            _topic_overlap(evidence_text, belief_text)
            or _contains_any(belief_text, self._LOCATION_TERMS)
        )

    def _condition_edge(
        self,
        *,
        edge_type: EvidenceEdgeType,
        evidence: EvidenceNode,
        condition: ConditionNode,
        confidence: float,
        rationale: str,
        temporal_ids: tuple[str, ...],
        candidate_condition_ids: tuple[str, ...],
    ) -> EvidenceEdge:
        return EvidenceEdge(
            edge_id=f"ev:{evidence.evidence_id}:{edge_type.value}:{condition.condition_id}",
            edge_type=edge_type,
            evidence_id=evidence.evidence_id,
            target_kind="condition",
            target_id=condition.condition_id,
            verifier=self.name,
            confidence=confidence,
            rationale=rationale,
            span=_short_span(evidence.text),
            metadata={
                "candidate_condition_id": condition.condition_id,
                "candidate_condition_ids": candidate_condition_ids,
                "temporal_context_ids": temporal_ids,
            },
        )


def _validate_edge(edge: EvidenceEdge) -> None:
    if edge.edge_type in (EvidenceEdgeType.BLOCKS, EvidenceEdgeType.RELEASES):
        if edge.target_kind != "condition":
            raise ValueError(f"{edge.edge_type.value} must target a condition")
    elif edge.edge_type == EvidenceEdgeType.SUPERSEDES:
        if edge.target_kind != "belief":
            raise ValueError("SUPERSEDES must target a belief")
        if not edge.replacement_belief_id:
            raise ValueError("SUPERSEDES requires replacement_belief_id")
    elif edge.edge_type in (EvidenceEdgeType.REAFFIRMS, EvidenceEdgeType.UNCERTAIN):
        if edge.target_kind != "belief":
            raise ValueError(f"{edge.edge_type.value} must target a belief")
    else:
        raise ValueError(f"unsupported evidence edge type: {edge.edge_type!r}")


def _normalize(text: str) -> str:
    return f" {re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()} "


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "user",
        "their",
        "them",
        "this",
        "that",
        "with",
        "from",
        "will",
        "can",
        "for",
        "and",
        "are",
        "has",
        "have",
        "had",
        "was",
        "were",
        "to",
        "by",
        "in",
        "on",
        "at",
        "as",
        "be",
        "is",
    }
    return {term for term in text.split() if len(term) > 3 and term not in stopwords}


def _topic_overlap(left: str, right: str) -> bool:
    if _content_terms(left) & _content_terms(right):
        return True
    return any(_contains_any(left, group) and _contains_any(right, group) for group in _TOPIC_GROUPS)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _short_span(text: str) -> str:
    return text.strip()[:160]
