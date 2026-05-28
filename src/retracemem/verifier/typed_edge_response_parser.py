from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from retracemem.schemas import BeliefNode, EvidenceEdge, EvidenceEdgeType, EvidenceNode

_CONDITION_EDGE_TYPES = {"BLOCKS", "RELEASES"}
_BELIEF_EDGE_TYPES = {"SUPERSEDES", "REAFFIRMS", "UNCERTAIN"}


@dataclass(frozen=True)
class EdgeTargetSpace:
    valid_belief_ids: frozenset[str]
    valid_condition_ids: frozenset[str]
    replacement_map: dict[str, BeliefNode]
    single_belief_id: str | None = None


def stable_edge_id(
    evidence_id: str,
    edge_type: str,
    target_id: str,
    replacement_belief_id: str | None,
    prompt_version: str,
) -> str:
    payload = f"{evidence_id}|{edge_type}|{target_id}|{replacement_belief_id or ''}|{prompt_version}"
    return f"ee-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def parse_typed_edge_response(
    response: str,
    *,
    new_evidence: EvidenceNode,
    target_space: EdgeTargetSpace,
    call_id: str,
    prompt_version: str,
) -> list[EvidenceEdge]:
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    data = json.loads(text)
    if not isinstance(data, dict) or "edges" not in data:
        raise ValueError("Invalid evidence-edge response: missing 'edges' key")

    seen_edge_ids: set[str] = set()
    edges: list[EvidenceEdge] = []
    for item in data["edges"]:
        edge_type_str = item.get("edge_type", "").upper()
        target_id = item.get("target_id")
        replacement_id = item.get("replacement_belief_id")

        if not target_id:
            raise ValueError(f"Edge item missing target_id: {item}")

        try:
            edge_type = EvidenceEdgeType(edge_type_str)
        except ValueError:
            raise ValueError(f"Unknown edge_type '{edge_type_str}': {item}")

        if edge_type_str in _CONDITION_EDGE_TYPES:
            if target_id not in target_space.valid_condition_ids:
                raise ValueError(
                    f"{edge_type_str} edge targets unknown condition '{target_id}'. "
                    f"Valid: {set(target_space.valid_condition_ids)}"
                )
            target_kind = "condition"
        elif edge_type_str in _BELIEF_EDGE_TYPES:
            if target_space.single_belief_id is not None and target_id != target_space.single_belief_id:
                raise ValueError(
                    f"{edge_type_str} edge must target candidate belief "
                    f"'{target_space.single_belief_id}', got '{target_id}'"
                )
            if target_id not in target_space.valid_belief_ids:
                raise ValueError(
                    f"{edge_type_str} edge must target a candidate belief id "
                    f"from {set(target_space.valid_belief_ids)}, got '{target_id}'"
                )
            target_kind = "belief"
        else:
            raise ValueError(f"Unhandled edge_type '{edge_type_str}'")

        if edge_type == EvidenceEdgeType.SUPERSEDES:
            if not replacement_id:
                raise ValueError(f"SUPERSEDES edge missing replacement_belief_id: {item}")
            if replacement_id not in target_space.replacement_map:
                raise ValueError(
                    f"SUPERSEDES edge references unknown replacement belief "
                    f"'{replacement_id}'. Valid: {set(target_space.replacement_map.keys())}"
                )
            replacement_belief = target_space.replacement_map[replacement_id]
            if new_evidence.evidence_id not in replacement_belief.source_evidence_ids:
                raise ValueError(
                    f"SUPERSEDES replacement '{replacement_id}' is not grounded in "
                    f"current evidence '{new_evidence.evidence_id}'. "
                    f"Replacement source_evidence_ids: {replacement_belief.source_evidence_ids}"
                )

        confidence = item.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                raise ValueError(
                    f"Confidence must be a number in [0.0, 1.0], got {confidence!r}: {item}"
                )

        edge_id = stable_edge_id(
            new_evidence.evidence_id,
            edge_type_str,
            target_id,
            replacement_id,
            prompt_version,
        )
        if edge_id in seen_edge_ids:
            raise ValueError(f"Duplicate edge in response: {edge_type_str} -> {target_id}")
        seen_edge_ids.add(edge_id)

        edges.append(EvidenceEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            evidence_id=new_evidence.evidence_id,
            target_kind=target_kind,
            target_id=target_id,
            verifier=prompt_version,
            replacement_belief_id=replacement_id,
            confidence=confidence,
            rationale=item.get("rationale"),
            model_call_trace_id=call_id,
        ))

    return edges
