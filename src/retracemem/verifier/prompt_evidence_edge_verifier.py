"""Prompt-based evidence-edge verifier for Stage A ReTrace-LLM.

Implements the ``EvidenceEdgeVerifier`` protocol using ``CachedLLMClient``.
All model interactions go through the replay-safe cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from retracemem.methods.contracts import EdgePredictionBatch
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "prompts", "retrace_llm")
_PROMPT_FILE = os.path.join(_PROMPT_DIR, "evidence_edge_prediction_v0.txt")
_PROMPT_VERSION = "evidence_edge_prediction_v0"
_RESPONSE_SCHEMA_VERSION = "evidence_edge_prediction_response_v0"
_PARSER_VERSION = "evidence_edge_prediction_parser_v0"

_CONDITION_EDGE_TYPES = {"BLOCKS", "RELEASES"}
_BELIEF_EDGE_TYPES = {"SUPERSEDES", "REAFFIRMS", "UNCERTAIN"}


def _load_prompt_template() -> str:
    path = os.path.normpath(_PROMPT_FILE)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _stable_edge_id(
    evidence_id: str,
    edge_type: str,
    target_id: str,
    replacement_belief_id: str | None,
) -> str:
    """Compute deterministic evidence edge_id from grounded inputs."""
    payload = f"{evidence_id}|{edge_type}|{target_id}|{replacement_belief_id or ''}|{_PROMPT_VERSION}"
    return f"ee-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


class PromptEvidenceEdgeVerifier:
    """Stage A evidence-edge prediction via LLM prompt and structured JSON parse.

    Implements ``EvidenceEdgeVerifier.verify_edges()``.
    """

    def __init__(
        self,
        client: CachedLLMClient,
        model_id: str = "gemini-pro",
        provider: str = "google",
        model_revision_or_api_version: str | None = None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.model_revision_or_api_version = model_revision_or_api_version
        self._template = _load_prompt_template()
        self._template_hash = hashlib.sha256(self._template.encode("utf-8")).hexdigest()

    def verify_edges(
        self,
        new_evidence: EvidenceNode,
        candidate_belief: BeliefNode,
        candidate_replacement_beliefs: tuple[BeliefNode, ...],
        candidate_conditions: tuple[ConditionNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> list[EvidenceEdge]:
        """Backward-compatible interface returning only proposed edges."""
        batch = self.verify_edges_with_trace(
            new_evidence=new_evidence,
            candidate_belief=candidate_belief,
            candidate_replacement_beliefs=candidate_replacement_beliefs,
            candidate_conditions=candidate_conditions,
            temporal_context=temporal_context,
        )
        return list(batch.proposed_edges)

    def verify_edges_with_trace(
        self,
        new_evidence: EvidenceNode,
        candidate_belief: BeliefNode,
        candidate_replacement_beliefs: tuple[BeliefNode, ...],
        candidate_conditions: tuple[ConditionNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> EdgePredictionBatch:
        """Traced verifier method preserving model_call_trace_id even for zero edges."""
        replacements_str = "\n".join(
            f"  - {b.belief_id}: \"{b.proposition}\""
            for b in candidate_replacement_beliefs
        ) or "  (none)"
        conditions_str = "\n".join(
            f"  - {c.condition_id}: \"{c.text}\""
            for c in candidate_conditions
        ) or "  (none)"

        prompt = self._template.replace("{evidence_text}", new_evidence.text)
        prompt = prompt.replace("{belief_proposition}", candidate_belief.proposition)
        prompt = prompt.replace("{belief_id}", candidate_belief.belief_id)
        prompt = prompt.replace("{replacement_beliefs}", replacements_str)
        prompt = prompt.replace("{candidate_conditions}", conditions_str)

        temporal_hash = hashlib.sha256(
            "|".join(e.evidence_id for e in temporal_context).encode("utf-8")
        ).hexdigest()

        trace = self.client.generate(
            prompt=prompt,
            model_id=self.model_id,
            provider=self.provider,
            prompt_template_hash=self._template_hash,
            response_schema_version=_RESPONSE_SCHEMA_VERSION,
            parser_version=_PARSER_VERSION,
            temperature=0.0,
            temporal_context_hash=temporal_hash,
            metadata={
                "evidence_id": new_evidence.evidence_id,
                "belief_id": candidate_belief.belief_id,
            },
        )

        if trace.status != "success" or not trace.response:
            raise ValueError(
                f"Evidence-edge prediction failed: status={trace.status}, "
                f"error={trace.error_message}"
            )

        replacement_map = {b.belief_id: b for b in candidate_replacement_beliefs}
        valid_condition_ids = {c.condition_id for c in candidate_conditions}

        edges = self._parse(
            trace.response,
            new_evidence,
            candidate_belief,
            replacement_map,
            valid_condition_ids,
            trace.call_id,
        )

        return EdgePredictionBatch(
            proposed_edges=tuple(edges),
            model_call_trace_id=trace.call_id,
            prompt_version=_PROMPT_VERSION,
            model_id=self.model_id,
            provider=self.provider,
            model_revision_or_api_version=self.model_revision_or_api_version,
        )

    def _parse(
        self,
        response: str,
        new_evidence: EvidenceNode,
        candidate_belief: BeliefNode,
        replacement_map: dict[str, BeliefNode],
        valid_condition_ids: set[str],
        call_id: str,
    ) -> list[EvidenceEdge]:
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        if not isinstance(data, dict) or "edges" not in data:
            raise ValueError(f"Invalid evidence-edge response: missing 'edges' key")

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
                if target_id not in valid_condition_ids:
                    raise ValueError(
                        f"{edge_type_str} edge targets unknown condition '{target_id}'. "
                        f"Valid: {valid_condition_ids}"
                    )
                target_kind = "condition"
            elif edge_type_str in _BELIEF_EDGE_TYPES:
                if target_id != candidate_belief.belief_id:
                    raise ValueError(
                        f"{edge_type_str} edge must target candidate belief "
                        f"'{candidate_belief.belief_id}', got '{target_id}'"
                    )
                target_kind = "belief"
            else:
                raise ValueError(f"Unhandled edge_type '{edge_type_str}'")

            if edge_type == EvidenceEdgeType.SUPERSEDES:
                if not replacement_id:
                    raise ValueError(f"SUPERSEDES edge missing replacement_belief_id: {item}")
                if replacement_id not in replacement_map:
                    raise ValueError(
                        f"SUPERSEDES edge references unknown replacement belief "
                        f"'{replacement_id}'. Valid: {set(replacement_map.keys())}"
                    )
                replacement_belief = replacement_map[replacement_id]
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

            edge_id = _stable_edge_id(
                new_evidence.evidence_id, edge_type_str, target_id, replacement_id,
            )
            if edge_id in seen_edge_ids:
                raise ValueError(
                    f"Duplicate edge in response: {edge_type_str} -> {target_id}"
                )
            seen_edge_ids.add(edge_id)

            edge = EvidenceEdge(
                edge_id=edge_id,
                edge_type=edge_type,
                evidence_id=new_evidence.evidence_id,
                target_kind=target_kind,
                target_id=target_id,
                verifier=_PROMPT_VERSION,
                replacement_belief_id=replacement_id,
                confidence=confidence,
                rationale=item.get("rationale"),
                model_call_trace_id=call_id,
            )
            edges.append(edge)

        return edges
