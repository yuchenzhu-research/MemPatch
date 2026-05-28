"""Batched prompt-based evidence-edge verifier for Stage A ReTrace-LLM.

Renders all candidate beliefs in a single prompt instead of one call per belief.
Returns the same ``EdgePredictionBatch`` contract so downstream RevisionGate
and DPA semantics are unchanged.

The per-belief ``PromptEvidenceEdgeVerifier`` remains the auditable controlled
reference implementation and is NOT modified.
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
_RESPONSE_SCHEMA_VERSION = "evidence_edge_prediction_batch_response_v1"
_PARSER_VERSION = "evidence_edge_prediction_batch_parser_v1"

_CONDITION_EDGE_TYPES = {"BLOCKS", "RELEASES"}
_BELIEF_EDGE_TYPES = {"SUPERSEDES", "REAFFIRMS", "UNCERTAIN"}


def _prompt_file(prompt_version: str) -> str:
    if prompt_version != "evidence_edge_prediction_batch_v1":
        raise ValueError(f"Unsupported batched prompt version: {prompt_version}")
    return os.path.join(_PROMPT_DIR, f"{prompt_version}.txt")


def _load_prompt_template(prompt_version: str) -> str:
    path = os.path.normpath(_prompt_file(prompt_version))
    with open(path, encoding="utf-8") as f:
        return f.read()


def _stable_edge_id(
    evidence_id: str,
    edge_type: str,
    target_id: str,
    replacement_belief_id: str | None,
    prompt_version: str,
) -> str:
    payload = f"{evidence_id}|{edge_type}|{target_id}|{replacement_belief_id or ''}|{prompt_version}"
    return f"ee-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


class PromptBatchedEvidenceEdgeVerifier:
    """Batched Stage A evidence-edge prediction: one LLM call for all beliefs."""

    def __init__(
        self,
        client: CachedLLMClient,
        model_id: str = "gemini-pro",
        provider: str = "google",
        model_revision_or_api_version: str | None = None,
        prompt_version: str = "evidence_edge_prediction_batch_v1",
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.model_revision_or_api_version = model_revision_or_api_version
        self.prompt_version = prompt_version
        self._template = _load_prompt_template(prompt_version)
        self._template_hash = hashlib.sha256(self._template.encode("utf-8")).hexdigest()

    def verify_edges_batch(
        self,
        new_evidence: EvidenceNode,
        candidate_beliefs: tuple[BeliefNode, ...],
        candidate_replacement_beliefs: tuple[BeliefNode, ...],
        candidate_conditions_by_belief: tuple[tuple[str, tuple[ConditionNode, ...]], ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> EdgePredictionBatch:
        """Single-call batched verifier over all candidate beliefs."""
        beliefs_str = "\n".join(
            f"  - {b.belief_id}: \"{b.proposition}\""
            for b in candidate_beliefs
        ) or "  (none)"

        replacements_str = "\n".join(
            f"  - {b.belief_id}: \"{b.proposition}\""
            for b in candidate_replacement_beliefs
        ) or "  (none)"

        cond_parts: list[str] = []
        for bid, conds in candidate_conditions_by_belief:
            for c in conds:
                cond_parts.append(f"  - [{bid}] {c.condition_id}: \"{c.text}\"")
        conditions_str = "\n".join(cond_parts) or "  (none)"

        prompt = self._template.replace("{evidence_text}", new_evidence.text)
        prompt = prompt.replace("{candidate_beliefs}", beliefs_str)
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
                "batch_belief_ids": [b.belief_id for b in candidate_beliefs],
            },
        )

        if trace.status != "success" or not trace.response:
            raise ValueError(
                f"Batched evidence-edge prediction failed: status={trace.status}, "
                f"error={trace.error_message}"
            )

        valid_belief_ids = {b.belief_id for b in candidate_beliefs}
        replacement_map = {b.belief_id: b for b in candidate_replacement_beliefs}
        valid_condition_ids: set[str] = set()
        condition_to_belief: dict[str, str] = {}
        for bid, conds in candidate_conditions_by_belief:
            for c in conds:
                valid_condition_ids.add(c.condition_id)
                condition_to_belief[c.condition_id] = bid

        edges = self._parse(
            trace.response,
            new_evidence,
            valid_belief_ids,
            replacement_map,
            valid_condition_ids,
            trace.call_id,
        )

        return EdgePredictionBatch(
            proposed_edges=tuple(edges),
            model_call_trace_id=trace.call_id,
            prompt_version=self.prompt_version,
            model_id=self.model_id,
            provider=self.provider,
            model_revision_or_api_version=self.model_revision_or_api_version,
            metadata={
                "batch_belief_count": len(candidate_beliefs),
                "batch_belief_ids": [b.belief_id for b in candidate_beliefs],
            },
        )

    def _parse(
        self,
        response: str,
        new_evidence: EvidenceNode,
        valid_belief_ids: set[str],
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
            raise ValueError("Invalid batched evidence-edge response: missing 'edges' key")

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
                if target_id not in valid_belief_ids:
                    raise ValueError(
                        f"{edge_type_str} edge must target a candidate belief id "
                        f"from {valid_belief_ids}, got '{target_id}'"
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
                self.prompt_version,
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
                verifier=self.prompt_version,
                replacement_belief_id=replacement_id,
                confidence=confidence,
                rationale=item.get("rationale"),
                model_call_trace_id=call_id,
            )
            edges.append(edge)

        return edges
