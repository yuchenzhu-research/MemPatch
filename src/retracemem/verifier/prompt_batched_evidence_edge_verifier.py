"""Batched prompt-based evidence-edge verifier for Stage A ReTrace-LLM.

Renders all candidate beliefs in a single prompt instead of one call per belief.
Returns the same ``EdgePredictionBatch`` contract so downstream RevisionGate
and DPA semantics are unchanged.

The per-belief ``PromptEvidenceEdgeVerifier`` remains the auditable controlled
reference implementation and is NOT modified.
"""
from __future__ import annotations

import hashlib
import os

from retracemem.methods.contracts import EdgePredictionBatch
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    EvidenceNode,
)
from retracemem.verifier.typed_edge_response_parser import (
    EdgeTargetSpace,
    parse_typed_edge_response,
)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "prompts", "retrace_llm")
_RESPONSE_SCHEMA_VERSION = "evidence_edge_prediction_batch_response_v1"
_PARSER_VERSION = "evidence_edge_prediction_batch_parser_v1"

def _prompt_file(prompt_version: str) -> str:
    if prompt_version != "evidence_edge_prediction_batch_v1":
        raise ValueError(f"Unsupported batched prompt version: {prompt_version}")
    return os.path.join(_PROMPT_DIR, f"{prompt_version}.txt")


def _load_prompt_template(prompt_version: str) -> str:
    path = os.path.normpath(_prompt_file(prompt_version))
    with open(path, encoding="utf-8") as f:
        return f.read()


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
        for bid, conds in candidate_conditions_by_belief:
            for c in conds:
                valid_condition_ids.add(c.condition_id)

        edges = parse_typed_edge_response(
            trace.response,
            new_evidence=new_evidence,
            target_space=EdgeTargetSpace(
                valid_belief_ids=frozenset(valid_belief_ids),
                valid_condition_ids=frozenset(valid_condition_ids),
                replacement_map=replacement_map,
            ),
            call_id=trace.call_id,
            prompt_version=self.prompt_version,
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
