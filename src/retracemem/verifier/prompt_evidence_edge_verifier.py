"""Prompt-based evidence-edge verifier for Stage A ReTrace-LLM.

Implements the ``EvidenceEdgeVerifier`` protocol using ``CachedLLMClient``.
All model interactions go through the replay-safe cache.
"""
from __future__ import annotations

import hashlib
import os

from retracemem.methods.contracts import EdgePredictionBatch
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    EvidenceEdge,
    EvidenceNode,
)
from retracemem.verifier.typed_edge_response_parser import (
    EdgeTargetSpace,
    parse_typed_edge_response,
)

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "prompts", "retrace_llm")
_DEFAULT_PROMPT_VERSION = "evidence_edge_prediction_v0"
_RESPONSE_SCHEMA_VERSION = "evidence_edge_prediction_response_v0"
_PARSER_VERSION = "evidence_edge_prediction_parser_v0"

def _prompt_file(prompt_version: str) -> str:
    if prompt_version not in {"evidence_edge_prediction_v0", "evidence_edge_prediction_v1"}:
        raise ValueError(f"Unsupported evidence-edge prompt version: {prompt_version}")
    return os.path.join(_PROMPT_DIR, f"{prompt_version}.txt")


def _load_prompt_template(prompt_version: str) -> str:
    path = os.path.normpath(_prompt_file(prompt_version))
    with open(path, encoding="utf-8") as f:
        return f.read()


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
        prompt_version: str = _DEFAULT_PROMPT_VERSION,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.model_revision_or_api_version = model_revision_or_api_version
        self.prompt_version = prompt_version
        self._template = _load_prompt_template(prompt_version)
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

        edges = parse_typed_edge_response(
            trace.response,
            new_evidence=new_evidence,
            target_space=EdgeTargetSpace(
                valid_belief_ids=frozenset({candidate_belief.belief_id}),
                valid_condition_ids=frozenset(valid_condition_ids),
                replacement_map=replacement_map,
                single_belief_id=candidate_belief.belief_id,
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
        )
