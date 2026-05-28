"""Prompt-based requirement inducer for Stage A ReTrace-LLM.

Implements the ``RequirementInducer`` protocol using ``CachedLLMClient``.
All model interactions go through the replay-safe cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import Any

from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge, EvidenceNode
from retracemem.verifier.contracts import RequirementProposal

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "prompts", "retrace_llm")
_PROMPT_FILE = os.path.join(_PROMPT_DIR, "requirement_induction_v0.txt")
_PROMPT_VERSION = "requirement_induction_v0"
_RESPONSE_SCHEMA_VERSION = "requirement_induction_response_v0"
_PARSER_VERSION = "requirement_induction_parser_v0"


def _load_prompt_template() -> str:
    path = os.path.normpath(_PROMPT_FILE)
    with open(path, encoding="utf-8") as f:
        return f.read()


class PromptRequirementInducer:
    """Stage A requirement induction via LLM prompt and structured JSON parse.

    Implements ``RequirementInducer.induce_requirements()``.
    """

    def __init__(
        self,
        client: CachedLLMClient,
        model_id: str = "gemini-pro",
        provider: str = "google",
        scope_id: str = "default",
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.scope_id = scope_id
        self._template = _load_prompt_template()
        self._template_hash = hashlib.sha256(self._template.encode("utf-8")).hexdigest()

    def induce_requirements(
        self,
        belief: BeliefNode,
        evidence_context: tuple[EvidenceNode, ...],
    ) -> list[RequirementProposal]:
        evidence_text = "\n".join(e.text for e in evidence_context)
        prompt = self._template.replace("{belief_proposition}", belief.proposition)
        prompt = prompt.replace("{evidence_context}", evidence_text)

        evidence_ids = tuple(e.evidence_id for e in evidence_context)
        context_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()

        trace = self.client.generate(
            prompt=prompt,
            model_id=self.model_id,
            provider=self.provider,
            prompt_template_hash=self._template_hash,
            response_schema_version=_RESPONSE_SCHEMA_VERSION,
            parser_version=_PARSER_VERSION,
            temperature=0.0,
            condition_context_hash=context_hash,
            metadata={"belief_id": belief.belief_id},
        )

        if trace.status != "success" or not trace.response:
            raise ValueError(
                f"Requirement induction failed: status={trace.status}, "
                f"error={trace.error_message}"
            )

        return self._parse(trace.response, belief, evidence_ids, trace.call_id)

    def _parse(
        self,
        response: str,
        belief: BeliefNode,
        evidence_ids: tuple[str, ...],
        call_id: str,
    ) -> list[RequirementProposal]:
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        if not isinstance(data, dict) or "requirements" not in data:
            raise ValueError(f"Invalid requirement induction response: missing 'requirements' key")

        proposals: list[RequirementProposal] = []
        for item in data["requirements"]:
            cid = item.get("condition_id")
            ctext = item.get("condition_text")
            if not cid or not ctext:
                raise ValueError(f"Requirement item missing condition_id or condition_text: {item}")

            condition = ConditionNode(
                condition_id=cid,
                scope_id=self.scope_id,
                text=ctext,
            )

            edge_id = f"dep-{belief.belief_id}-{cid}-{uuid.uuid4().hex[:8]}"
            dependency_edge = DependencyEdge(
                edge_id=edge_id,
                belief_id=belief.belief_id,
                condition_id=cid,
                inducer=_PROMPT_VERSION,
                edge_type="REQUIRES",
                supporting_evidence_ids=evidence_ids,
                model_call_trace_id=call_id,
                confidence=item.get("confidence"),
                rationale=item.get("rationale"),
            )

            proposals.append(RequirementProposal(condition=condition, dependency_edge=dependency_edge))

        return proposals
