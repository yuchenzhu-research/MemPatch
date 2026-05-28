"""Prompt-based requirement inducer for Stage A ReTrace-LLM.

Implements the ``RequirementInducer`` protocol using ``CachedLLMClient``.
All model interactions go through the replay-safe cache.
"""
from __future__ import annotations

import hashlib
import json
import os
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


def _normalize_condition_text(text: str) -> str:
    return " ".join(text.strip().split())


def _stable_condition_id(scope_id: str, normalized_text: str) -> str:
    """Compute deterministic condition_id from scope and normalized text."""
    payload = f"{scope_id}|{normalized_text}"
    return f"c-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _stable_dep_edge_id(belief_id: str, condition_id: str) -> str:
    """Compute deterministic dependency edge_id."""
    payload = f"{belief_id}|{condition_id}|{_PROMPT_VERSION}"
    return f"dep-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


class PromptRequirementInducer:
    """Stage A requirement induction via LLM prompt and structured JSON parse.

    Implements ``RequirementInducer.induce_requirements()``.

    Scope identity is derived from ``belief.metadata["scope_id"]``. No
    default or global scope fallback is permitted.
    """

    def __init__(
        self,
        client: CachedLLMClient,
        model_id: str = "gemini-pro",
        provider: str = "google",
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self._template = _load_prompt_template()
        self._template_hash = hashlib.sha256(self._template.encode("utf-8")).hexdigest()

    def induce_requirements(
        self,
        belief: BeliefNode,
        evidence_context: tuple[EvidenceNode, ...],
    ) -> list[RequirementProposal]:
        scope_id = (belief.metadata or {}).get("scope_id")
        if not scope_id:
            raise ValueError(
                f"PromptRequirementInducer requires explicit scope_id in "
                f"belief.metadata['scope_id']; got {belief.metadata}"
            )

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

        return self._parse(trace.response, belief, scope_id, evidence_ids, trace.call_id)

    def _parse(
        self,
        response: str,
        belief: BeliefNode,
        scope_id: str,
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

        seen_normalized: set[str] = set()
        proposals: list[RequirementProposal] = []
        for item in data["requirements"]:
            ctext = item.get("condition_text")
            if not ctext or not ctext.strip():
                raise ValueError(f"Requirement item missing or empty condition_text: {item}")

            normalized = _normalize_condition_text(ctext)
            if normalized in seen_normalized:
                raise ValueError(
                    f"Duplicate normalized condition in induction response: '{normalized}'"
                )
            seen_normalized.add(normalized)

            confidence = item.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                    raise ValueError(
                        f"Confidence must be a number in [0.0, 1.0], got {confidence!r}: {item}"
                    )

            cid = _stable_condition_id(scope_id, normalized)
            edge_id = _stable_dep_edge_id(belief.belief_id, cid)

            condition = ConditionNode(
                condition_id=cid,
                scope_id=scope_id,
                text=normalized,
            )

            dependency_edge = DependencyEdge(
                edge_id=edge_id,
                belief_id=belief.belief_id,
                condition_id=cid,
                inducer=_PROMPT_VERSION,
                edge_type="REQUIRES",
                supporting_evidence_ids=evidence_ids,
                model_call_trace_id=call_id,
                confidence=confidence,
                rationale=item.get("rationale"),
            )

            proposals.append(RequirementProposal(condition=condition, dependency_edge=dependency_edge))

        return proposals
