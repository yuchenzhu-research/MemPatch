"""Prompt-based typed belief extractor for Stage A ReTrace-LLM.

Implements the ``TypedBeliefExtractor`` protocol using ``CachedLLMClient``.
All model interactions go through the replay-safe cache; no real provider
SDK is required for offline contract testing.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, EvidenceNode

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "prompts", "retrace_llm")
_PROMPT_FILE = os.path.join(_PROMPT_DIR, "belief_extraction_v0.txt")
_PROMPT_VERSION = "belief_extraction_v0"
_RESPONSE_SCHEMA_VERSION = "belief_extraction_response_v0"
_PARSER_VERSION = "belief_extraction_parser_v0"


def _load_prompt_template() -> str:
    path = os.path.normpath(_PROMPT_FILE)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _stable_belief_id(scope_id: str, evidence_id: str, normalized_proposition: str) -> str:
    """Compute deterministic belief_id from grounded inputs."""
    payload = f"{scope_id}|{evidence_id}|{normalized_proposition}"
    return f"b-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _normalize_proposition(text: str) -> str:
    return " ".join(text.strip().split())


class PromptTypedBeliefExtractor:
    """Stage A belief extraction via LLM prompt and structured JSON parse.

    Implements ``TypedBeliefExtractor.extract()``.
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

    def extract(
        self,
        evidence: EvidenceNode,
        scope_id: str,
    ) -> list[BeliefNode]:
        if not scope_id:
            raise ValueError("scope_id is required and cannot be empty")

        prompt = self._template.replace("{evidence_text}", evidence.text)
        trace = self.client.generate(
            prompt=prompt,
            model_id=self.model_id,
            provider=self.provider,
            prompt_template_hash=self._template_hash,
            response_schema_version=_RESPONSE_SCHEMA_VERSION,
            parser_version=_PARSER_VERSION,
            temperature=0.0,
            metadata={"scope_id": scope_id, "evidence_id": evidence.evidence_id},
        )

        if trace.status != "success" or not trace.response:
            raise ValueError(
                f"Belief extraction failed: status={trace.status}, "
                f"error={trace.error_message}"
            )

        return self._parse(trace.response, evidence, scope_id, trace.call_id)

    def _parse(
        self,
        response: str,
        evidence: EvidenceNode,
        scope_id: str,
        call_id: str,
    ) -> list[BeliefNode]:
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        if not isinstance(data, dict) or "beliefs" not in data:
            raise ValueError(f"Invalid belief extraction response: missing 'beliefs' key")

        seen_normalized: set[str] = set()
        beliefs: list[BeliefNode] = []
        for item in data["beliefs"]:
            prop = item.get("proposition")
            if not prop or not prop.strip():
                raise ValueError(f"Belief item missing or empty proposition: {item}")

            normalized = _normalize_proposition(prop)
            if normalized in seen_normalized:
                raise ValueError(
                    f"Duplicate normalized proposition in extraction response: '{normalized}'"
                )
            seen_normalized.add(normalized)

            confidence = item.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                    raise ValueError(
                        f"Confidence must be a number in [0.0, 1.0], got {confidence!r}: {item}"
                    )

            bid = _stable_belief_id(scope_id, evidence.evidence_id, normalized)

            belief = BeliefNode(
                belief_id=bid,
                proposition=normalized,
                source_evidence_ids=(evidence.evidence_id,),
                extractor_version=_PROMPT_VERSION,
                confidence=confidence,
                metadata={"scope_id": scope_id, "model_call_trace_id": call_id},
            )
            beliefs.append(belief)

        return beliefs
