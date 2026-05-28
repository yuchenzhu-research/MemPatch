from __future__ import annotations

import hashlib
import json
from typing import Protocol
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode, EvidenceNode


class TypedBeliefExtractor(Protocol):
    """Protocol for extracting BeliefNodes from an EvidenceNode."""

    def extract(
        self,
        evidence: EvidenceNode,
        scope_id: str,
    ) -> list[BeliefNode]:
        """Extract BeliefNodes from the given EvidenceNode."""
        ...


class ManualTypedBeliefExtractor:
    """Development-only deterministic fixture for typed belief extraction.

    This class extracts predefined beliefs mapped to specific evidence IDs
    strictly for testing the DPA pipeline.
    """

    def __init__(self, mappings: dict[str, list[BeliefNode]] | None = None) -> None:
        self.mappings = mappings or {}

    def extract(
        self,
        evidence: EvidenceNode,
        scope_id: str,
    ) -> list[BeliefNode]:
        if not scope_id:
            raise ValueError("scope_id is required and cannot be empty")

        beliefs = self.mappings.get(evidence.evidence_id, [])
        for belief in beliefs:
            # Grounding check: evidence.evidence_id must be in belief.source_evidence_ids.
            if evidence.evidence_id not in belief.source_evidence_ids:
                raise ValueError(
                    f"Grounding violation: Belief {belief.belief_id} does not list "
                    f"source evidence {evidence.evidence_id} in source_evidence_ids {belief.source_evidence_ids}"
                )

        return beliefs


class PromptTypedBeliefExtractor:
    """Prompt-based belief extractor for live benchmark wiring."""

    prompt_version = "typed_belief_extraction_v1"
    response_schema_version = "typed_belief_extraction_response_v1"
    parser_version = "typed_belief_extraction_parser_v1"

    def __init__(
        self,
        client: CachedLLMClient,
        model_id: str,
        provider: str,
        max_beliefs: int = 8,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.max_beliefs = max_beliefs
        self.template = (
            "Extract up to {max_beliefs} concise atomic user-memory beliefs from "
            "the evidence below. Return strict JSON only with shape "
            "{{\"beliefs\":[{{\"proposition\":\"...\"}}]}}. Do not include explanations.\n\n"
            "Evidence ID: {evidence_id}\n"
            "Evidence text:\n{evidence_text}"
        )
        self.template_hash = hashlib.sha256(self.template.encode("utf-8")).hexdigest()

    def extract(self, evidence: EvidenceNode, scope_id: str) -> list[BeliefNode]:
        prompt = self.template.format(
            max_beliefs=self.max_beliefs,
            evidence_id=evidence.evidence_id,
            evidence_text=evidence.text,
        )
        trace = self.client.generate(
            prompt=prompt,
            model_id=self.model_id,
            provider=self.provider,
            prompt_template_hash=self.template_hash,
            response_schema_version=self.response_schema_version,
            parser_version=self.parser_version,
            temperature=0.0,
        )
        if trace.status != "success" or not trace.response:
            raise ValueError(
                f"Belief extraction failed: status={trace.status}, error={trace.error_message}"
            )
        return self._parse(trace.response, evidence, scope_id)

    def _parse(
        self,
        response: str,
        evidence: EvidenceNode,
        scope_id: str,
    ) -> list[BeliefNode]:
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        beliefs_raw = data.get("beliefs") if isinstance(data, dict) else None
        if not isinstance(beliefs_raw, list):
            raise ValueError("Belief extraction response missing beliefs list")
        beliefs: list[BeliefNode] = []
        seen: set[str] = set()
        for idx, item in enumerate(beliefs_raw[: self.max_beliefs]):
            if not isinstance(item, dict):
                continue
            proposition = str(item.get("proposition") or "").strip()
            if not proposition or proposition in seen:
                continue
            seen.add(proposition)
            beliefs.append(
                BeliefNode(
                    belief_id=f"b:{scope_id}:{evidence.evidence_id}:{idx}",
                    proposition=proposition,
                    source_evidence_ids=(evidence.evidence_id,),
                    confidence=1.0,
                    metadata={"extractor": self.prompt_version, "scope_id": scope_id},
                )
            )
        return beliefs
