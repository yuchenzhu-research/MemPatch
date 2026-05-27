from __future__ import annotations

import hashlib
import json
from typing import Any

from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType


VERIFIER_PROMPT_TEMPLATE = """You are a Truth Maintenance System verifier. Analyze the relationship between a new piece of evidence and an existing user belief.

New Evidence:
"{evidence_text}"

Existing Belief:
"{belief_proposition}"

Determine the relation label between them. The relation label MUST be one of the following:
- SUPPORT: The new evidence confirms or continues to support the existing belief.
- SUPERSEDE: The new evidence directly replaces the existing belief (e.g. user updated their choice/preference or moved to a new state). You must provide the "target_belief" representing the new state/preference in the JSON.
- BLOCK: The new evidence defeats a prerequisite/condition required for the existing belief to be currently valid (e.g. broken leg blocks running habit). You must specify the "condition" that is blocked.
- CONDITION: The belief remains true historically, but executing it now requires a specific condition which may not be present (e.g., outdoor tennis requires good weather).
- NONE: The new evidence is completely irrelevant to the existing belief.
- UNCERTAIN: The information is insufficient or contradictory.

Provide your response in raw JSON format matching this schema:
{{
  "relation": "SUPPORT" | "SUPERSEDE" | "BLOCK" | "CONDITION" | "NONE" | "UNCERTAIN",
  "target_belief": "new belief statement if SUPERSEDE, else null",
  "condition": "condition text if BLOCK or CONDITION, else null",
  "rationale": "short explanation of the relationship decision",
  "confidence": float between 0.0 and 1.0
}}
"""


class PromptRelationVerifier:
    """LLM prompt-based relation verifier."""

    def __init__(
        self,
        client: CachedLLMClient,
        model_id: str = "gemini-pro",
        provider: str = "google",
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.temperature = temperature

    def verify(
        self,
        new_evidence: EpisodicEvidence,
        candidate_belief: Belief,
        context: dict[str, object] | None = None,
    ) -> RelationPrediction:
        del context

        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            evidence_text=new_evidence.text,
            belief_proposition=candidate_belief.proposition,
        )

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        try:
            trace = self.client.generate(
                prompt=prompt,
                model_id=self.model_id,
                provider=self.provider,
                temperature=self.temperature,
                prompt_template_hash=prompt_hash,
            )

            if trace.status != "success" or not trace.response:
                return RelationPrediction(
                    relation=RelationType.UNCERTAIN,
                    evidence_id=new_evidence.id,
                    belief_id=candidate_belief.id,
                    rationale=f"LLM API failure: {trace.error_message}",
                    confidence=0.0,
                )

            # Clean json block wrapper if any
            response_text = trace.response.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            data = json.loads(response_text)
            relation_str = data.get("relation", "UNCERTAIN").upper()
            try:
                relation_val = RelationType(relation_str)
            except ValueError:
                relation_val = RelationType.UNCERTAIN

            return RelationPrediction(
                relation=relation_val,
                evidence_id=new_evidence.id,
                belief_id=candidate_belief.id,
                condition=data.get("condition"),
                rationale=data.get("rationale"),
                confidence=data.get("confidence", 0.0),
            )

        except Exception as e:
            return RelationPrediction(
                relation=RelationType.UNCERTAIN,
                evidence_id=new_evidence.id,
                belief_id=candidate_belief.id,
                rationale=f"Parse failure: {str(e)}",
                confidence=0.0,
            )
