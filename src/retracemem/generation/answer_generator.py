from __future__ import annotations

import hashlib
import os
from typing import Any, Union

from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import BeliefNode


class PromptAnswerGenerator:
    """Production-capable prompt-based answer generator.

    Fuses the query with the current query-relevant authorized belief basis
    and generates the final response using a versioned prompt template.
    """

    def __init__(self, client: CachedLLMClient, prompt_filepath: str | None = None) -> None:
        self.client = client
        self.prompt_filepath = prompt_filepath or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir,
            os.pardir,
            os.pardir,
            "prompts",
            "generation",
            "answer_generation_v0.txt",
        )
        with open(self.prompt_filepath, "r", encoding="utf-8") as f:
            self.template = f.read()
        self.template_hash = hashlib.sha256(self.template.encode("utf-8")).hexdigest()

    def generate_answer(
        self,
        query: str,
        authorized_basis: list[dict[str, Any]] | list[BeliefNode] | tuple[BeliefNode, ...],
        model_id: str,
        provider: str,
    ) -> str:
        props = []
        for b in authorized_basis:
            if isinstance(b, dict):
                prop = b.get("proposition") or b.get("text") or ""
            else:
                prop = getattr(b, "proposition", "")
            if prop:
                props.append(prop)

        context_text = "\n".join(f"- {p}" for p in props)
        prompt = self.template.replace("{context_text}", context_text).replace("{query}", query)

        trace = self.client.generate(
            prompt=prompt,
            model_id=model_id,
            provider=provider,
            prompt_template_hash=self.template_hash,
            response_schema_version="answer_generation_response_v0",
            parser_version="answer_generation_parser_v0",
            temperature=0.0,
        )

        if trace.status != "success" or not trace.response:
            raise ValueError(f"Answer generation failed: status={trace.status}, error={trace.error_message}")

        return trace.response
