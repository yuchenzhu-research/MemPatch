from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import uuid
import hashlib
from retracemem.schemas import ModelCallTrace

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        model_revision_or_api_version: str | None = None,
        prompt_template_hash: str | None = None,
        response_schema_version: str | None = None,
        parser_version: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        condition_context_hash: str | None = None,
        temporal_context_hash: str | None = None,
        eligible_for_replay: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ModelCallTrace:
        """
        Generates a completion for the given prompt and returns a ModelCallTrace.
        All API calls must pass through this interface.
        """
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    A mock provider for deterministic testing without external API requirements.
    """
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "mocked response",
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        self.responses = responses or {}
        self.default_response = default_response
        self.status = status
        self.error_message = error_message
        self.last_prompt: str | None = None
        self.calls_count = 0

    def generate(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        model_revision_or_api_version: str | None = None,
        prompt_template_hash: str | None = None,
        response_schema_version: str | None = None,
        parser_version: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        condition_context_hash: str | None = None,
        temporal_context_hash: str | None = None,
        eligible_for_replay: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ModelCallTrace:
        self.last_prompt = prompt
        self.calls_count += 1
        
        response_text = self.responses.get(prompt, self.default_response) if self.status == "success" else None
        call_id = f"mock-call-{uuid.uuid4()}"
        input_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        
        prompt_tokens = len(prompt.split()) if prompt else 0
        completion_tokens = len(response_text.split()) if response_text else 0
        
        return ModelCallTrace(
            call_id=call_id,
            provider=provider,
            model_id=model_id,
            model_revision_or_api_version=model_revision_or_api_version,
            prompt_template_hash=prompt_template_hash,
            response_schema_version=response_schema_version,
            parser_version=parser_version,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            input_hash=input_hash,
            condition_context_hash=condition_context_hash,
            temporal_context_hash=temporal_context_hash,
            status=self.status,
            response=response_text,
            latency_ms=10.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            retries=0,
            error_message=self.error_message,
            eligible_for_replay=eligible_for_replay,
            metadata=metadata or {},
        )
