"""Helper for constructing :class:`ModelCallTrace` objects uniformly.

Every provider mode (OpenAI-compatible, Anthropic Messages, Ollama) must emit
the same ``ModelCallTrace`` shape so the cache, cost accounting, and replay
machinery are provider-agnostic.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

from retracemem.schemas import ModelCallTrace


def build_call_trace(
    *,
    prompt: str,
    provider: str,
    model_id: str,
    call_id_prefix: str,
    status: str = "success",
    response: str | None = None,
    latency_ms: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    retries: int = 0,
    error_message: str | None = None,
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
    return ModelCallTrace(
        call_id=f"{call_id_prefix}-{uuid.uuid4()}",
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
        input_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        condition_context_hash=condition_context_hash,
        temporal_context_hash=temporal_context_hash,
        status=status,
        response=response,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
        retries=retries,
        error_message=error_message,
        eligible_for_replay=eligible_for_replay,
        metadata=metadata or {},
    )
