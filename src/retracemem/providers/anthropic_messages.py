"""Anthropic Messages (``/v1/messages``) provider.

Implements the same :class:`~retracemem.providers.base.BaseLLMProvider`
interface as the OpenAI-compatible provider, so the evaluation runner is
agnostic to whether the backing model is Claude-style or OpenAI-style.
"""
from __future__ import annotations

import time
from typing import Any

from retracemem.providers._trace import build_call_trace
from retracemem.providers._transport import (
    TransportError,
    http_post_json,
    make_redactor,
)
from retracemem.providers.base import BaseLLMProvider
from retracemem.schemas import ModelCallTrace

DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


class AnthropicMessagesProvider(BaseLLMProvider):
    """Provider for Anthropic's Messages API (Claude message schema)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 0,
        extra_headers: dict[str, str] | None = None,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url or DEFAULT_ENDPOINT
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = dict(extra_headers or {})
        self.anthropic_version = anthropic_version
        self.default_max_tokens = default_max_tokens

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": self.anthropic_version,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        headers.update(self.extra_headers)
        return headers

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str | None:
        content = data.get("content")
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text = "".join(parts)
            return text or None
        if isinstance(content, str):
            return content
        return None

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
        common = dict(
            prompt=prompt,
            provider=provider,
            model_id=model_id,
            call_id_prefix="anthropic-call",
            model_revision_or_api_version=model_revision_or_api_version,
            prompt_template_hash=prompt_template_hash,
            response_schema_version=response_schema_version,
            parser_version=parser_version,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            condition_context_hash=condition_context_hash,
            temporal_context_hash=temporal_context_hash,
            eligible_for_replay=eligible_for_replay,
            metadata=metadata,
        )

        if not self.api_key:
            return build_call_trace(
                status="failure",
                error_message=(
                    f"API key missing for provider '{provider}' (anthropic-messages): "
                    f"set x-api-key via api_key_env or pass api_key."
                ),
                **common,
            )

        payload: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens or self.default_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        redactor = make_redactor([self.api_key])
        start = time.perf_counter()
        try:
            data, attempts = http_post_json(
                self.base_url,
                payload,
                self._headers(),
                timeout=self.timeout,
                max_retries=self.max_retries,
                redactor=redactor,
            )
        except TransportError as e:
            return build_call_trace(
                status="failure",
                error_message=str(e),
                latency_ms=(time.perf_counter() - start) * 1000.0,
                prompt_tokens=len(prompt.split()),
                total_tokens=len(prompt.split()),
                retries=max(0, e.attempts - 1),
                **common,
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        text = self._extract_text(data)
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        if not prompt_tokens and not completion_tokens:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(text.split()) if text else 0

        return build_call_trace(
            status="success",
            response=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retries=max(0, attempts - 1),
            **common,
        )
