"""Ollama native chat (``/api/chat``) provider.

Targets a locally running Ollama server. Local auth is typically ``none``, so
no API key is required by default.
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

DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"


class OllamaChatProvider(BaseLLMProvider):
    """Provider for the Ollama native ``/api/chat`` endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 0,
        extra_headers: dict[str, str] | None = None,
        stream: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url or DEFAULT_ENDPOINT
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = dict(extra_headers or {})
        self.stream = stream

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

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
            call_id_prefix="ollama-call",
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

        # Ollama keeps evaluation reproducible with stream disabled.
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if seed is not None:
            options["seed"] = seed
        if options:
            payload["options"] = options

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
        message = data.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if text is None:
            text = data.get("response")

        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
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
