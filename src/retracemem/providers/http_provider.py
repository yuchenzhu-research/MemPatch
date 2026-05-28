from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import uuid
from typing import Any

from retracemem.providers.base import BaseLLMProvider
from retracemem.schemas import ModelCallTrace


class HTTPLLMProvider(BaseLLMProvider):
    """
    A production-quality LLM provider that communicates with remote APIs over HTTP.
    Uses Python's standard library urllib.request to avoid heavy SDK dependencies.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout = timeout

    def _resolve_endpoint_and_headers(self, provider: str) -> tuple[str, dict[str, str]]:
        headers = {"Content-Type": "application/json"}
        
        # If base_url is explicitly set, use it.
        if self.base_url:
            endpoint = self.base_url
        elif provider.lower() in ("google", "gemini"):
            endpoint = "https://generativelanguage.googleapis.com/v1beta/chat/completions"
        else:
            endpoint = "https://api.openai.com/v1/chat/completions"

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            # For Gemini endpoints, also set x-goog-api-key just in case
            if "generativelanguage.googleapis.com" in endpoint:
                headers["x-goog-api-key"] = self.api_key

        return endpoint, headers

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
        call_id = f"http-call-{uuid.uuid4()}"
        input_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        
        if not self.api_key:
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
                status="failure",
                response=None,
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                retries=0,
                error_message="API Key missing: please set GEMINI_API_KEY or OPENAI_API_KEY env vars.",
                eligible_for_replay=eligible_for_replay,
                metadata=metadata or {},
            )

        endpoint, headers = self._resolve_endpoint_and_headers(provider)

        # Build payload
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed

        start_time = time.perf_counter()
        status = "success"
        response_text = None
        error_message = None
        prompt_tokens = 0
        completion_tokens = 0

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                
                # Check for standard chat completions structure
                choices = resp_data.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    choice = choices[0]
                    message = choice.get("message")
                    if message and isinstance(message, dict):
                        response_text = message.get("content")
                    elif isinstance(choice, str):
                        response_text = choice
                
                if response_text is None:
                    # In case of alternative API structure
                    response_text = resp_data.get("response") or str(resp_data)
                
                # Parse usage metrics
                usage = resp_data.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                else:
                    # Fallback token estimate
                    prompt_tokens = len(prompt.split())
                    completion_tokens = len(response_text.split()) if response_text else 0

        except Exception as e:
            status = "failure"
            error_message = f"{type(e).__name__}: {str(e)}"
            # Basic fallback counts on failure
            prompt_tokens = len(prompt.split())

        latency_ms = (time.perf_counter() - start_time) * 1000.0

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
            status=status,
            response=response_text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            retries=0,
            error_message=error_message,
            eligible_for_replay=eligible_for_replay,
            metadata=metadata or {},
        )
