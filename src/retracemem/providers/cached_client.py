from __future__ import annotations
import time
import hashlib
import dataclasses
from typing import Any, Optional
from retracemem.schemas import ModelCallTrace
from retracemem.providers.base import BaseLLMProvider
from retracemem.cache.jsonl_cache import JSONLCache, calculate_cache_key
from retracemem.evaluation.cost_accounting import CostAccounting

class CachedLLMClient:
    """
    A caching wrapper around a BaseLLMProvider.
    Attempts to read from a replay-safe cache. On cache misses, delegates to the
    underlying provider, updates the cache, and logs usage to a CostAccounting instance.
    """
    def __init__(
        self,
        cache: JSONLCache,
        provider_client: BaseLLMProvider,
        cost_accountant: CostAccounting | None = None,
    ) -> None:
        self.cache = cache
        self.provider_client = provider_client
        self.cost_accountant = cost_accountant or CostAccounting()

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
        input_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        # Compute strict cache key
        cache_key = calculate_cache_key(
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
        )

        # Lookup in replay-safe cache
        cached_trace = self.cache.lookup(cache_key)
        if cached_trace is not None:
            # Cache hit: record call using cached trace details
            self.cost_accountant.record_call(
                latency_ms=cached_trace.latency_ms,
                prompt_tokens=cached_trace.prompt_tokens,
                completion_tokens=cached_trace.completion_tokens,
                status=cached_trace.status,
                is_cache_hit=True,
                config={
                    "model_id": model_id,
                    "provider": provider,
                    "temperature": temperature,
                },
            )
            return cached_trace

        # Cache miss: run using underlying provider client
        start_time = time.perf_counter()
        
        trace = self.provider_client.generate(
            prompt=prompt,
            model_id=model_id,
            provider=provider,
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
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Override latency in trace if it was not provided or if we measure it locally
        if not trace.latency_ms:
            trace = dataclasses.replace(trace, latency_ms=latency_ms)
            
        if not trace.input_hash:
            trace = dataclasses.replace(trace, input_hash=input_hash)

        # Append to JSONL Cache Log
        self.cache.log_and_write(cache_key, trace)

        # Update cost accounting metrics
        error_type = None
        if trace.status != "success":
            error_type = trace.error_message or "UnknownError"

        self.cost_accountant.record_call(
            latency_ms=trace.latency_ms,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            status=trace.status,
            is_cache_hit=False,
            config={
                "model_id": model_id,
                "provider": provider,
                "temperature": temperature,
            },
            error_type=error_type,
        )

        return trace
