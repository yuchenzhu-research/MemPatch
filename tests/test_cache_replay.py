from __future__ import annotations
import os
import tempfile
from retracemem.schemas import ModelCallTrace
from retracemem.cache.jsonl_cache import JSONLCache, calculate_cache_key
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.evaluation.cost_accounting import CostAccounting


def test_cache_replay_hits_and_misses() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "cache.jsonl")
        
        # 1. Setup cache, mock provider, cost accountant, and client
        cache = JSONLCache(cache_file)
        mock_provider = MockLLMProvider(default_response="Response A")
        cost_accountant = CostAccounting()
        client = CachedLLMClient(cache, mock_provider, cost_accountant)

        # First call: Cache miss, calls provider
        prompt = "Hello world"
        trace1 = client.generate(prompt=prompt, model_id="gpt-4", provider="openai")
        
        assert trace1.response == "Response A"
        assert mock_provider.calls_count == 1
        assert cost_accountant.cache_hits == 0
        assert cost_accountant.cache_misses == 1
        assert cost_accountant.tokens["total"] > 0

        # Second call: Cache hit, does not call provider
        trace2 = client.generate(prompt=prompt, model_id="gpt-4", provider="openai")
        
        assert trace2.response == "Response A"
        assert trace2.call_id == trace1.call_id  # Replayed call trace!
        assert mock_provider.calls_count == 1   # Still 1 call to provider
        assert cost_accountant.cache_hits == 1
        assert cost_accountant.cache_misses == 1


def test_non_replayable_or_failed_traces_are_not_replayed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "cache.jsonl")
        
        # 1. Setup cache, mock provider with error, and cost accountant
        cache = JSONLCache(cache_file)
        mock_provider = MockLLMProvider(status="failure", error_message="Rate limit")
        cost_accountant = CostAccounting()
        client = CachedLLMClient(cache, mock_provider, cost_accountant)

        # Call with failing provider
        prompt = "test failure"
        trace1 = client.generate(prompt=prompt, model_id="gpt-4", provider="openai")
        
        assert trace1.status == "failure"
        assert trace1.error_message == "Rate limit"
        assert mock_provider.calls_count == 1
        assert cost_accountant.cache_hits == 0
        assert cost_accountant.cache_misses == 1
        assert cost_accountant.error_counts["Rate limit"] == 1

        # Second call with failing provider: should not replay, because status != success
        # It should call provider again
        trace2 = client.generate(prompt=prompt, model_id="gpt-4", provider="openai")
        assert mock_provider.calls_count == 2
        assert cost_accountant.cache_hits == 0
        assert cost_accountant.cache_misses == 2

        # 2. Call with eligible_for_replay = False
        mock_provider_success = MockLLMProvider(default_response="No Replay Success")
        client2 = CachedLLMClient(cache, mock_provider_success, cost_accountant)
        
        trace_no_replay = client2.generate(
            prompt="no replay prompt",
            model_id="gpt-4",
            provider="openai",
            eligible_for_replay=False
        )
        
        assert trace_no_replay.status == "success"
        assert trace_no_replay.eligible_for_replay is False
        assert mock_provider_success.calls_count == 1

        # Second call to same prompt: should cache miss because eligible_for_replay is False
        trace_no_replay2 = client2.generate(
            prompt="no replay prompt",
            model_id="gpt-4",
            provider="openai",
            eligible_for_replay=False
        )
        assert mock_provider_success.calls_count == 2
        assert cost_accountant.cache_hits == 0
