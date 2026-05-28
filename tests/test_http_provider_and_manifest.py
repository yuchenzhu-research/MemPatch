from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import pytest
from retracemem.evaluation.manifest import RunConfiguration, RunManifest, compute_file_sha256, get_git_commit_sha
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.schemas import ModelCallTrace


def test_http_provider_resolves_endpoints_and_headers() -> None:
    # 1. No base_url specified, google provider
    p1 = HTTPLLMProvider(api_key="test-key")
    url, headers = p1._resolve_endpoint_and_headers("google")
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"

    # 2. No base_url specified, openai provider
    p2 = HTTPLLMProvider(api_key="test-key")
    url, headers = p2._resolve_endpoint_and_headers("openai")
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert "x-goog-api-key" not in headers

    # 3. Explicit base_url specified
    p3 = HTTPLLMProvider(api_key="test-key", base_url="https://custom.api.com/chat")
    url, headers = p3._resolve_endpoint_and_headers("openai")
    assert url == "https://custom.api.com/chat"


def test_http_provider_selects_provider_specific_environment_keys(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    provider = HTTPLLMProvider()
    _, openai_headers = provider._resolve_endpoint_and_headers("openai")
    _, gemini_headers = provider._resolve_endpoint_and_headers("gemini")

    assert openai_headers["Authorization"] == "Bearer openai-key"
    assert gemini_headers["Authorization"] == "Bearer gemini-key"


def test_http_provider_missing_key_is_provider_specific(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    trace = HTTPLLMProvider().generate(
        prompt="Hello",
        model_id="gpt-4o",
        provider="openai",
    )

    assert trace.status == "failure"
    assert "OPENAI_API_KEY" in trace.error_message
    assert "GEMINI_API_KEY" not in trace.error_message
    assert "gemini-key" not in trace.error_message


def test_http_provider_rejects_unsupported_provider() -> None:
    provider = HTTPLLMProvider(api_key="test-key")

    with pytest.raises(ValueError, match="Unsupported HTTP provider"):
        provider._resolve_endpoint_and_headers("unknown")


@patch("urllib.request.urlopen")
def test_http_provider_successful_generate(mock_urlopen) -> None:
    # Setup mock HTTP response matching chat completions format
    mock_response = MagicMock()
    response_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Simulated output text"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5
        }
    }
    mock_response.read.return_value = json.dumps(response_payload).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    p = HTTPLLMProvider(api_key="test-key")
    trace = p.generate(
        prompt="Hello",
        model_id="gpt-4o",
        provider="openai",
        temperature=0.7,
        eligible_for_replay=True,
    )

    assert trace.status == "success"
    assert trace.response == "Simulated output text"
    assert trace.prompt_tokens == 10
    assert trace.completion_tokens == 5
    assert trace.total_tokens == 15
    assert trace.error_message is None
    # Confirm key is not exposed
    assert "test-key" not in json.dumps(trace.metadata)


@patch("urllib.request.urlopen")
def test_http_provider_missing_usage_fallback(mock_urlopen) -> None:
    # Setup mock HTTP response without usage field
    mock_response = MagicMock()
    response_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Output word counts estimate"
                }
            }
        ]
    }
    mock_response.read.return_value = json.dumps(response_payload).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    p = HTTPLLMProvider(api_key="test-key")
    trace = p.generate(
        prompt="Hello world input",  # 3 words
        model_id="gpt-4o",
        provider="openai",
    )

    assert trace.status == "success"
    assert trace.response == "Output word counts estimate"  # 4 words
    assert trace.prompt_tokens == 3
    assert trace.completion_tokens == 4
    assert trace.total_tokens == 7


@patch("urllib.request.urlopen")
def test_http_provider_http_failure(mock_urlopen) -> None:
    # Setup mock URL open exception
    mock_urlopen.side_effect = Exception("Rate limit exceeded")

    p = HTTPLLMProvider(api_key="test-key")
    trace = p.generate(
        prompt="Hello",
        model_id="gpt-4o",
        provider="openai",
    )

    assert trace.status == "failure"
    assert trace.response is None
    assert "Rate limit" in trace.error_message


@patch("urllib.request.urlopen")
def test_http_provider_redacts_secret_from_error_message(mock_urlopen) -> None:
    mock_urlopen.side_effect = Exception("token secret-key appeared in provider error")

    trace = HTTPLLMProvider(api_key="secret-key").generate(
        prompt="Hello",
        model_id="gpt-4o",
        provider="openai",
    )

    assert trace.status == "failure"
    assert "secret-key" not in trace.error_message
    assert "[REDACTED]" in trace.error_message


def test_manifest_and_git_sha_computations() -> None:
    sha = get_git_commit_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("manifest data content")

        checksum = compute_file_sha256(test_file)
        expected = hashlib.sha256(b"manifest data content").hexdigest()
        assert checksum == expected

        # RunConfiguration
        config = RunConfiguration(
            run_id="test-run-123",
            stage_and_method_name="AB-1C_StageA",
            provider_name="openai",
            model_id="gpt-4",
            cache_path=test_file,
            dataset_checksum="dataset-hash-123",
        )
        assert config.run_id == "test-run-123"
        assert config.git_commit_sha == sha

        # RunManifest
        manifest = RunManifest(
            config=config,
            aggregate_cost={"total_tokens": 100},
            instance_count=1,
            output_path=test_file,
        )
        
        manifest_file = os.path.join(tmpdir, "manifest.json")
        manifest.save(manifest_file)
        
        assert os.path.exists(manifest_file)
        with open(manifest_file) as f:
            manifest_data = json.load(f)
            
        assert manifest_data["config"]["run_id"] == "test-run-123"
        assert manifest_data["output_checksum"] == expected
        assert manifest_data["aggregate_cost"]["total_tokens"] == 100


def test_capped_provider_wrapper() -> None:
    from scripts.run_controlled_ab_dev import CappedProviderWrapper
    from retracemem.providers.base import MockLLMProvider

    mock_provider = MockLLMProvider(default_response="response")
    # Wrap with 2 calls max, and 100 tokens max
    capped = CappedProviderWrapper(mock_provider, max_calls=2, max_tokens=100)

    # First call: OK (prompt has 1 word, response has 1 word => 2 tokens)
    trace1 = capped.generate(prompt="Hello", model_id="m", provider="p")
    assert trace1.total_tokens == 2

    # Second call: OK
    trace2 = capped.generate(prompt="Hello", model_id="m", provider="p")
    assert trace2.total_tokens == 2

    # Third call: Exceeds call cap!
    with pytest.raises(RuntimeError) as excinfo:
        capped.generate(prompt="Hello", model_id="m", provider="p")
    assert "Hard call cap" in str(excinfo.value)

    # Test token cap
    mock_provider_large = MockLLMProvider(default_response="a " * 200)  # 200 words
    capped_token = CappedProviderWrapper(mock_provider_large, max_calls=5, max_tokens=50)
    
    with pytest.raises(RuntimeError) as excinfo:
        capped_token.generate(prompt="Hello", model_id="m", provider="p")
    assert "Hard token cap" in str(excinfo.value)

