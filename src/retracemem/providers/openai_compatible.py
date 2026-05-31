from __future__ import annotations
from typing import Any
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.schemas import ModelCallTrace

class OpenAICompatibleProvider(HTTPLLMProvider):
    """
    An OpenAI-compatible provider that communicates with remote APIs over HTTP.
    Inherits endpoint resolution and standard chat completions payload formatting.
    """
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            extra_headers=extra_headers,
        )
