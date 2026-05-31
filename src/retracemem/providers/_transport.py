"""Shared HTTP transport helpers for provider-agnostic LLM clients.

All provider request logic that talks to a remote endpoint funnels through
``http_post_json`` so retry, timeout, and secret-redaction behaviour stay
identical across the OpenAI-compatible, Anthropic Messages, and Ollama
provider modes. Only the standard library is used (``urllib``) to avoid heavy
SDK dependencies, consistent with the repository's "standard library first"
contract in ``AGENTS.md``.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable


def make_redactor(secret_values: Iterable[str | None]) -> Callable[[str], str]:
    """Return a function that scrubs known secret values from a string.

    Secrets (API keys) must never appear in persisted traces or error
    messages. The returned callable replaces every non-empty secret with
    ``[REDACTED]``.
    """
    secrets = [s for s in secret_values if s]

    def _redact(text: str) -> str:
        redacted = text
        for value in secrets:
            redacted = redacted.replace(value, "[REDACTED]")
        return redacted

    return _redact


class TransportError(Exception):
    """Raised when an HTTP request ultimately fails after all retries.

    ``message`` is already secret-redacted by the time this is raised.
    """

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


def http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    max_retries: int = 0,
    redactor: Callable[[str], str] | None = None,
    retry_backoff_s: float = 0.0,
) -> tuple[dict[str, Any], int]:
    """POST ``payload`` as JSON and return ``(parsed_response, attempts)``.

    Retries up to ``max_retries`` additional times on transport/HTTP errors
    (so a total of ``max_retries + 1`` attempts). The error message attached to
    a raised :class:`TransportError` is passed through ``redactor`` first so no
    secret leaks into logs or traces.
    """
    redact = redactor or (lambda s: s)
    attempts = 0
    last_error: str = "unknown transport error"
    total_attempts = max(0, max_retries) + 1

    for attempt in range(total_attempts):
        attempts = attempt + 1
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), attempts
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            raw = f"{type(e).__name__}: {str(e)}"
            if body:
                raw = f"{raw}; body={body[:1000]}"
            last_error = redact(raw)
        except Exception as e:  # noqa: BLE001 - normalize all transport errors
            last_error = redact(f"{type(e).__name__}: {str(e)}")

        if attempt < total_attempts - 1 and retry_backoff_s > 0:
            time.sleep(retry_backoff_s)

    raise TransportError(last_error, attempts=attempts)
