"""Shared Gemini API helpers for ReTrace-Bench scripts (env key only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def resolve_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def require_api_key() -> str:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        except ImportError:
            pass
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set in the environment.")
    return api_key


def redact_secrets(text: str, api_key: str | None = None) -> str:
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    redacted = text
    if key:
        redacted = redacted.replace(key, "[REDACTED]")
    return redacted


def call_gemini_generate(*, prompt: str, model: str | None = None, temperature: float = 0.0) -> str:
    """Call Gemini generateContent with JSON response and no tools/grounding."""
    api_key = require_api_key()
    model_name = model or resolve_model()
    url = f"{GEMINI_API_BASE}/models/{model_name}:generateContent"
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(redact_secrets(f"HTTP {exc.code}: {err_body}", api_key)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(redact_secrets(f"Network error: {exc.reason}", api_key)) from exc

    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(redact_secrets(f"No candidates in response: {json.dumps(body)[:500]}", api_key))
    parts = candidates[0].get("content", {}).get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    text = "".join(texts).strip()
    if not text:
        raise RuntimeError("Gemini returned empty text.")
    return text
