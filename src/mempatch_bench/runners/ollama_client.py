"""Small stdlib Ollama client for local smoke tests."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an invalid response."""


@dataclass(frozen=True)
class OllamaGeneration:
    text: str
    raw: dict[str, Any]
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 240.0,
        retries: int = 1,
        retry_sleep_seconds: float = 3.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            request = Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    parsed = json.loads(body) if body else {}
                    parsed["_client_latency_seconds"] = time.perf_counter() - started
                    return parsed
            except HTTPError as exc:
                last_error = OllamaError(f"HTTP {exc.code} from Ollama {path}")
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.retry_sleep_seconds)
        raise OllamaError(f"Ollama request failed for {path}: {last_error}")

    def tags(self) -> dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        try:
            with urlopen(url, timeout=min(self.timeout_seconds, 10.0)) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise OllamaError(f"Ollama tags request failed: {exc}") from exc

    def model_names(self) -> list[str]:
        payload = self.tags()
        return [str(row.get("name") or row.get("model")) for row in payload.get("models") or [] if row.get("name") or row.get("model")]

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        system: str = "Return only the requested JSON. Do not use Markdown.",
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = 42,
        num_ctx: int = 8192,
        num_predict: int = 512,
        keep_alive: str = "5m",
    ) -> OllamaGeneration:
        options: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        if seed is not None:
            options["seed"] = seed
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": options,
            "keep_alive": keep_alive,
        }
        raw = self._post("/api/chat", payload)
        message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
        text = str(message.get("content") or raw.get("response") or "")
        return OllamaGeneration(
            text=text,
            raw=raw,
            latency_seconds=float(raw.get("_client_latency_seconds") or 0.0),
            input_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
        )

    def unload(self, model: str) -> None:
        try:
            self._post("/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": 0})
        except OllamaError:
            return
