from __future__ import annotations
import os
import json
import hashlib
import dataclasses
from typing import Any, Dict, Optional
from retracemem.schemas import ModelCallTrace


def calculate_cache_key(
    provider: str,
    model_id: str,
    model_revision_or_api_version: str | None,
    prompt_template_hash: str | None,
    response_schema_version: str | None,
    parser_version: str | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    input_hash: str | None,
    condition_context_hash: str | None,
    temporal_context_hash: str | None,
) -> str:
    """
    Computes a strict SHA256 hash representing the unique cache key from the 13 configuration fields.
    """
    payload = {
        "provider": provider,
        "model_id": model_id,
        "model_revision_or_api_version": model_revision_or_api_version,
        "prompt_template_hash": prompt_template_hash,
        "response_schema_version": response_schema_version,
        "parser_version": parser_version,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "input_hash": input_hash,
        "condition_context_hash": condition_context_hash,
        "temporal_context_hash": temporal_context_hash,
    }
    # Deterministic JSON serialization
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class JSONLCache:
    """
    An append-only JSONL event logger and replay-safe parsing cache.
    Only successful and replay-eligible traces are cached in-memory for replaying.
    """
    def __init__(self, cache_path: str) -> None:
        self.cache_path = cache_path
        self.cache: Dict[str, ModelCallTrace] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_path):
            return
        
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    trace = self._deserialize_trace(data)
                    
                    # Retrieve or recalculate cache key
                    cache_key = data.get("cache_key")
                    if not cache_key:
                        cache_key = calculate_cache_key(
                            provider=trace.provider,
                            model_id=trace.model_id,
                            model_revision_or_api_version=trace.model_revision_or_api_version,
                            prompt_template_hash=trace.prompt_template_hash,
                            response_schema_version=trace.response_schema_version,
                            parser_version=trace.parser_version,
                            temperature=trace.temperature,
                            top_p=trace.top_p,
                            max_tokens=trace.max_tokens,
                            seed=trace.seed,
                            input_hash=trace.input_hash,
                            condition_context_hash=trace.condition_context_hash,
                            temporal_context_hash=trace.temporal_context_hash,
                        )
                    
                    # Replay-safe check: only status == "success" and eligible_for_replay == True
                    if trace.status == "success" and trace.eligible_for_replay:
                        self.cache[cache_key] = trace
                    else:
                        # If a newer record with status != success or eligible_for_replay == False is encountered,
                        # do we invalidate the cached successful one?
                        # Usually, for a replay-safe cache, once we have a valid replayable item, we can reuse it,
                        # but in append-only logs, if a later attempt explicitly says "eligible_for_replay=False",
                        # or if we want the absolute latest state, we might pop it.
                        # However, typically replay cache serves to avoid duplicate API calls.
                        # Thus, if a successful response exists, it's reusable.
                        # Let's keep it simple: we populate the cache with success & eligible_for_replay records.
                        pass
                except Exception:
                    continue

    def _deserialize_trace(self, data: dict[str, Any]) -> ModelCallTrace:
        fields = {f.name for f in dataclasses.fields(ModelCallTrace)}
        kwargs = {k: v for k, v in data.items() if k in fields}
        return ModelCallTrace(**kwargs)

    def lookup(self, cache_key: str) -> Optional[ModelCallTrace]:
        """
        Looks up a replay-safe trace by its cache key.
        """
        return self.cache.get(cache_key)

    def log_and_write(self, cache_key: str, trace: ModelCallTrace) -> None:
        """
        Appends the ModelCallTrace to the JSONL log file.
        Updates the in-memory cache if the trace is a success and eligible for replay.
        """
        data = dataclasses.asdict(trace)
        data["cache_key"] = cache_key
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(os.path.abspath(self.cache_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            
        if trace.status == "success" and trace.eligible_for_replay:
            self.cache[cache_key] = trace
