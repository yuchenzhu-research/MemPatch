from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml
from retracemem.providers.base import BaseLLMProvider, MockLLMProvider
from retracemem.providers.openai_compatible import OpenAICompatibleProvider

def load_providers_config() -> dict[str, Any]:
    """Loads configs/providers.yaml from repo root."""
    # Find repository root. We can traverse up from this file's location.
    current_file = Path(__file__).resolve()
    # Path is src/retracemem/providers/provider_factory.py, repo root is 4 parents up
    repo_root = current_file.parents[3]
    config_path = repo_root / "configs" / "providers.yaml"
    
    if not config_path.exists():
        return {}
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "providers" in data:
                return data["providers"]
    except Exception:
        pass
    return {}

def get_provider(
    provider_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """
    Factory to retrieve an LLM provider instance.
    Loads provider metadata dynamically from configs/providers.yaml.
    Fails closed if the provider is unknown (unless 'mock').
    """
    provider_name_lower = provider_name.lower()
    if provider_name_lower == "mock":
        return MockLLMProvider(**kwargs)
        
    config = load_providers_config()
    if provider_name_lower not in config:
        raise ValueError(f"Unsupported provider: '{provider_name}'. Must be registered in configs/providers.yaml.")
        
    prov_meta = config[provider_name_lower]
    api_key_env = prov_meta.get("api_key_env")
    resolved_api_key = api_key
    
    if not resolved_api_key and api_key_env:
        resolved_api_key = os.environ.get(api_key_env)
        
    # If key is still missing, but we are running in live/non-mock mode, we will let the provider raise ValueError on generate
    resolved_base_url = base_url or prov_meta.get("default_base_url")
    
    # Store metadata on the provider instance for tracing
    provider_inst = OpenAICompatibleProvider(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        **kwargs,
    )
    # Inject metadata properties for tracing
    provider_inst._api_key_env = api_key_env
    provider_inst._provider_type = prov_meta.get("provider_type", "openai_compatible")
    
    return provider_inst
