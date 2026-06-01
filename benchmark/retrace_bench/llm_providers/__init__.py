from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider
from benchmark.retrace_bench.llm_providers.openai_provider import OpenAIProvider
from benchmark.retrace_bench.llm_providers.gemini_provider import GeminiProvider
from benchmark.retrace_bench.llm_providers.anthropic_provider import AnthropicProvider
from benchmark.retrace_bench.llm_providers.deepseek_provider import DeepSeekProvider


def get_provider(
    provider_name: str, model_name: str, api_key: str | None = None
) -> BaseLLMProvider:
    name_lower = provider_name.lower()
    if "openai" in name_lower:
        return OpenAIProvider(api_key=api_key, model=model_name)
    elif "gemini" in name_lower:
        return GeminiProvider(api_key=api_key, model=model_name)
    elif "anthropic" in name_lower:
        return AnthropicProvider(api_key=api_key, model=model_name)
    elif "deepseek" in name_lower:
        return DeepSeekProvider(api_key=api_key, model=model_name)
    else:
        # Default fallback to OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model_name)
