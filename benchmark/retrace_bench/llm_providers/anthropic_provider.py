import os
from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            if "status" in prompt.lower() or "directjudge" in prompt.lower():
                return '{"status": "AUTHORIZED", "rationale": "Mocked Anthropic DirectJudge prediction"}'
            if "action" in prompt.lower():
                return '{"action": "NO_REVISION", "target_id": "none"}'
            return "Mocked Anthropic response"
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return "".join([block.text for block in message.content])
        except ImportError:
            return "Error: anthropic package not installed. Mocked response returned."
