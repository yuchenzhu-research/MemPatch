import os
from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            if "status" in prompt.lower() or "directjudge" in prompt.lower():
                return '{"status": "AUTHORIZED", "rationale": "Mocked DeepSeek DirectJudge prediction"}'
            if "action" in prompt.lower():
                return '{"action": "NO_REVISION", "target_id": "none"}'
            return "Mocked DeepSeek response"
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return resp.choices[0].message.content or ""
        except ImportError:
            return "Error: openai package not installed for DeepSeek. Mocked response returned."
