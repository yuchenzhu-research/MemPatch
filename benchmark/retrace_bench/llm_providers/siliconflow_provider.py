import os

from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


class SiliconFlowProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "deepseek-ai/DeepSeek-V3"):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return '{"error": "SILICONFLOW_API_KEY is not set"}'
        try:
            import openai

            client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.siliconflow.cn/v1",
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except ImportError:
            return '{"error": "openai package not installed"}'

