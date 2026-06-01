import json
import os
import time

from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


class SiliconFlowProvider(BaseLLMProvider):
    # Bounded retry for transient gateway/network errors (e.g. load-balancer
    # "400 Bad Request" HTML responses, rate limits, 5xx). A persistent failure
    # degrades to an error JSON so a single bad call does not abort a long run;
    # the scorer's format_failure_rate then flags that case.
    MAX_ATTEMPTS = 4

    def __init__(self, api_key: str | None = None, model: str = "deepseek-ai/DeepSeek-V3"):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return '{"error": "SILICONFLOW_API_KEY is not set"}'
        try:
            import openai
        except ImportError:
            return '{"error": "openai package not installed"}'

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.siliconflow.cn/v1",
        )
        last_err: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # transient gateway/network/rate-limit errors
                last_err = exc
                if attempt < self.MAX_ATTEMPTS - 1:
                    time.sleep(min(2 ** attempt, 8))
        return json.dumps({"error": f"siliconflow request failed after {self.MAX_ATTEMPTS} attempts: {type(last_err).__name__}"})
