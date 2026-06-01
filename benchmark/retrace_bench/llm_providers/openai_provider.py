import os
from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            # Return a generic stub response that is valid JSON/YAML depending on prompt context
            if "status" in prompt.lower() or "directjudge" in prompt.lower():
                # DirectJudge stub outputting JSON status
                return '{"status": "AUTHORIZED", "rationale": "Mocked OpenAI DirectJudge prediction"}'
            if "action" in prompt.lower():
                # Action proposal stub
                return '{"action": "NO_REVISION", "target_id": "none"}'
            return "Mocked OpenAI response"
        
        # Real client initialization (lazy load)
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return resp.choices[0].message.content or ""
        except ImportError:
            return "Error: openai package not installed. Mocked response returned."
