import os
from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            if "status" in prompt.lower() or "directjudge" in prompt.lower():
                return '{"status": "AUTHORIZED", "rationale": "Mocked Gemini DirectJudge prediction"}'
            if "action" in prompt.lower():
                return '{"action": "NO_REVISION", "target_id": "none"}'
            return "Mocked Gemini response"
        try:
            # Placeholder for actual Gemini SDK call (google-generativeai)
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            return "Error: google-generativeai package not installed. Mocked response returned."
