from benchmark.retrace_bench.baselines.latest_only import LatestOnlyBaseline
from benchmark.retrace_bench.baselines.retrieve_all import RetrieveAllBaseline
from benchmark.retrace_bench.baselines.directjudge import DirectJudgeBaseline
from benchmark.retrace_bench.baselines.crud_memory import CRUDMemoryBaseline
from benchmark.retrace_bench.baselines.prompt_proposer import PromptProposerBaseline
from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


def get_baseline(name: str, provider: BaseLLMProvider | None = None):
    name_lower = name.lower()
    if name_lower == "latest_only":
        return LatestOnlyBaseline()
    elif name_lower == "retrieve_all":
        return RetrieveAllBaseline()
    elif name_lower == "directjudge" or name_lower == "directjudge_api_stub":
        return DirectJudgeBaseline(provider=provider)
    elif name_lower == "crud_memory":
        return CRUDMemoryBaseline()
    elif name_lower == "prompt_proposer" or name_lower == "prompt_proposer_api_stub":
        return PromptProposerBaseline(provider=provider)
    else:
        raise ValueError(f"Unknown baseline name: {name}")
