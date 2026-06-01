from typing import List
from benchmark.retrace_bench.schemas import Scenario, Prediction
from benchmark.retrace_bench.baselines import get_baseline
from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


def run_evaluation_loop(
    scenarios: List[Scenario],
    baseline_name: str,
    provider: BaseLLMProvider | None = None
) -> List[Prediction]:
    model = get_baseline(baseline_name, provider=provider)
    predictions = []

    for scen in scenarios:
        for q in scen.probe_queries:
            pred = model.predict(scen, q)
            predictions.append(pred)

    return predictions
