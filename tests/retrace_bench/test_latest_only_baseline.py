from benchmark.retrace_bench.generation.expand_scenarios import expand_templates
from benchmark.retrace_bench.baselines.latest_only import LatestOnlyBaseline
from benchmark.retrace_bench.schemas import Prediction


def test_latest_only_baseline():
    scenarios = expand_templates(num_scenarios=2, seed=7)
    model = LatestOnlyBaseline()
    
    for scen in scenarios:
        for q in scen.probe_queries:
            pred = model.predict(scen, q)
            assert isinstance(pred, Prediction)
            assert pred.scenario_id == scen.scenario_id
            assert pred.query_id == q.query_id
            assert pred.predicted_answer in ("A", "B", "C", "D")
