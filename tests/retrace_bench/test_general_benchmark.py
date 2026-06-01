import json

from scripts.generate_retrace_bench_blueprints import blueprint
from scripts.render_retrace_bench_dataset import render_one
from scripts.validate_retrace_bench_dataset import validate_dataset
from benchmark.retrace_bench.scorers_general import score_prediction


def test_general_blueprint_render_validate():
    import random

    rows = [render_one(blueprint(i, random.Random(42 + i)), random.Random(7)) for i in range(40)]
    report = validate_dataset(rows)
    assert report["errors"] == []
    assert report["rates"]["distractors"] >= 0.40
    assert report["rates"]["cross_scope"] >= 0.30


def test_general_gold_perfect_prediction_scores_one():
    import random

    scenario = render_one(blueprint(0, random.Random(42)), random.Random(7))
    gold = scenario["hidden_gold"]
    prediction = {
        "response": {
            "answer": gold["expected_answer"],
            "decision": gold["expected_decision"],
            "memory_state": gold["expected_memory_state"],
            "evidence_event_ids": gold["expected_evidence_event_ids"],
            "failure_diagnosis": gold["expected_failure_diagnosis"],
        }
    }
    metrics = score_prediction(scenario, prediction)
    assert metrics["answer_key_fact_accuracy"] == 1.0
    assert metrics["answer_exact_match"] == 1.0
    assert metrics["memory_state_accuracy"] == 1.0
    assert metrics["evidence_f1"] == 1.0
    assert json.dumps(scenario)
