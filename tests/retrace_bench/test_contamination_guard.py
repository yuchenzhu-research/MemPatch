import pytest
from pathlib import Path
from benchmark.retrace_bench.utils.contamination import check_contamination


def test_contamination_guard():
    # Safe path
    check_contamination("data/retrace_learn/train.jsonl")
    
    # Unsafe paths
    with pytest.raises(RuntimeError, match="ReTrace-Bench is evaluation-only"):
        check_contamination("data/retrace_bench/v1/scenarios.jsonl")

    with pytest.raises(RuntimeError, match="ReTrace-Bench is evaluation-only"):
        check_contamination(Path("data/retrace_bench/main_3000_en/scenarios.jsonl"))
