from pathlib import Path
import pytest
from retrace_learn.training import check_contamination


def test_contamination_guard_path_rejects():
    # 1. Config path itself under data/retrace_bench rejects
    with pytest.raises(RuntimeError, match="ReTrace-Bench is evaluation-only"):
        check_contamination(Path("data/retrace_bench/v1/config.yaml"))

    with pytest.raises(RuntimeError, match="ReTrace-Bench is evaluation-only"):
        check_contamination("data/retrace_bench/v1/config.yaml")


def test_contamination_guard_field_rejects():
    # 2. YAML field rejects
    config = {
        "train_data": "data/retrace_bench/v1/public_dev.jsonl",
        "epochs": 3,
    }
    with pytest.raises(RuntimeError, match="ReTrace-Bench is evaluation-only"):
        check_contamination(config)


def test_contamination_guard_nested_list_rejects():
    # 3. Nested list containing retrace_bench path rejects
    config = {
        "train_data": ["data/retrace_learn/v1/train.jsonl", "data/retrace_bench/v1/public_dev.jsonl"],
        "epochs": 3,
    }
    with pytest.raises(RuntimeError, match="ReTrace-Bench is evaluation-only"):
        check_contamination(config)


def test_contamination_guard_normal_passes():
    # 4. Normal data/retrace_learn path passes
    config = {
        "train_data": "data/retrace_learn/v1/boundary_audit/boundary_audit_dev.jsonl",
        "epochs": 3,
        "nested": {
            "path": Path("data/retrace_learn/v1/internal_dev/controlled_ab_cases.json")
        }
    }
    # Should not raise any exception
    check_contamination(config)
