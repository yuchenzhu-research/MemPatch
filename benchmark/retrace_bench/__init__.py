"""ReTrace-Bench public evaluator package."""

__version__ = "1.1.0"

from benchmark.retrace_bench.api import (
    AUXILIARY_METRICS,
    DECISIONS,
    FAILURE_MODES,
    HEADLINE_METRICS,
    MEMORY_STATUSES,
    evaluate_predictions,
    load_predictions,
    load_scenarios,
    normalize_prediction,
)
from benchmark.retrace_bench.utils.contamination import check_contamination

__all__ = [
    "AUXILIARY_METRICS",
    "DECISIONS",
    "FAILURE_MODES",
    "HEADLINE_METRICS",
    "MEMORY_STATUSES",
    "check_contamination",
    "evaluate_predictions",
    "load_predictions",
    "load_scenarios",
    "normalize_prediction",
]
