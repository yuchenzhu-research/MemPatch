"""Evaluation helpers and metric wrappers."""

from retracemem.evaluation.cost_tracker import CostTracker
from retracemem.evaluation.jsonl import read_jsonl, records_to_jsonable, write_jsonl
from retracemem.evaluation.records import evaluation_record_from_backend_output
from retracemem.evaluation.diagnostics import (
    calculate_unsupported_revision_rate,
    calculate_protected_belief_preservation,
)
from retracemem.evaluation.report_tables import generate_ablation_table

__all__ = [
    "CostTracker",
    "evaluation_record_from_backend_output",
    "read_jsonl",
    "records_to_jsonable",
    "write_jsonl",
    "calculate_unsupported_revision_rate",
    "calculate_protected_belief_preservation",
    "generate_ablation_table",
]

