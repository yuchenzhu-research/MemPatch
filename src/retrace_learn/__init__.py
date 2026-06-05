"""MemPatch Revision Module — learned revision response for RMI.

Four internal roles (see ``docs/mempatch_revision_module.md``):

  Scenario View Builder → Revision Response Policy
  → DPA-Consistent Projection → Benchmark-grounded Feedback

The model proposes benchmark-compatible revision responses; DPA authorizes;
MemPatch-Bench evaluates ``memory_state``. Implementation paths:
``src/retrace_learn/`` (learned roles) + ``src/retracemem/`` (DPA projection).
"""
from __future__ import annotations

from retrace_learn.schemas import (
    CANDIDATE_PATH_TYPES,
    CANONICAL_ACTIONS,
    FINAL_STATUSES,
    GraphExtractionExample,
    RevisionAction,
    RLRolloutExample,
    SchemaValidationError,
    TypedRevisionExample,
)

__all__ = [
    "CANONICAL_ACTIONS",
    "FINAL_STATUSES",
    "CANDIDATE_PATH_TYPES",
    "RevisionAction",
    "GraphExtractionExample",
    "TypedRevisionExample",
    "RLRolloutExample",
    "SchemaValidationError",
]
