"""MemPatch scaffold: learned revision response for shared-memory integration.

The trainable pipeline has three implementation roles (only the first two are
learned):

    scenario / event_trace / multi-subagent submissions
      -> Scenario View Builder              (learned)
      -> Revision Response Policy           (learned)
      -> Benchmark-grounded feedback        (training protocol)

The deterministic commit path (parser + RevisionGate + DPA via ``authorize``)
is internal to the scaffold: the model proposes; DPA authorizes; the benchmark
evaluates ``memory_state``.
``reward.py`` supplies benchmark-grounded training signal; the defeat-path ranker
is a future/optional extension.

The typed action vocabulary, ``RevisionGate``, and ``authorize(...)`` kernel are
reused from ``retracemem``, never duplicated.
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
