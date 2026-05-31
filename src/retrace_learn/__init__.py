"""ReTrace-Learn: a trainable shared-memory revision authorization framework.

ReTrace-Learn upgrades ReTrace Stage A from a prompt-scaffolded proposer to a
learnable pipeline:

    raw dialogue / multi-subagent submissions
      -> learned graph extractor            (Module 1)
      -> learned typed revision proposer     (Module 2)
      -> parser + RevisionGate + DPA runtime (Module 3, deterministic kernel)
      -> DPA-in-the-loop reward              (Module 4)
      -> optional learned defeat-path ranker (Section E)

It is compatible with the canonical ReTrace runtime: the typed action vocabulary,
``RevisionGate``, and ``authorize(...)`` kernel are reused, never duplicated.
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
