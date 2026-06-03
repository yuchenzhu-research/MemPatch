"""ReTrace-Learn: a trainable shared-memory revision authorization framework.

ReTrace-Learn upgrades ReTrace Stage A from a prompt-scaffolded proposer to a
learnable pipeline. v1 has three paper-facing stages (only the first two are
learned):

    raw dialogue / multi-subagent submissions
      -> Graph Builder                       (Stage 1, learned)
      -> Proposal Policy                      (Stage 2, learned)
      -> DPA-guided RSFT / DPO                (Stage 3, training protocol)

The deterministic commit path (parser + RevisionGate + DPA runtime, i.e.
ReTrace-Engine) is an implementation detail of stages 2-3, not a separate paper
module. ``reward.py`` is the DPA-guided training signal; the defeat-path ranker
is a future/optional extension.

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
