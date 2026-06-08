"""MemPatch Revision Module — learned revision response for RMI.

Pipeline (see ``AGENTS.md``):

  scenario view → learned proposer → DPA runtime → benchmark projection

``mempatch_learn`` holds the learned side; ``mempatch_dpa`` authorizes updates.
MemPatch-Bench (``benchmark/``) scores the final ``response`` object.
"""
from __future__ import annotations

from mempatch_learn.schemas import (
    CANDIDATE_PATH_TYPES,
    CANONICAL_ACTIONS,
    FINAL_STATUSES,
    RevisionAction,
    SchemaValidationError,
)

__all__ = [
    "CANONICAL_ACTIONS",
    "FINAL_STATUSES",
    "CANDIDATE_PATH_TYPES",
    "RevisionAction",
    "SchemaValidationError",
]
