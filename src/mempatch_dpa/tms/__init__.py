"""Truth-maintenance-inspired DPA authorization for MemPatch."""

from mempatch_dpa.tms.authorization import AuthorizationEngine
from mempatch_dpa.tms.gate import RevisionGate

__all__ = ["RevisionGate", "AuthorizationEngine"]
