"""Truth-maintenance-inspired DPA authorization for MemPatch."""

from mempatch.dpa.tms.authorization import AuthorizationEngine
from mempatch.dpa.tms.gate import RevisionGate

__all__ = ["RevisionGate", "AuthorizationEngine"]
