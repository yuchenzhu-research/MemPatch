"""MemPatch method: revision module + DPA authorization kernel.

- ``mempatch.revision`` — Path A/B revision (view → proposer → projection)
- ``mempatch.dpa`` — deterministic Defeat-Path Authorization (``authorize``)

Official benchmark scoring is in the separate ``benchmark`` package.
"""

from mempatch.dpa import authorize

__all__ = ["authorize"]
