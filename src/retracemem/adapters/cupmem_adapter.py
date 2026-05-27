from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from retracemem.adapters.stale_adapter import StaleAdapter
from retracemem.schemas import EvidenceRecord, QueryRecord


class CUPMemAdapter(StaleAdapter):
    """Adapter for CUPMem data formats, aligning with STALE schemas."""

    def __init__(self, reference_root: str | Path = "reference/STALE") -> None:
        super().__init__(reference_root)
