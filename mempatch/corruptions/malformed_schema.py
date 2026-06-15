from __future__ import annotations
from typing import Any

def corrupt(prediction: dict[str, Any]) -> dict[str, Any]:
    corrupted = dict(prediction)
    corrupted["response"] = dict(prediction["response"])
    # Corrupt memory_state to be a list instead of a dict to violate schema rules
    corrupted["response"]["memory_state"] = ["corrupted", "list", "instead", "of", "dict"]
    return corrupted
