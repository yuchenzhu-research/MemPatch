from __future__ import annotations
from typing import Any

def corrupt(prediction: dict[str, Any]) -> dict[str, Any]:
    corrupted = dict(prediction)
    corrupted["response"] = dict(prediction["response"])
    # Clear out the evidence citations to simulate false negatives
    corrupted["response"]["evidence_event_ids"] = []
    return corrupted
