from __future__ import annotations
from typing import Any

def corrupt(prediction: dict[str, Any]) -> dict[str, Any]:
    corrupted = dict(prediction)
    corrupted["response"] = dict(prediction["response"])
    orig_state = corrupted["response"].get("memory_state", {})
    new_state = {}
    for mid, status in orig_state.items():
        if status == "current":
            new_state[mid] = "blocked"
        else:
            new_state[mid] = "current"
    corrupted["response"]["memory_state"] = new_state
    return corrupted
