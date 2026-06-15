from __future__ import annotations
from typing import Any

def corrupt(prediction: dict[str, Any]) -> dict[str, Any]:
    corrupted = dict(prediction)
    corrupted["response"] = dict(prediction["response"])
    orig = corrupted["response"].get("decision")
    # Toggle decision to a different valid one
    if orig == "use_current_memory":
        corrupted["response"]["decision"] = "refuse_due_to_policy"
    else:
        corrupted["response"]["decision"] = "use_current_memory"
    return corrupted
