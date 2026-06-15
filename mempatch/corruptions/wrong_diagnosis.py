from __future__ import annotations
from typing import Any

def corrupt(prediction: dict[str, Any]) -> dict[str, Any]:
    corrupted = dict(prediction)
    corrupted["response"] = dict(prediction["response"])
    orig = corrupted["response"].get("failure_diagnosis")
    if orig == "under_update":
        corrupted["response"]["failure_diagnosis"] = "conflict_collapse"
    else:
        corrupted["response"]["failure_diagnosis"] = "under_update"
    return corrupted
