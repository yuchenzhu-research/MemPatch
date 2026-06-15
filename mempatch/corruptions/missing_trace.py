from __future__ import annotations
from typing import Any

def corrupt(prediction: dict[str, Any]) -> dict[str, Any]:
    corrupted = dict(prediction)
    corrupted["response"] = dict(prediction["response"])
    # Delete the answer field to simulate missing execution trace / response data
    if "answer" in corrupted["response"]:
        del corrupted["response"]["answer"]
    return corrupted
