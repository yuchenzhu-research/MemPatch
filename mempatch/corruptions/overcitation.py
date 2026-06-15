from __future__ import annotations
from typing import Any

def corrupt(prediction: dict[str, Any], scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    corrupted = dict(prediction)
    corrupted["response"] = dict(prediction["response"])
    orig = list(corrupted["response"].get("evidence_event_ids", []))
    
    # Inject a distractor event ID
    distractor_id = "e-case-distractor-overcite"
    if scenario:
        # Try to find a non-gold event from the scenario
        events = scenario.get("public_input", {}).get("event_trace", [])
        gold_evs = scenario.get("hidden_gold", {}).get("expected_evidence_event_ids", [])
        for e in events:
            eid = e.get("event_id")
            if eid and eid not in gold_evs and eid not in orig:
                distractor_id = eid
                break
    
    orig.append(distractor_id)
    corrupted["response"]["evidence_event_ids"] = orig
    return corrupted
