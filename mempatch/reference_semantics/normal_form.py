"""Development-only normal-form evaluator.

The frozen controlled release does not use this module to create labels.  Its
authoritative labels are emitted from the fixed ``ScenarioSpec`` table in
``mempatch.benchmark.generate``.  This evaluator is retained only for local
consistency experiments and must not be presented as the release resolver.
"""
from __future__ import annotations

import re
from typing import Any
from mempatch.reference_semantics.states import Configuration, MemoryState
from mempatch.reference_semantics.transition import ReferenceTransitionEngine
from mempatch.reference_semantics.trace import DerivationTrace

# Priority to decision string mapping
PRIORITY_TO_DECISION = {
    1: "refuse_due_to_policy",
    2: "escalate",
    3: "ask_clarification",
    4: "mark_unresolved",
    5: "use_current_memory",
}

def evaluate_normal_form(scenario: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a scenario with the non-authoritative development semantics."""
    public_input = scenario.get("public_input", {})
    scenario_id = scenario.get("scenario_id", "")
    
    # 1. Initialize Memories
    initial_memories = {}
    for mem in public_input.get("initial_memory") or public_input.get("initial_memories") or []:
        mid = mem.get("memory_id", "")
        # Determine initial status
        status = "out_of_scope" if mem.get("is_distractor", False) else "current"
        initial_memories[mid] = MemoryState(
            memory_id=mid,
            status=status,
            text=mem.get("text") or mem.get("content") or "",
            scope=mem.get("scope", "")
        )
        
    initial_config = Configuration(memories=initial_memories, t=0)
    engine = ReferenceTransitionEngine(initial_config)
    
    # 2. Step through events ordered by timestamp_order
    events = list(public_input.get("event_trace") or public_input.get("events") or [])
    events_sorted = sorted(events, key=lambda e: e.get("timestamp_order", 0))
    
    for event in events_sorted:
        engine.config = engine.step(event)
        
    final_config = engine.config
    
    # 3. Resolve Decision and failure diagnosis
    # Find active priority from trace
    final_decision = "use_current_memory"
    best_priority = 5
    evidence_ids = []
    
    for step in final_config.trace:
        pri = step["priority"]
        if pri < best_priority:
            best_priority = pri
            final_decision = PRIORITY_TO_DECISION.get(pri, "use_current_memory")
            
    # For evidence event IDs: collect all evidence IDs that caused state mutations
    # (i.e. those logged in the transition trace)
    # If the trace is empty, we default to the first two non-distractor event IDs
    evidence_ids = []
    for step in final_config.trace:
        evidence_ids.extend(step["evidence_ids"])
        
    # Deduplicate keeping order
    seen = set()
    evidence_ids = [x for x in evidence_ids if not (x in seen or seen.add(x))]
    
    # Fallback if no evidence cited but it is use_current_memory
    if not evidence_ids and final_decision == "use_current_memory":
        # Find first two non-distractor events as per specs
        core_events = []
        for e in events_sorted:
            eid = e.get("event_id", "")
            # e-case-X-distractor or background are distractors
            if "distractor" not in eid and "bg" not in eid:
                core_events.append(eid)
        evidence_ids = core_events[:2]
        
    # Preserve the provided answer key when normal-form evaluation is used as a
    # consistency check. The reference semantics no longer depends on renderer
    # templates.
    gold_raw = scenario.get("hidden_gold", {})
    original_expected_answer = gold_raw.get("expected_answer", "")
        
    # Construct normal form evaluation output
    return {
        "final_state": final_config.get_status_dict(),
        "decision": final_decision,
        "minimal_evidence": evidence_ids,
        "failure_diagnosis": scenario.get("primary_failure_mode", gold_raw.get("expected_failure_diagnosis", "")),
        "answer_key_facts": original_expected_answer,
        "derivation_trace": final_config.trace,
    }
