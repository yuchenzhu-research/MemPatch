from typing import List
from benchmark.retrace_bench.schemas_v2 import ScenarioV2, ManifestV2


def validate_scenario_v2(scenario: ScenarioV2) -> None:
    """Validates a single ScenarioV2 instance against schema and integrity rules.
    Raises ValueError on validation failure.
    """
    errors = []

    # 1. Version starts with "2"
    if not scenario.version.startswith("2"):
        errors.append(f"Scenario version '{scenario.version}' must start with '2'")

    # 2. Unique scenario_id
    if not scenario.scenario_id:
        errors.append("scenario_id is empty")

    # 3. Unique event_ids within scenario
    event_ids = set()
    for e in scenario.event_trace:
        if e.event_id in event_ids:
            errors.append(f"Duplicate event_id '{e.event_id}' in event_trace")
        event_ids.add(e.event_id)

    # 4. Unique memory_ids within scenario
    memory_ids = set()
    for m in scenario.memory_snapshot:
        if m.memory_id in memory_ids:
            errors.append(f"Duplicate memory_id '{m.memory_id}' in memory_snapshot")
        memory_ids.add(m.memory_id)

    # Collect memory_ids from lifecycle operations (e.g. CREATE ops)
    for op in scenario.hidden_memory_lifecycle.operations:
        memory_ids.add(op.memory_id)
        if op.target_memory_id:
            memory_ids.add(op.target_memory_id)
    for op in scenario.lifecycle_operations:
        memory_ids.add(op.memory_id)
        if op.target_memory_id:
            memory_ids.add(op.target_memory_id)

    # 5. Task reference_event_ids exist in event_trace
    for task in scenario.tasks:
        for ref_id in task.reference_event_ids:
            if ref_id not in event_ids:
                errors.append(f"Task '{task.task_id}' references non-existent event_id '{ref_id}'")

    # 6. Memory source_event_ids exist in event_trace
    for m in scenario.memory_snapshot:
        for src_id in m.source_event_ids:
            if src_id not in event_ids:
                errors.append(f"Memory '{m.memory_id}' source_event_id '{src_id}' does not exist in event_trace")

    # 7. Evidence event_ids exist in event_trace
    # In hidden memory lifecycle operations
    for op in scenario.hidden_memory_lifecycle.operations:
        for ev_id in op.evidence_event_ids:
            if ev_id not in event_ids:
                errors.append(f"Lifecycle operation memory '{op.memory_id}' evidence_event_id '{ev_id}' does not exist in event_trace")

    # In lifecycle operations
    for op in scenario.lifecycle_operations:
        for ev_id in op.evidence_event_ids:
            if ev_id not in event_ids:
                errors.append(f"Lifecycle operation memory '{op.memory_id}' evidence_event_id '{ev_id}' does not exist in event_trace")

    # In tasks gold_behavior
    for task in scenario.tasks:
        gb = task.gold_behavior
        # Check gold_evidence supporting_event_ids
        for ev_id in gb.gold_evidence.supporting_event_ids:
            if ev_id not in event_ids:
                errors.append(f"Task '{task.task_id}' gold_evidence supporting_event_id '{ev_id}' does not exist in event_trace")
        # Check structured actions evidence_event_ids
        for action in gb.gold_actions:
            for ev_id in action.evidence_event_ids:
                if ev_id not in event_ids:
                    errors.append(f"Task '{task.task_id}' action target '{action.target_id}' evidence_event_id '{ev_id}' does not exist in event_trace")

    # 8. Structured revision target IDs exist if visible in memory_snapshot
    for task in scenario.tasks:
        gb = task.gold_behavior
        for action in gb.gold_actions:
            # action.target_id should exist in memory_ids
            if action.target_id not in memory_ids:
                errors.append(f"Task '{task.task_id}' action references non-existent target_id '{action.target_id}'")
            if action.replacement_id and action.replacement_id not in memory_ids:
                errors.append(f"Task '{task.task_id}' action references non-existent replacement_id '{action.replacement_id}'")

    # 9. Contamination policy says evaluation_only
    policy_dict = scenario.metadata.contamination_policy
    policy_val = policy_dict.get("policy") or policy_dict.get("type") or policy_dict.get("status")
    # Also support check if "evaluation_only" string exists in policy_dict values
    has_eval_only = False
    for v in policy_dict.values():
        if isinstance(v, str) and "evaluation_only" in v.lower():
            has_eval_only = True
    if not has_eval_only and policy_val != "evaluation_only":
        errors.append("Contamination policy metadata must specify 'evaluation_only'")

    if errors:
        raise ValueError(f"Scenario '{scenario.scenario_id}' validation failed:\n" + "\n".join(errors))


def validate_manifest_v2(manifest: ManifestV2) -> None:
    """Validates a ManifestV2 instance against rules."""
    errors = []
    if not manifest.version.startswith("2"):
        errors.append(f"Manifest version '{manifest.version}' must start with '2'")
    
    # Check if contamination policy contains evaluation_only
    policy_dict = manifest.metadata.contamination_policy
    has_eval_only = False
    for v in policy_dict.values():
        if isinstance(v, str) and "evaluation_only" in v.lower():
            has_eval_only = True
    if not has_eval_only:
        errors.append("Manifest contamination policy metadata must specify 'evaluation_only'")

    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))
