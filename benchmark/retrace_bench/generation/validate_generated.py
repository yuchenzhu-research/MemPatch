from typing import List, Tuple
from benchmark.retrace_bench.schemas import Scenario, ValidationReport, ProbeQuery
from benchmark.retrace_bench.taxonomy import FinalStatus, RevisionActionType, ProbeType


def validate_scenarios(scenarios: List[Scenario]) -> Tuple[ValidationReport, List[Scenario], List[Scenario]]:
    """Validate scenarios according to the 10 core rules.

    Returns:
        ValidationReport, accepted_scenarios, rejected_scenarios
    """
    errors = []
    accepted = []
    rejected = []

    seen_query_ids = set()

    for idx, scen in enumerate(scenarios):
        scen_errors = []

        # 1. every scenario has exactly 4 probe queries
        if len(scen.probe_queries) != 4:
            scen_errors.append(f"Scenario {scen.scenario_id} has {len(scen.probe_queries)} queries (expected 4).")

        # 2. all probe types are present
        probe_types = {q.probe_type for q in scen.probe_queries}
        expected_types = {ProbeType.STATE_RESOLUTION, ProbeType.PREMISE_RESISTANCE, ProbeType.POLICY_ADAPTATION, ProbeType.AUDIT_LOCALIZATION}
        if not expected_types.issubset(probe_types):
            scen_errors.append(f"Scenario {scen.scenario_id} misses probe types. Present: {probe_types}")

        # 3. all gold final statuses use allowed labels
        for memory_id, status in scen.gold_final_statuses.items():
            if status not in FinalStatus:
                scen_errors.append(f"Scenario {scen.scenario_id} has invalid status label '{status}' for belief '{memory_id}'.")

        # 4. all gold actions use allowed labels
        for action in scen.gold_revision_actions:
            if action.action_type not in RevisionActionType:
                scen_errors.append(f"Scenario {scen.scenario_id} has invalid action label '{action.action_type}'.")

        # 5. memory ids are unique
        memory_ids = [entry.entry_id for entry in scen.memory_snapshot]
        if len(memory_ids) != len(set(memory_ids)):
            scen_errors.append(f"Scenario {scen.scenario_id} has duplicate memory entry IDs: {memory_ids}")

        # 6. query ids are unique
        for q in scen.probe_queries:
            if q.query_id in seen_query_ids:
                scen_errors.append(f"Duplicate query ID '{q.query_id}' across dataset.")
            seen_query_ids.add(q.query_id)

        # 7. evidence ids referenced by audit queries exist
        evidence_ids = {entry.entry_id for entry in scen.memory_snapshot if entry.entry_type == "evidence"}
        for q in scen.probe_queries:
            if q.probe_type == ProbeType.AUDIT_LOCALIZATION:
                # The gold answer points to option A, check if option A exists in evidence list
                ans_val = q.options.get(q.gold_answer, "")
                if ans_val and ans_val not in evidence_ids and q.gold_answer != "D":
                    scen_errors.append(f"Audit query in scenario {scen.scenario_id} references missing evidence '{ans_val}'")

        # 9. no examples are marked as ReTrace-Learn training data
        if scen.metadata.get("is_training") or "train" in scen.metadata.get("split", "").lower():
            scen_errors.append(f"Scenario {scen.scenario_id} is incorrectly marked as ReTrace-Learn training data.")

        if scen_errors:
            errors.extend(scen_errors)
            rejected.append(scen)
        else:
            accepted.append(scen)

    # 8. revision family distribution is non-empty
    families_present = {scen.revision_family for scen in scenarios}
    if not families_present:
        errors.append("Dataset revision family distribution is empty.")

    # 10. manifest says evaluation-only (checked externally in build/validate script)

    is_valid = len(errors) == 0
    report = ValidationReport(
        is_valid=is_valid,
        errors=errors,
        num_checked=len(scenarios)
    )

    return report, accepted, rejected
