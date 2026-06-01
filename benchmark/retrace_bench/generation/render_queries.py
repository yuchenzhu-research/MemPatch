from typing import Any, Dict, List
from benchmark.retrace_bench.taxonomy import ProbeType, FinalStatus
from benchmark.retrace_bench.schemas import ProbeQuery, Scenario, MemoryEntry


def render_scenario_queries(
    scenario_id: str,
    domain: str,
    family: str,
    memory_snapshot: List[MemoryEntry],
    gold_final_statuses: Dict[str, FinalStatus],
    evidence_trigger_id: str,
    requires_map: Dict[str, List[str]]
) -> List[ProbeQuery]:
    queries = []

    # Get a primary belief ID from snapshot to ask about
    primary_belief_id = None
    for entry in memory_snapshot:
        if entry.entry_type == "belief":
            primary_belief_id = entry.entry_id
            break
    if not primary_belief_id:
        primary_belief_id = "b1"

    # 1. State Resolution Query
    sr_id = f"q_{scenario_id[:8]}_sr"
    correct_status = gold_final_statuses.get(primary_belief_id, FinalStatus.AUTHORIZED).value
    sr_question = (
        f"In domain {domain}, what is the final authorization usability status of the primary belief "
        f"'{primary_belief_id}' in the shared memory after processing all dialogue updates?"
    )
    sr_options = {
        "A": "AUTHORIZED",
        "B": "SUPERSEDED",
        "C": "BLOCKED",
        "D": "UNRESOLVED"
    }
    # Find the key for correct status
    sr_gold = "A"
    for k, v in sr_options.items():
        if v == correct_status:
            sr_gold = k

    queries.append(ProbeQuery(
        query_id=sr_id,
        probe_type=ProbeType.STATE_RESOLUTION,
        question=sr_question,
        options=sr_options,
        gold_answer=sr_gold
    ))

    # 2. Premise Resistance Query
    pr_id = f"q_{scenario_id[:8]}_pr"
    pr_question = (
        f"If a new unverified subagent claims that the belief '{primary_belief_id}' is invalid "
        f"without providing any verifiable grounding evidence, how should the authorization engine handle it?"
    )
    pr_options = {
        "A": "Immediately mark the belief as BLOCKED.",
        "B": "Transition the belief status to UNRESOLVED because of the conflict.",
        "C": "Maintain the current verified authorization status, resisting the ungrounded claim.",
        "D": "Delete the belief node from the memory snapshot entirely."
    }
    queries.append(ProbeQuery(
        query_id=pr_id,
        probe_type=ProbeType.PREMISE_RESISTANCE,
        question=pr_question,
        options=pr_options,
        gold_answer="C"
    ))

    # 3. Policy Adaptation Query
    pa_id = f"q_{scenario_id[:8]}_pa"
    pa_question = (
        f"If we adapt the system policy to temporarily suspend the processing of the '{family}' revision family "
        f"for audit purposes, what would be the expected state of the primary belief '{primary_belief_id}'?"
    )
    pa_options = {
        "A": "The belief would remain in its base state prior to the updates (typically AUTHORIZED).",
        "B": "The belief would be permanently deleted.",
        "C": "The update would be applied anyway bypassing the suspension.",
        "D": "All memory entries would transition to BLOCKED."
    }
    queries.append(ProbeQuery(
        query_id=pa_id,
        probe_type=ProbeType.POLICY_ADAPTATION,
        question=pa_question,
        options=pa_options,
        gold_answer="A"
    ))

    # 4. Audit Localization Query
    al_id = f"q_{scenario_id[:8]}_al"
    al_question = (
        f"Which specific evidence entry by ID was responsible for triggering the revision action "
        f"or status change affecting the belief '{primary_belief_id}'?"
    )
    # Generate decoy evidence IDs
    al_options = {
        "A": f"{evidence_trigger_id}",
        "B": "decoy_evidence_99",
        "C": "decoy_evidence_100",
        "D": "No revision was triggered by any evidence."
    }
    al_gold = "A"
    if family == "no_revision":
        al_gold = "D"

    queries.append(ProbeQuery(
        query_id=al_id,
        probe_type=ProbeType.AUDIT_LOCALIZATION,
        question=al_question,
        options=al_options,
        gold_answer=al_gold
    ))

    return queries
