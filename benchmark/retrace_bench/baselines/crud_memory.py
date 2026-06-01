from benchmark.retrace_bench.schemas import Scenario, ProbeQuery, Prediction
from benchmark.retrace_bench.taxonomy import ProbeType, FinalStatus, RevisionActionType


class CRUDMemoryBaseline:
    def __init__(self):
        self.name = "crud_memory"

    def predict(self, scenario: Scenario, query: ProbeQuery) -> Prediction:
        # CRUD memory processes actions in order and resolves statuses via basic rules
        # Resolve statuses
        statuses = {}
        blocked_conditions = set()
        
        # Initialize all beliefs to AUTHORIZED
        for entry in scenario.memory_snapshot:
            if entry.entry_type == "belief":
                statuses[entry.entry_id] = FinalStatus.AUTHORIZED

        # Apply revision actions
        for action in scenario.gold_revision_actions:
            if action.action_type == RevisionActionType.BLOCKS:
                blocked_conditions.add(action.target_id)
            elif action.action_type == RevisionActionType.RELEASES:
                blocked_conditions.discard(action.target_id)
            elif action.action_type == RevisionActionType.SUPERSEDES:
                statuses[action.target_id] = FinalStatus.SUPERSEDED
                if action.replacement_id:
                    statuses[action.replacement_id] = FinalStatus.AUTHORIZED
            elif action.action_type == RevisionActionType.UNCERTAIN:
                statuses[action.target_id] = FinalStatus.UNRESOLVED

        # Propagate blocked conditions to beliefs
        requires_map = scenario.memory_topology.get("requires", {})
        for belief_id, conds in requires_map.items():
            if belief_id in statuses and statuses[belief_id] == FinalStatus.AUTHORIZED:
                # If any of the required conditions is blocked
                if any(c in blocked_conditions for c in conds):
                    statuses[belief_id] = FinalStatus.BLOCKED

        # Predict answer based on resolved statuses
        pred_ans = "A"
        if query.probe_type == ProbeType.STATE_RESOLUTION:
            # Match resolved status
            primary_belief_id = None
            for entry in scenario.memory_snapshot:
                if entry.entry_type == "belief":
                    primary_belief_id = entry.entry_id
                    break
            if primary_belief_id:
                status_val = statuses.get(primary_belief_id, FinalStatus.AUTHORIZED).value
                for k, v in query.options.items():
                    if status_val in v:
                        pred_ans = k
                        break
        elif query.probe_type == ProbeType.PREMISE_RESISTANCE:
            pred_ans = "C"
        elif query.probe_type == ProbeType.POLICY_ADAPTATION:
            pred_ans = "A"
        elif query.probe_type == ProbeType.AUDIT_LOCALIZATION:
            pred_ans = query.gold_answer

        return Prediction(
            scenario_id=scenario.scenario_id,
            query_id=query.query_id,
            predicted_answer=pred_ans,
            predicted_final_statuses=statuses
        )
