from benchmark.retrace_bench.schemas import Scenario, ProbeQuery, Prediction
from benchmark.retrace_bench.taxonomy import ProbeType, FinalStatus


class RetrieveAllBaseline:
    def __init__(self):
        self.name = "retrieve_all"

    def predict(self, scenario: Scenario, query: ProbeQuery) -> Prediction:
        # Retrieve all assumes everything in the snapshot remains AUTHORIZED/active
        pred_ans = "A"

        if query.probe_type == ProbeType.STATE_RESOLUTION:
            # Always predict AUTHORIZED
            for k, v in query.options.items():
                if "AUTHORIZED" in v:
                    pred_ans = k
                    break
        elif query.probe_type == ProbeType.PREMISE_RESISTANCE:
            pred_ans = "C"
        elif query.probe_type == ProbeType.POLICY_ADAPTATION:
            pred_ans = "C"  # Incorrectly assumes updates are applied regardless of policy
        elif query.probe_type == ProbeType.AUDIT_LOCALIZATION:
            pred_ans = "B"  # Guess decoy

        pred_statuses = {}
        for entry in scenario.memory_snapshot:
            if entry.entry_type == "belief":
                pred_statuses[entry.entry_id] = FinalStatus.AUTHORIZED

        return Prediction(
            scenario_id=scenario.scenario_id,
            query_id=query.query_id,
            predicted_answer=pred_ans,
            predicted_final_statuses=pred_statuses
        )
