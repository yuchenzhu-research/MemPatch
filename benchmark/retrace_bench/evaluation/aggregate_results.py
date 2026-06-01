import json
from pathlib import Path
from typing import List, Dict, Any
from benchmark.retrace_bench.schemas import Scenario, Prediction
from benchmark.retrace_bench.utils.jsonl import read_jsonl
from benchmark.retrace_bench.evaluation.scorers import calculate_metrics


def aggregate_and_save(
    scenarios_path: str | Path,
    predictions_path: str | Path,
    out_report_path: str | Path
) -> Dict[str, Any]:
    scenarios_data = read_jsonl(scenarios_path)
    predictions_data = read_jsonl(predictions_path)

    # Reconstruct Scenario dataclasses from dict
    # (Since Scenario is read as raw dict, we convert it back or simply read properties. 
    # To keep it standard, we will parse dict keys).
    scenarios: List[Scenario] = []
    for d in scenarios_data:
        # DialogueTurns
        dialogue = []
        for t in d.get("dialogue_history", []):
            from benchmark.retrace_bench.schemas import DialogueTurn
            dialogue.append(DialogueTurn(speaker=t["speaker"], text=t["text"], timestamp=t.get("timestamp"), metadata=t.get("metadata", {})))
        
        # MemoryEntries
        snapshot = []
        for e in d.get("memory_snapshot", []):
            from benchmark.retrace_bench.schemas import MemoryEntry
            snapshot.append(MemoryEntry(entry_id=e["entry_id"], content=e["content"], entry_type=e["entry_type"], metadata=e.get("metadata", {})))
        
        # RevisionActions
        actions = []
        for a in d.get("gold_revision_actions", []):
            from benchmark.retrace_bench.schemas import RevisionAction
            from benchmark.retrace_bench.taxonomy import RevisionActionType
            actions.append(RevisionAction(
                action_type=RevisionActionType(a["action_type"]),
                target_id=a["target_id"],
                replacement_id=a.get("replacement_id"),
                evidence_ids=a.get("evidence_ids", []),
                rationale=a.get("rationale")
            ))

        # ProbeQueries
        queries = []
        for q in d.get("probe_queries", []):
            from benchmark.retrace_bench.schemas import ProbeQuery
            from benchmark.retrace_bench.taxonomy import ProbeType
            queries.append(ProbeQuery(
                query_id=q["query_id"],
                probe_type=ProbeType(q["probe_type"]),
                question=q["question"],
                options=q["options"],
                gold_answer=q["gold_answer"]
            ))

        from benchmark.retrace_bench.taxonomy import Domain, RevisionFamily, FinalStatus
        scenarios.append(Scenario(
            scenario_id=d["scenario_id"],
            domain=Domain(d["domain"]),
            revision_family=RevisionFamily(d["revision_family"]),
            conflict_type=d["conflict_type"],
            memory_topology=d["memory_topology"],
            dialogue_history=dialogue,
            memory_snapshot=snapshot,
            gold_final_statuses={k: FinalStatus(v) for k, v in d["gold_final_statuses"].items()},
            gold_revision_actions=actions,
            probe_queries=queries,
            metadata=d.get("metadata", {})
        ))

    # Reconstruct Prediction dataclasses
    predictions: List[Prediction] = []
    for p in predictions_data:
        from benchmark.retrace_bench.taxonomy import FinalStatus
        pred_statuses = None
        if p.get("predicted_final_statuses"):
            pred_statuses = {k: FinalStatus(v) for k, v in p["predicted_final_statuses"].items() if v}
        
        predictions.append(Prediction(
            scenario_id=p["scenario_id"],
            query_id=p["query_id"],
            predicted_answer=p["predicted_answer"],
            predicted_final_statuses=pred_statuses,
            metadata=p.get("metadata", {})
        ))

    metrics = calculate_metrics(scenarios, predictions)

    # Save to file
    out_path = Path(out_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
