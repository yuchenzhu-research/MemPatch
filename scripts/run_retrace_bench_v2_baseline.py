import os
import json
import argparse
import sys
from benchmark.retrace_bench.schemas_v2 import scenario_from_dict, PredictionV2, scenario_to_dict
from benchmark.retrace_bench.taxonomy_v2 import TaskProtocolV2, MemoryStatusV2


def latest_only_baseline(scenario) -> list:
    predictions = []
    # Find the latest event in the event trace
    latest_event_content = ""
    if scenario.event_trace:
        latest_event = scenario.event_trace[-1]
        latest_event_content = latest_event.content.get("text", "")

    for task in scenario.tasks:
        response = {}
        if task.protocol == TaskProtocolV2.BLACK_BOX_TASK:
            # Heuristically check if it's a multiple choice task (looking for "A", "B", etc. or options in prompt)
            # Default to "A" if options are detected or return a short excerpt of the latest event
            if "A)" in task.prompt or "A." in task.prompt or "options" in task.prompt.lower():
                response["answer"] = "A"
            else:
                response["answer"] = latest_event_content or "Default Answer"
        
        elif task.protocol == TaskProtocolV2.MEMORY_STATE_TASK:
            # Predict the status of all visible memories. Default to AUTHORIZED
            statuses = {}
            for m in scenario.memory_snapshot:
                statuses[m.memory_id] = MemoryStatusV2.AUTHORIZED.value
            # Also cover any gold memory statuses mentioned in task if any
            for m_id in task.gold_behavior.gold_memory_statuses.keys():
                statuses[m_id] = MemoryStatusV2.AUTHORIZED.value
            response["memory_statuses"] = statuses
            
        elif task.protocol == TaskProtocolV2.STRUCTURED_REVISION_TASK:
            # Return NO_REVISION or empty list
            response["actions"] = []
            
        elif task.protocol == TaskProtocolV2.ORACLE_DIAGNOSTIC_TASK:
            response["inspect"] = {"status": "ok"}
            
        pred = PredictionV2(
            scenario_id=scenario.scenario_id,
            task_id=task.task_id,
            response=response,
            metadata={"baseline": "latest_only_v2"}
        )
        predictions.append(pred)
    return predictions


def retrieve_all_baseline(scenario) -> list:
    predictions = []
    # Retrieve all events text concatenated
    all_context = " ".join([e.content.get("text", "") for e in scenario.event_trace if e.content.get("text")])

    for task in scenario.tasks:
        response = {}
        if task.protocol == TaskProtocolV2.BLACK_BOX_TASK:
            if "A)" in task.prompt or "options" in task.prompt.lower():
                response["answer"] = "A"
            else:
                response["answer"] = f"Concatenated trace context (len={len(all_context)})"
        
        elif task.protocol == TaskProtocolV2.MEMORY_STATE_TASK:
            statuses = {m.memory_id: MemoryStatusV2.AUTHORIZED.value for m in scenario.memory_snapshot}
            response["memory_statuses"] = statuses
            
        elif task.protocol == TaskProtocolV2.STRUCTURED_REVISION_TASK:
            response["actions"] = []
            
        elif task.protocol == TaskProtocolV2.ORACLE_DIAGNOSTIC_TASK:
            response["inspect"] = {"status": "ok", "context_length": len(all_context)}
            
        pred = PredictionV2(
            scenario_id=scenario.scenario_id,
            task_id=task.task_id,
            response=response,
            metadata={"baseline": "retrieve_all_v2"}
        )
        predictions.append(pred)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run ReTrace-Bench v2 Toy Baseline")
    parser.add_argument("--data", required=True, help="Path to dataset directory")
    parser.add_argument("--baseline", required=True, choices=["latest_only_v2", "retrieve_all_v2"], help="Baseline strategy")
    parser.add_argument("--out", required=True, help="Output path for JSONL predictions")
    args = parser.parse_args()

    scenarios_path = os.path.join(args.data, "scenarios.jsonl")
    if not os.path.exists(scenarios_path):
        print(f"Error: scenarios.jsonl not found in {args.data}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    predictions = []
    try:
        with open(scenarios_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s_dict = json.loads(line)
                scenario = scenario_from_dict(s_dict)
                
                if args.baseline == "latest_only_v2":
                    preds = latest_only_baseline(scenario)
                elif args.baseline == "retrieve_all_v2":
                    preds = retrieve_all_baseline(scenario)
                
                predictions.extend(preds)
    except Exception as e:
        print(f"Failed to process scenarios: {e}", file=sys.stderr)
        sys.exit(1)

    # Save to file
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(scenario_to_dict(pred)) + "\n")
        print(f"✓ Wrote {len(predictions)} predictions to {args.out}")
    except Exception as e:
        print(f"Failed to write predictions: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
