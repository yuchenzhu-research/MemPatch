import os
import json
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark.retrace_bench.schemas_v2 import scenario_from_dict, PredictionV2, scenario_to_dict
from benchmark.retrace_bench.taxonomy_v2 import TaskProtocolV2, MemoryStatusV2


def _prediction_key(row: dict) -> tuple[str, str] | None:
    scenario_id = row.get("scenario_id")
    task_id = row.get("task_id")
    if isinstance(scenario_id, str) and isinstance(task_id, str):
        return (scenario_id, task_id)
    return None


def _read_jsonl_tolerant(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[resume] Ignoring invalid JSONL line {line_no} in {path}", file=sys.stderr)
    return rows


def _append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate only the first N scenarios")
    parser.add_argument("--append", action="store_true", help="Append predictions as they finish instead of rewriting --out at the end")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing --out JSONL by skipping completed scenario/task pairs")
    args = parser.parse_args()

    scenarios_path = os.path.join(args.data, "scenarios.jsonl")
    if not os.path.exists(scenarios_path):
        print(f"Error: scenarios.jsonl not found in {args.data}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    scenarios = []
    try:
        with open(scenarios_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s_dict = json.loads(line)
                scenarios.append(scenario_from_dict(s_dict))
                if args.max_cases is not None and len(scenarios) >= args.max_cases:
                    break
    except Exception as e:
        print(f"Failed to process scenarios: {e}", file=sys.stderr)
        sys.exit(1)

    predictions = []
    completed_keys = set()
    if args.resume and os.path.exists(args.out):
        expected_keys = {(scenario.scenario_id, task.task_id) for scenario in scenarios for task in scenario.tasks}
        existing_by_key = {}
        for row in _read_jsonl_tolerant(args.out):
            key = _prediction_key(row)
            if key in expected_keys:
                existing_by_key[key] = row
        predictions = [existing_by_key[key] for scenario in scenarios for task in scenario.tasks if (key := (scenario.scenario_id, task.task_id)) in existing_by_key]
        completed_keys = set(existing_by_key)
        _write_jsonl(args.out, predictions)
        print(f"[resume] Loaded {len(predictions)} existing predictions from {args.out}", flush=True)
    elif args.append and os.path.exists(args.out):
        os.unlink(args.out)

    for scenario in scenarios:
        if args.baseline == "latest_only_v2":
            preds = latest_only_baseline(scenario)
        elif args.baseline == "retrieve_all_v2":
            preds = retrieve_all_baseline(scenario)
        for pred in preds:
            pred_row = scenario_to_dict(pred)
            key = _prediction_key(pred_row)
            if key in completed_keys:
                continue
            predictions.append(pred_row)
            if args.append or args.resume:
                _append_jsonl(args.out, pred_row)

    if not args.append and not args.resume:
        try:
            _write_jsonl(args.out, predictions)
        except Exception as e:
            print(f"Failed to write predictions: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"✓ Wrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
