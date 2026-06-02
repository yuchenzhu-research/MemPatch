#!/usr/bin/env python3
"""Run general ReTrace-Bench baselines.

For backward compatibility this script remains named as before, but when the
input records use ``schema_version=retrace_bench_general_1`` it evaluates the
new neutral English benchmark schema requested for the general release.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from benchmark.retrace_bench.general_taxonomy import FAILURE_MODES, MEMORY_STATUSES, NON_ANSWER_DECISIONS
from benchmark.retrace_bench.llm_providers import get_provider
from benchmark.retrace_bench.scorers_general import aggregate_metrics, score_prediction
from retracemem.authorization import EvidenceProposalBatch, authorize
from retracemem.methods.contracts import SharedCandidateView
from retracemem.schemas import BeliefNode, EvidenceEdge, EvidenceEdgeType, EvidenceNode


def _infer_diagnosis_from_memory_state(memory_state: dict[str, str]) -> str:
    """Infer a failure diagnosis from the PREDICTED memory state only.

    Intentionally weak but non-cheating: it never reads hidden gold such as
    ``primary_failure_mode``. Non-oracle baselines use this instead of echoing
    the gold diagnosis, which previously produced perfect diagnosis accuracy.
    """
    statuses = set((memory_state or {}).values())
    if "should_not_store" in statuses:
        return "policy_violation"
    if "unresolved" in statuses:
        return "conflict_collapse"
    if "out_of_scope" in statuses:
        return "scope_leakage"
    if "outdated" in statuses:
        return "stale_memory_reuse"
    if "deleted" in statuses:
        return "failure_to_forget"
    if "restored" in statuses:
        return "failure_to_release_or_restore"
    return "under_update"


def _infer_decision_from_memory_state(memory_state: dict[str, str]) -> str:
    """Infer a non-answer decision from PREDICTED memory state only.

    Deliberately conservative (no gold). Distractor ``out_of_scope`` labels are
    common, so we do not escalate on those; only policy/conflict states change
    the decision away from ``use_current_memory``.
    """
    statuses = set((memory_state or {}).values())
    if "should_not_store" in statuses:
        return "refuse_due_to_policy"
    if "unresolved" in statuses:
        return "mark_unresolved"
    return "use_current_memory"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "scenarios.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


BAR_LENGTH = 32


def progress_wait(current: int, total: int, scenario_id: str, done: threading.Event) -> None:
    width = BAR_LENGTH
    start = time.monotonic()
    pos = 0
    direction = 1
    while not done.wait(0.12):
        cells = ["."] * width
        cells[pos] = "="
        cells[min(pos + 1, width - 1)] = ">"
        elapsed = time.monotonic() - start
        print(
            f"\r [{''.join(cells)}]  Time: {elapsed:5.1f}s | Total: -- | {scenario_id} {current}/{total}",
            end="",
            flush=True,
        )
        pos += direction
        if pos >= width - 2 or pos <= 0:
            direction *= -1


def progress_done(
    current: int,
    total: int,
    scenario_id: str,
    *,
    step_seconds: float,
    total_seconds: float,
    metrics: dict[str, float],
) -> None:
    cur_len = int(BAR_LENGTH * current / max(total, 1))
    rest_len = max(BAR_LENGTH - cur_len - 1, 0)
    bar = "=" * cur_len + ">" + "." * rest_len
    msg = (
        f"Mem: {metrics.get('memory_state_accuracy', 0.0):.3f} | "
        f"EvF1: {metrics.get('evidence_f1', 0.0):.3f} | "
        f"Diag: {metrics.get('failure_diagnosis_accuracy', 0.0):.3f}"
    )
    print(
        f"\r [{bar}]  Time: {step_seconds:5.1f}s | Total: {total_seconds:5.1f}s | "
        f"{msg} | {scenario_id} {current}/{total}",
        flush=True,
    )


def latest_only(scenario: dict[str, Any]) -> dict[str, Any]:
    events = scenario["public_input"]["event_trace"]
    latest = events[-1]
    memory_state = {m["memory_id"]: "current" for m in scenario["public_input"]["initial_memory"]}
    for mid in latest.get("related_memory_ids", []):
        memory_state[mid] = "current"
    return {
        "answer": latest["text"],
        "decision": "use_current_memory",
        "memory_state": memory_state,
        "evidence_event_ids": [latest["event_id"]],
        "failure_diagnosis": _infer_diagnosis_from_memory_state(memory_state),
    }


def retrieve_all(scenario: dict[str, Any]) -> dict[str, Any]:
    events = scenario["public_input"]["event_trace"]
    text = " ".join(e["text"] for e in events)
    state = {m["memory_id"]: "current" for m in scenario["public_input"]["initial_memory"]}
    evidence = [e["event_id"] for e in events if e["trust_level"] in {"verified", "trusted"}]
    return {
        "answer": text,
        "decision": "use_current_memory",
        "memory_state": state,
        "evidence_event_ids": evidence,
        "failure_diagnosis": _infer_diagnosis_from_memory_state(state),
    }


def _scenario_tokens(scenario: dict[str, Any]) -> set[str]:
    text = " ".join(
        [
            scenario.get("workflow_context", ""),
            " ".join(t.get("prompt", "") for t in scenario.get("tasks", [])),
        ]
    ).lower()
    return {tok.strip(".,;:!?()[]{}\"'") for tok in text.split() if len(tok.strip(".,;:!?()[]{}\"'")) >= 3}


def rag_lexical(scenario: dict[str, Any]) -> dict[str, Any]:
    """Simple lexical RAG baseline: retrieve top visible events/memories, then answer from them.

    This intentionally has no temporal graph or mutation semantics. It is a
    fair weak RAG baseline for testing whether retrieval alone handles stale,
    scoped, and introduced-memory cases.
    """
    query_tokens = _scenario_tokens(scenario)
    events = scenario["public_input"]["event_trace"]
    scored = []
    for event in events:
        toks = {tok.strip(".,;:!?()[]{}\"'").lower() for tok in event["text"].split()}
        score = len(query_tokens & toks) + (2 if event.get("trust_level") == "verified" else 0)
        scored.append((score, event))
    top_events = [event for _, event in sorted(scored, key=lambda x: (x[0], x[1]["timestamp"]), reverse=True)[:3]]
    best = top_events[0] if top_events else events[-1]
    state = {m["memory_id"]: "current" for m in scenario["public_input"]["initial_memory"]}
    for mid in best.get("related_memory_ids", []):
        state[mid] = "current"
    return {
        "answer": best["text"],
        "decision": "use_current_memory",
        "memory_state": state,
        "evidence_event_ids": [e["event_id"] for e in top_events],
        "failure_diagnosis": _infer_diagnosis_from_memory_state(state),
    }


def crud_memory(scenario: dict[str, Any]) -> dict[str, Any]:
    """CRUD-like memory baseline with last-write-wins over visible IDs.

    It updates/deletes/restores memory IDs from chronological events but lacks
    DPA-style dependency and scope reasoning.
    """
    state = {m["memory_id"]: "current" for m in scenario["public_input"]["initial_memory"]}
    answer_event = scenario["public_input"]["event_trace"][-1]
    evidence: list[str] = []
    for event in scenario["public_input"]["event_trace"]:
        text = event["text"].lower()
        related = event.get("related_memory_ids", [])
        if event.get("trust_level") == "untrusted":
            continue
        if event.get("trust_level") == "verified":
            evidence = [event["event_id"]]
            answer_event = event
        for mid in related:
            if mid not in state:
                state[mid] = "current"
        if "remove" in text or "do not store" in text:
            for mid in related:
                state[mid] = "deleted" if "remove" in text else "should_not_store"
        elif "restore" in text or "cleared" in text:
            for mid in related:
                state[mid] = "restored"
        elif "unresolved" in text or "incompatible" in text:
            for mid in related:
                state[mid] = "unresolved"
    return {
        "answer": answer_event["text"],
        "decision": _infer_decision_from_memory_state(state),
        "memory_state": state,
        "evidence_event_ids": evidence or [answer_event["event_id"]],
        "failure_diagnosis": _infer_diagnosis_from_memory_state(state),
    }


def mem0_style(scenario: dict[str, Any]) -> dict[str, Any]:
    """Mem0-style fact mutation baseline.

    Keeps a compact latest-fact store and handles add/update/delete keywords,
    but deliberately does not model prerequisite conditions or defeat paths.
    """
    state = {m["memory_id"]: "current" for m in scenario["public_input"]["initial_memory"]}
    answer_event = None
    evidence: list[str] = []
    for event in scenario["public_input"]["event_trace"]:
        text = event["text"].lower()
        if event.get("trust_level") == "untrusted":
            continue
        if event.get("trust_level") == "verified":
            answer_event = event
            evidence = [event["event_id"]]
        for mid in event.get("related_memory_ids", []):
            if mid not in state:
                state[mid] = "current"
            if "do not store" in text or "private" in text or "credential" in text:
                state[mid] = "should_not_store"
            elif "remove" in text or "obsolete" in text:
                state[mid] = "deleted"
            elif "updated" in text or "verified system record" in text:
                state[mid] = "current"
    answer_event = answer_event or scenario["public_input"]["event_trace"][-1]
    decision = _infer_decision_from_memory_state(state)
    return {
        "answer": answer_event["text"],
        "decision": decision,
        "memory_state": state,
        "evidence_event_ids": evidence or [answer_event["event_id"]],
        "failure_diagnosis": _infer_diagnosis_from_memory_state(state),
    }


def retrace_oracle_engine(scenario: dict[str, Any]) -> dict[str, Any]:
    """Route blueprint-derived typed revisions through ReTrace authorize(...).

    This is an oracle-proposer baseline, not a learned model: it tests whether
    the deterministic ReTrace-Engine executes the general benchmark's typed
    memory updates correctly once the right typed proposal is supplied.
    """
    events = scenario["public_input"]["event_trace"]
    evidence_nodes = tuple(
        EvidenceNode(
            evidence_id=e["event_id"],
            session_id=scenario["scenario_id"],
            timestamp=e["timestamp"],
            text=e["text"],
            source_dataset="retrace_bench_general",
            source_pointer=scenario["scenario_id"],
            metadata={"trust_level": e["trust_level"], "visibility_scope": e["visibility_scope"]},
        )
        for e in events
    )
    evidence_by_id = {e.evidence_id: e for e in evidence_nodes}
    gold = scenario["hidden_gold"]
    expected_state = gold.get("expected_memory_state", {})
    initial_memories = scenario["public_input"]["initial_memory"]
    target_memory = initial_memories[0]
    target_id = target_memory["memory_id"]
    introduced = gold.get("rubric", {}).get("introduced_memories", {})
    replacement_id = next(iter(introduced.keys()), None)
    evidence_id = gold.get("expected_evidence_event_ids", [events[-1]["event_id"]])[0]
    candidate_beliefs = (
        BeliefNode(
            belief_id=target_id,
            proposition=target_memory["text"],
            source_evidence_ids=tuple(target_memory.get("source_event_ids", ())),
        ),
    )
    replacement_beliefs = tuple(
        BeliefNode(
            belief_id=mid,
            proposition=mem["text"],
            source_evidence_ids=tuple(mem.get("source_event_ids", ())),
        )
        for mid, mem in introduced.items()
    )
    proposal_edges: list[EvidenceEdge] = []
    if replacement_id and expected_state.get(target_id) == "outdated":
        proposal_edges.append(
            EvidenceEdge(
                edge_id=f"edge-{scenario['scenario_id']}-super",
                edge_type=EvidenceEdgeType.SUPERSEDES,
                evidence_id=evidence_id,
                target_kind="belief",
                target_id=target_id,
                verifier="retrace_oracle_engine",
                replacement_belief_id=replacement_id,
            )
        )
    elif expected_state.get(target_id) == "unresolved":
        proposal_edges.append(
            EvidenceEdge(
                edge_id=f"edge-{scenario['scenario_id']}-uncertain",
                edge_type=EvidenceEdgeType.UNCERTAIN,
                evidence_id=evidence_id,
                target_kind="belief",
                target_id=target_id,
                verifier="retrace_oracle_engine",
            )
        )
    view = SharedCandidateView(
        instance_id=scenario["scenario_id"],
        query_id=f"q-{scenario['scenario_id']}",
        query=scenario["tasks"][0]["prompt"],
        evidence_context=evidence_nodes,
        new_evidence=evidence_by_id[evidence_id],
        candidate_beliefs=candidate_beliefs,
        candidate_replacement_beliefs=replacement_beliefs,
        metadata={"source": "retrace_bench_general"},
    )
    result = authorize(view, (EvidenceProposalBatch(edges=tuple(proposal_edges)),))
    fine = result.trace["fine_grained_statuses"]
    status_map = {"AUTHORIZED": "current", "SUPERSEDED": "outdated", "UNRESOLVED": "unresolved", "BLOCKED": "blocked"}
    memory_state = {m["memory_id"]: ("out_of_scope" if m.get("is_distractor") else "current") for m in initial_memories}
    for bid, status in fine.items():
        memory_state[bid] = status_map.get(status, "current")
    for mid in introduced:
        memory_state[mid] = expected_state.get(mid, "current")
    # Gold-replay reference: the oracle already reads hidden_gold for answer,
    # evidence, and diagnosis, so it must also replay the gold decision rather
    # than reconstruct it from a hard-coded failure_mode -> decision mapping.
    # Each failure mode admits several valid decisions (see DECISIONS_BY_MODE in
    # scripts/generate_retrace_templateheldout_test.py), so the old mapping was
    # systematically wrong on the template-heldout split.
    decision = gold.get("expected_decision")
    if decision is None:
        print(
            "[retrace_oracle_engine] WARNING: scenario "
            f"{scenario['scenario_id']} has no hidden_gold.expected_decision; "
            "falling back to use_current_memory (non-canonical split).",
            file=sys.stderr,
        )
        decision = "use_current_memory"
    return {
        "answer": gold.get("expected_answer"),
        "decision": decision,
        "memory_state": memory_state,
        "evidence_event_ids": gold.get("expected_evidence_event_ids", []),
        "failure_diagnosis": scenario.get("primary_failure_mode"),
        "retrace_trace": result.trace,
    }


def heuristic_memory_state(scenario: dict[str, Any]) -> dict[str, Any]:
    events = scenario["public_input"]["event_trace"]
    verified = [e for e in events if e["trust_level"] == "verified"]
    chosen = verified[-1] if verified else events[-1]
    state = {m["memory_id"]: ("out_of_scope" if m.get("is_distractor") else "current") for m in scenario["public_input"]["initial_memory"]}
    for mid in chosen.get("related_memory_ids", []):
        state[mid] = "current"
    decision = _infer_decision_from_memory_state(state)
    return {
        "answer": chosen["text"] if decision == "use_current_memory" else f"{decision}: {chosen['text']}",
        "decision": decision,
        "memory_state": state,
        "evidence_event_ids": [chosen["event_id"]],
        "failure_diagnosis": _infer_diagnosis_from_memory_state(state),
    }


def _parse_llm_json_response(raw: str) -> dict[str, Any]:
    """Parse an LLM JSON object, tolerating common fenced-code wrapping."""
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced_body = "\n".join(lines).strip()
        try:
            parsed = json.loads(fenced_body)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for start in (idx for idx, char in enumerate(text) if char == "{"):
        try:
            parsed, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise json.JSONDecodeError("No JSON object found in LLM response", raw, 0)


def llm_json_answerer(
    scenario: dict[str, Any],
    provider: Any,
    *,
    generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if provider is None:
        raise ValueError("llm_json_answerer requires --provider and --model")
    memory_ids = [m["memory_id"] for m in scenario["public_input"]["initial_memory"]]
    for event in scenario["public_input"]["event_trace"]:
        for memory_id in event.get("related_memory_ids", []):
            if memory_id not in memory_ids:
                memory_ids.append(memory_id)
    prompt = {
        "instruction": (
            "Answer as strict JSON only. Do not use Markdown. "
            "Use exact enum strings for decision, memory_state values, and failure_diagnosis. "
            "Do not invent memory IDs or event IDs."
        ),
        "schema": {
            "answer": "short final answer/action text",
            "decision": ["use_current_memory", *NON_ANSWER_DECISIONS],
            "memory_state": {mid: list(MEMORY_STATUSES) for mid in memory_ids},
            "evidence_event_ids": "minimal list of event_id strings from public_input.event_trace",
            "failure_diagnosis": list(FAILURE_MODES),
        },
        "workflow_context": scenario["workflow_context"],
        "public_input": scenario["public_input"],
        "tasks": scenario["tasks"],
    }
    kwargs = {"temperature": 0}
    kwargs.update(generation_kwargs or {})
    raw = provider.generate(json.dumps(prompt, ensure_ascii=False), **kwargs)
    try:
        parsed = _parse_llm_json_response(raw)
    except json.JSONDecodeError:
        parsed = {"answer": raw, "decision": None, "memory_state": {}, "evidence_event_ids": [], "failure_diagnosis": None}
    return parsed


BASELINES = {
    "latest_only": latest_only,
    "retrieve_all": retrieve_all,
    "rag_lexical": rag_lexical,
    "crud_memory": crud_memory,
    "mem0_style": mem0_style,
    "retrace_oracle_engine": retrace_oracle_engine,
    "heuristic_memory_state": heuristic_memory_state,
}

# Experiment grouping. Oracle baselines may read hidden gold and are upper
# bounds / mechanism sanity checks, NOT deployable comparable baselines.
ORACLE_BASELINES = {"retrace_oracle_engine"}

BASELINE_GROUPS = {
    "latest_only": "sanity",
    "retrieve_all": "sanity",
    "rag_lexical": "memory_baseline",
    "crud_memory": "memory_baseline",
    "mem0_style": "memory_baseline",
    "heuristic_memory_state": "memory_baseline",
    "retrace_oracle_engine": "oracle",
    "llm_json_answerer": "api_baseline",
}


def baseline_group(name: str) -> str:
    return BASELINE_GROUPS.get(name, "structured_method")


def is_oracle_baseline(name: str) -> bool:
    return name in ORACLE_BASELINES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ReTrace-Bench baseline evaluation.")
    parser.add_argument("--data", required=True, help="Path to scenarios.jsonl or a directory containing it")
    parser.add_argument("--baseline", required=True, choices=sorted([*BASELINES, "llm_json_answerer"]))
    parser.add_argument("--out", required=True, help="Path to output predictions JSONL")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate only the first N scenarios")
    parser.add_argument("--progress", action="store_true", help="Print per-scenario progress")
    parser.add_argument("--append", action="store_true", help="Append predictions as they finish instead of rewriting --out at the end")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum completion tokens for LLM baselines")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="For SiliconFlow/DeepSeek reasoning models, request non-thinking mode when supported",
    )
    args = parser.parse_args(argv)

    scenarios = read_jsonl(resolve_data_path(args.data))
    if args.max_cases is not None:
        scenarios = scenarios[: args.max_cases]
    provider = None
    if args.baseline == "llm_json_answerer":
        api_key = args.api_key
        if not api_key and args.provider:
            env_name = {
                "deepseek": "DEEPSEEK_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "siliconflow": "SILICONFLOW_API_KEY",
            }.get(args.provider.lower())
            api_key = os.getenv(env_name or "")
        provider = get_provider(args.provider or "openai", args.model or "gpt-4o-mini", api_key=api_key)
    generation_kwargs: dict[str, Any] = {}
    if args.baseline == "llm_json_answerer":
        generation_kwargs["max_tokens"] = args.max_tokens
        if args.disable_thinking and args.provider and args.provider.lower() == "siliconflow":
            generation_kwargs["extra_body"] = {"enable_thinking": False}

    predictions: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    out = Path(args.out)
    if args.append and out.exists():
        out.unlink()
    run_start = time.monotonic()
    for idx, scenario in enumerate(scenarios, start=1):
        spinner_done: threading.Event | None = None
        spinner_thread: threading.Thread | None = None
        step_start = time.monotonic()
        if args.progress:
            if args.baseline == "llm_json_answerer":
                spinner_done = threading.Event()
                spinner_thread = threading.Thread(
                    target=progress_wait,
                    args=(idx, len(scenarios), scenario["scenario_id"], spinner_done),
                    daemon=True,
                )
                spinner_thread.start()
            else:
                print(f"[{idx}/{len(scenarios)}] {scenario['scenario_id']}", flush=True)
        if args.baseline == "llm_json_answerer":
            try:
                response = llm_json_answerer(scenario, provider, generation_kwargs=generation_kwargs)
            finally:
                if spinner_done is not None:
                    spinner_done.set()
                if spinner_thread is not None:
                    spinner_thread.join()
        else:
            response = BASELINES[args.baseline](scenario)
        pred = {
            "scenario_id": scenario["scenario_id"],
            "baseline": args.baseline,
            "group": baseline_group(args.baseline),
            "is_oracle": is_oracle_baseline(args.baseline),
            "response": response,
            "domain": scenario.get("domain"),
            "primary_failure_mode": scenario.get("primary_failure_mode"),
            "expected_decision": scenario.get("hidden_gold", {}).get("expected_decision"),
            "decision_aliases": scenario.get("hidden_gold", {}).get("decision_aliases") or scenario.get("hidden_gold", {}).get("rubric", {}).get("decision_aliases") or scenario.get("decision_aliases"),
        }
        pred["metrics"] = score_prediction(scenario, pred)
        predictions.append(pred)
        scored.append(pred)
        if args.progress:
            progress_done(
                idx,
                len(scenarios),
                scenario["scenario_id"],
                step_seconds=time.monotonic() - step_start,
                total_seconds=time.monotonic() - run_start,
                metrics=pred["metrics"],
            )
        if args.append:
            append_jsonl(out, pred)

    if not args.append:
        write_jsonl(out, predictions)
    metrics_path = out.with_suffix(".metrics.json")
    aggregate = aggregate_metrics(scored)
    aggregate["baseline"] = args.baseline
    aggregate["group"] = baseline_group(args.baseline)
    aggregate["is_oracle"] = is_oracle_baseline(args.baseline)
    metrics_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {out}")
    print(f"Wrote metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
