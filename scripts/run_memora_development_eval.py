#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

from retracemem.adapters.memora_adapter import MemoraAdapter
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.evaluation.manifest import RunConfiguration, RunManifest, compute_file_sha256
from retracemem.generation.answer_generator import PromptAnswerGenerator
from retracemem.methods.contracts import SharedCandidateView
from retracemem.methods.controlled_retrace import ControlledReTraceLLM
from retracemem.methods.directjudge import DirectJudgeLLM
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.schemas import BeliefNode, EvidenceNode
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier

DISCLAIMER = "MEMORA DEVELOPMENT EVALUATION ONLY — NOT FINAL PAPER RESULT."
PERSONAS = [
    "academic_researcher",
    "business_executive",
    "content_writer",
    "creative_designer",
    "financial_analyst",
    "management_consultant",
    "marketing_manager",
    "sales_manager",
    "software_engineer",
    "startup_founder",
]


class CountingProvider:
    def __init__(self, inner: Any, max_calls: int, max_tokens: int) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls = 0
        self.tokens = 0

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"Hard live call cap reached: {self.max_calls}")
        if self.tokens >= self.max_tokens:
            raise RuntimeError(f"Hard live token cap reached: {self.max_tokens}")
        trace = self.inner.generate(*args, **kwargs)
        self.calls += 1
        self.tokens += trace.total_tokens
        return trace


class StageBMockProvider(MockLLMProvider):
    def generate(self, prompt: str, **kwargs: Any) -> Any:
        belief_ids = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("- b:") and ":" in stripped:
                belief_ids.append(stripped.removeprefix("- ").split(": \"", 1)[0].strip())
        verdicts = [{"belief_id": bid, "status": "USABLE", "rationale": "mock replay"} for bid in belief_ids]
        self.default_response = json.dumps({"verdicts": verdicts})
        return super().generate(prompt, **kwargs)


def flatten_values(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "value" in value:
            items.append(value)
        else:
            for child in value.values():
                items.extend(flatten_values(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(flatten_values(child))
    return items


def evidence_text_for_sessions(sessions: list[dict[str, Any]], session_ids: set[int]) -> str:
    parts = []
    for session in sessions:
        sid = session.get("session_id")
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = -1
        if sid_int not in session_ids:
            continue
        turns = []
        for turn in session.get("conversation", []):
            if isinstance(turn, dict):
                speaker = turn.get("speaker") or turn.get("role") or "unknown"
                message = turn.get("message") or turn.get("content") or ""
                if message:
                    turns.append(f"{speaker}: {message}")
        if turns:
            parts.append(f"session {sid}: " + "\n".join(turns))
    return "\n\n".join(parts)


def build_view(period: str, persona: str, question: dict[str, Any], sessions: list[dict[str, Any]]) -> SharedCandidateView:
    qid = str(question.get("question_id"))
    memory_items = flatten_values(question.get("memory_evidence", {}))
    forgetting_items = flatten_values(question.get("forgetting_evidence", {}))
    session_ids = {int(item.get("session_id")) for item in memory_items + forgetting_items if str(item.get("session_id", "")).isdigit()}
    context_text = evidence_text_for_sessions(sessions, session_ids)
    if not context_text:
        context_text = json.dumps({"memory_evidence": question.get("memory_evidence"), "forgetting_evidence": question.get("forgetting_evidence")}, ensure_ascii=False)
    evidence = EvidenceNode(
        evidence_id=f"memora:{period}:{persona}:{qid}:evidence",
        session_id=f"{period}:{persona}",
        timestamp=question.get("question_date"),
        text=context_text,
        source_dataset="memora",
        source_pointer=f"{period}/{persona}/{qid}",
    )
    beliefs = []
    for idx, item in enumerate(memory_items):
        val = str(item.get("value") or "").strip()
        if val:
            beliefs.append(BeliefNode(belief_id=f"b:{period}:{persona}:{qid}:memory:{idx}", proposition=val, source_evidence_ids=(evidence.evidence_id,), confidence=1.0, metadata={"memora_role": "memory_presence", "session_id": item.get("session_id"), "question_date": question.get("question_date")}))
    for idx, item in enumerate(forgetting_items):
        val = str(item.get("value") or "").strip()
        if val:
            beliefs.append(BeliefNode(belief_id=f"b:{period}:{persona}:{qid}:forget:{idx}", proposition=val, source_evidence_ids=(evidence.evidence_id,), confidence=1.0, metadata={"memora_role": "forgetting_absence", "session_id": item.get("session_id"), "question_date": question.get("question_date")}))
    if not beliefs:
        beliefs.append(BeliefNode(belief_id=f"b:{period}:{persona}:{qid}:fallback", proposition=str(question.get("question") or ""), source_evidence_ids=(evidence.evidence_id,), confidence=0.5, metadata={"memora_role": "fallback", "question_date": question.get("question_date")}))
    return SharedCandidateView(
        instance_id=f"memora:{period}:{persona}:{qid}",
        query_id=f"memora:{period}:{persona}:{qid}",
        query=str(question.get("question") or ""),
        evidence_context=(evidence,),
        new_evidence=evidence,
        candidate_beliefs=tuple(beliefs),
        candidate_replacement_beliefs=(),
        candidate_conditions_by_belief=tuple((belief.belief_id, ()) for belief in beliefs),
        dependency_edges_by_belief=tuple((belief.belief_id, ()) for belief in beliefs),
    )


def make_client(mode: str, cache_path: str, max_calls: int, max_tokens: int, *, stage: str) -> tuple[CachedLLMClient, CountingProvider | None]:
    if mode == "live-dev":
        counter = CountingProvider(HTTPLLMProvider(), max_calls=max_calls, max_tokens=max_tokens)
        return CachedLLMClient(JSONLCache(cache_path), counter, CostAccounting()), counter
    provider = StageBMockProvider(default_response='{"verdicts": []}') if stage == "B" else MockLLMProvider(default_response='{"edges": []}')
    return CachedLLMClient(JSONLCache(cache_path), provider, CostAccounting()), None


def make_answer_client(mode: str, cache_path: str, max_calls: int, max_tokens: int) -> tuple[CachedLLMClient, CountingProvider | None]:
    if mode == "live-dev":
        counter = CountingProvider(HTTPLLMProvider(), max_calls=max_calls, max_tokens=max_tokens)
        return CachedLLMClient(JSONLCache(cache_path), counter, CostAccounting()), counter
    provider = MockLLMProvider(default_response="mock answer")
    return CachedLLMClient(JSONLCache(cache_path), provider, CostAccounting()), None


def summarize_cost(client: CachedLLMClient, counter: CountingProvider | None) -> dict[str, Any]:
    cost = client.cost_accountant.to_dict()
    return {"cost": cost, "outbound_calls": counter.calls if counter else 0, "outbound_tokens": counter.tokens if counter else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Memora development Stage A/B runner.")
    parser.add_argument("--mode", choices=("replay", "live-dev"), default="replay")
    parser.add_argument("--live-approved", action="store_true")
    parser.add_argument("--reference-root", default="reference/Memora")
    parser.add_argument("--period", default="weekly")
    parser.add_argument("--persona", default="academic_researcher")
    parser.add_argument("--all-personas", action="store_true")
    parser.add_argument("--limit-questions", type=int, default=5)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--stage-a-prompt-version", default="evidence_edge_prediction_v1")
    parser.add_argument("--max-calls", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=4000000)
    parser.add_argument("--output-dir", default="outputs/memora_development_eval")
    args = parser.parse_args()
    if args.mode == "live-dev" and not args.live_approved:
        raise SystemExit("Refusing live execution without --live-approved")
    out = Path(args.output_dir)
    if args.mode == "live-dev" and ((out / "memora_development_report.json").exists() or (out / "memora_development_manifest.json").exists()):
        raise SystemExit("Refusing to overwrite existing live output directory")
    out.mkdir(parents=True, exist_ok=True)
    adapter = MemoraAdapter(args.reference_root)
    personas = PERSONAS if args.all_personas else [args.persona]
    selected: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for persona in personas:
        sessions = adapter.load_sessions(args.period, persona)
        questions = adapter.load_evaluation_questions(args.period, persona)[: max(args.limit_questions, 0)]
        if questions:
            selected.append((persona, sessions, questions))
    total_questions = sum(len(qs) for _persona, _sessions, qs in selected)
    print(f"[PLAN] {DISCLAIMER} period={args.period} personas={len(selected)} questions={total_questions} provider={args.provider} model={args.model}", flush=True)
    cache_prefix = out / f"cache_{uuid.uuid4()}"
    client_a, counter_a = make_client(args.mode, str(cache_prefix) + "_stage_a.jsonl", args.max_calls, args.max_tokens, stage="A")
    client_b, counter_b = make_client(args.mode, str(cache_prefix) + "_stage_b.jsonl", args.max_calls, args.max_tokens, stage="B")
    answer_a, answer_counter_a = make_answer_client(args.mode, str(cache_prefix) + "_answer_a.jsonl", args.max_calls, args.max_tokens)
    answer_b, answer_counter_b = make_answer_client(args.mode, str(cache_prefix) + "_answer_b.jsonl", args.max_calls, args.max_tokens)
    verifier = PromptEvidenceEdgeVerifier(client=client_a, model_id=args.model, provider=args.provider, prompt_version=args.stage_a_prompt_version)
    stage_a_runner = ControlledReTraceLLM(edge_verifier=verifier, client=client_a)
    stage_b_runner = DirectJudgeLLM(client=client_b, model_id=args.model, provider=args.provider)
    gen_a = PromptAnswerGenerator(answer_a)
    gen_b = PromptAnswerGenerator(answer_b)
    rows = []
    errors = []
    done = 0
    started = time.time()
    for persona, sessions, questions in selected:
        for question in questions:
            qid = str(question.get("question_id"))
            view = build_view(args.period, persona, question, sessions)
            row: dict[str, Any] = {"period": args.period, "persona": persona, "question_id": qid, "question": view.query, "evaluation": question.get("evaluation", {}), "stage_a": {}, "stage_b": {}}
            try:
                res_a = stage_a_runner.run(view)
                basis_a = [belief for belief in view.candidate_beliefs if belief.belief_id in res_a.authorized_belief_ids]
                ans_a = gen_a.generate_answer(view.query, list(basis_a), model_id=args.model, provider=args.provider)
                row["stage_a"] = {"authorized_belief_ids": list(res_a.authorized_belief_ids), "excluded_belief_ids": list(res_a.excluded_belief_ids), "answer": ans_a, "provenance": res_a.provenance}
            except Exception as exc:
                msg = f"Stage A {persona}/{qid}: {type(exc).__name__}: {exc}"
                row["stage_a"] = {"error": msg}
                errors.append(msg)
            try:
                res_b = stage_b_runner.judge(view)
                basis_b = [belief for belief in view.candidate_beliefs if belief.belief_id in res_b.authorized_belief_ids]
                ans_b = gen_b.generate_answer(view.query, list(basis_b), model_id=args.model, provider=args.provider)
                row["stage_b"] = {"authorized_belief_ids": list(res_b.authorized_belief_ids), "excluded_belief_ids": list(res_b.excluded_belief_ids), "answer": ans_b, "provenance": res_b.provenance}
            except Exception as exc:
                msg = f"Stage B {persona}/{qid}: {type(exc).__name__}: {exc}"
                row["stage_b"] = {"error": msg}
                errors.append(msg)
            rows.append(row)
            done += 1
            outbound = sum(c.calls for c in (counter_a, counter_b, answer_counter_a, answer_counter_b) if c)
            tokens = sum(c.tokens for c in (counter_a, counter_b, answer_counter_a, answer_counter_b) if c)
            print(f"[PROGRESS] persona={persona} period={args.period} questions {done}/{total_questions} outbound {outbound}/{args.max_calls} tokens {tokens}/{args.max_tokens} elapsed {time.time() - started:.1f}s", flush=True)
    report = {"disclaimer": DISCLAIMER, "official_final_result": False, "mode": args.mode, "provider": args.provider, "model": args.model, "stage_a_prompt_version": args.stage_a_prompt_version, "period": args.period, "personas": personas, "questions_executed": len(rows), "errors": errors, "cost": {"stage_a": summarize_cost(client_a, counter_a), "stage_b": summarize_cost(client_b, counter_b), "answer_a": summarize_cost(answer_a, answer_counter_a), "answer_b": summarize_cost(answer_b, answer_counter_b)}, "rows": rows}
    report_path = out / "memora_development_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    data_files = [Path(args.reference_root) / "data" / args.period / persona / f"evaluation_questions_{persona}.json" for persona in personas]
    data_checksum = hashlib.sha256("".join(compute_file_sha256(str(p)) for p in data_files if p.exists()).encode("utf-8")).hexdigest()
    memora_sha = subprocess.run(["git", "-C", "reference/Memora", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    manifest = RunManifest(config=RunConfiguration(run_id=f"memora-dev-{uuid.uuid4()}", stage_and_method_name="Memora-development-StageAB-controlled", provider_name=args.provider, model_id=args.model, temperature=0.0, prompt_hashes={"stage_a": verifier._template_hash, "answer": gen_a.template_hash}, cache_path=str(cache_prefix), dataset_checksum=data_checksum, metadata={"memora_upstream_sha": memora_sha, "period": args.period, "personas": personas, "questions_executed": len(rows), "development_only": True}), aggregate_cost=report["cost"], instance_count=len(rows), output_path=str(report_path), errors_or_retries=[{"error": e} for e in errors], metadata={"official_final_result": False})
    manifest_path = out / "memora_development_manifest.json"
    manifest.save(str(manifest_path))
    print(f"[DONE] report={report_path} manifest={manifest_path} errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
