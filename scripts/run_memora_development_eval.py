#!/usr/bin/env python3
"""Memora Oracle-Conditioned Authorization Diagnostic runner.

MEMORA ORACLE-CONDITIONED AUTHORIZATION DIAGNOSTIC ONLY —
NOT OFFICIAL END-TO-END MEMORA RESULT.

Candidate beliefs originate from Memora evaluation annotations
(memory_evidence / forgetting_evidence), not from end-to-end memory
extraction.  Do not interpret output as a paper result.
"""
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
from retracemem.adapters.memora_oracle_diagnostic import build_oracle_diagnostic_view
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.evaluation.manifest import RunConfiguration, RunManifest, compute_file_sha256
from retracemem.generation.answer_generator import PromptAnswerGenerator
from retracemem.methods.batched_controlled_retrace import BatchedControlledReTraceLLM
from retracemem.methods.controlled_retrace import ControlledReTraceLLM
from retracemem.methods.directjudge import DirectJudgeLLM
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.budget import BudgetWrappedProvider, GlobalBudget
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.verifier.prompt_batched_evidence_edge_verifier import PromptBatchedEvidenceEdgeVerifier
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier

DISCLAIMER = (
    "MEMORA ORACLE-CONDITIONED AUTHORIZATION DIAGNOSTIC ONLY — "
    "NOT OFFICIAL END-TO-END MEMORA RESULT."
)
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


# ---------------------------------------------------------------------------
# Mock providers for replay mode
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _progress(
    tag: str, persona: str, qid: str, budget: GlobalBudget, started: float,
) -> None:
    elapsed = time.time() - started
    print(
        f"[{tag}] persona={persona} qid={qid} "
        f"calls={budget.calls}/{budget.max_calls} "
        f"tokens={budget.tokens}/{budget.max_tokens} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

def make_client(
    mode: str, cache_path: str, budget: GlobalBudget, stage: str,
    timeout: float,
) -> CachedLLMClient:
    if mode == "live-dev":
        inner = HTTPLLMProvider(timeout=timeout)
        wrapped = BudgetWrappedProvider(inner, budget, stage)
        return CachedLLMClient(JSONLCache(cache_path), wrapped, CostAccounting())
    if stage == "stage_b":
        provider = StageBMockProvider(default_response='{"verdicts": []}')
    else:
        provider = MockLLMProvider(default_response='{"edges": []}')
    return CachedLLMClient(JSONLCache(cache_path), provider, CostAccounting())


def make_answer_client(
    mode: str, cache_path: str, budget: GlobalBudget, stage: str,
    timeout: float,
) -> CachedLLMClient:
    if mode == "live-dev":
        inner = HTTPLLMProvider(timeout=timeout)
        wrapped = BudgetWrappedProvider(inner, budget, stage)
        return CachedLLMClient(JSONLCache(cache_path), wrapped, CostAccounting())
    provider = MockLLMProvider(default_response="mock answer")
    return CachedLLMClient(JSONLCache(cache_path), provider, CostAccounting())


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _write_report(
    out: Path, rows: list[dict[str, Any]], errors: list[str],
    args: Any, budget: GlobalBudget, verifier_hash: str, answer_hash: str,
    personas: list[str],
) -> tuple[Path, Path]:
    report = {
        "disclaimer": DISCLAIMER,
        "oracle_conditioned_candidates": True,
        "official_end_to_end_result": False,
        "candidate_source": "memora_evaluation_annotations",
        "scoring": "pending",
        "mode": args.mode,
        "stage_a_execution": args.stage_a_execution,
        "provider": args.provider,
        "model": args.model,
        "stage_a_prompt_version": args.stage_a_prompt_version,
        "http_timeout_seconds": args.http_timeout_seconds,
        "period": args.period,
        "personas": personas,
        "questions_executed": len(rows),
        "errors": errors,
        "global_budget": budget.summary(),
        "rows": rows,
    }
    report_path = out / "memora_development_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    data_files = [
        Path(args.reference_root) / "data" / args.period / p
        / f"evaluation_questions_{p}.json"
        for p in personas
    ]
    data_checksum = hashlib.sha256(
        "".join(
            compute_file_sha256(str(p)) for p in data_files if p.exists()
        ).encode("utf-8")
    ).hexdigest()
    memora_sha = subprocess.run(
        ["git", "-C", "reference/Memora", "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    manifest = RunManifest(
        config=RunConfiguration(
            run_id=f"memora-diag-{uuid.uuid4()}",
            stage_and_method_name="Memora-oracle-conditioned-StageAB",
            provider_name=args.provider,
            model_id=args.model,
            temperature=0.0,
            prompt_hashes={"stage_a": verifier_hash, "answer": answer_hash},
            cache_path=str(out / "caches"),
            dataset_checksum=data_checksum,
            metadata={
                "memora_upstream_sha": memora_sha,
                "period": args.period,
                "personas": personas,
                "questions_executed": len(rows),
                "oracle_conditioned_candidates": True,
                "official_end_to_end_result": False,
                "stage_a_execution": args.stage_a_execution,
            },
        ),
        aggregate_cost=budget.summary(),
        instance_count=len(rows),
        output_path=str(report_path),
        errors_or_retries=[{"error": e} for e in errors],
        metadata={"official_end_to_end_result": False},
    )
    manifest_path = out / "memora_development_manifest.json"
    manifest.save(str(manifest_path))
    return report_path, manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memora Oracle-Conditioned Authorization Diagnostic.",
    )
    parser.add_argument("--mode", choices=("replay", "live-dev"), default="replay")
    parser.add_argument("--live-approved", action="store_true")
    parser.add_argument("--reference-root", default="reference/Memora")
    parser.add_argument("--period", default="weekly")
    parser.add_argument("--persona", default="academic_researcher")
    parser.add_argument("--all-personas", action="store_true")
    parser.add_argument("--limit-questions", type=int, default=5)
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument(
        "--stage-a-execution", choices=("per-belief", "batched"),
        default="batched",
    )
    parser.add_argument(
        "--stage-a-prompt-version",
        default="evidence_edge_prediction_batch_v1",
    )
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--http-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output-dir", default="outputs/memora_oracle_diag",
    )
    args = parser.parse_args()

    if args.mode == "live-dev" and not args.live_approved:
        raise SystemExit("Refusing live execution without --live-approved")
    out = Path(args.output_dir)
    if args.mode == "live-dev" and (
        (out / "memora_development_report.json").exists()
        or (out / "memora_development_manifest.json").exists()
    ):
        raise SystemExit("Refusing to overwrite existing live output directory")
    out.mkdir(parents=True, exist_ok=True)

    adapter = MemoraAdapter(args.reference_root)
    personas = PERSONAS if args.all_personas else [args.persona]

    selected: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for persona in personas:
        sessions = adapter.load_sessions(args.period, persona)
        questions = adapter.load_evaluation_questions(
            args.period, persona,
        )[: max(args.limit_questions, 0)]
        if questions:
            selected.append((persona, sessions, questions))

    total_questions = sum(len(qs) for _, _, qs in selected)
    print(
        f"[PLAN] {DISCLAIMER}\n"
        f"  period={args.period} personas={len(selected)} "
        f"questions={total_questions}\n"
        f"  provider={args.provider} model={args.model}\n"
        f"  stage_a_execution={args.stage_a_execution} "
        f"timeout={args.http_timeout_seconds}s\n"
        f"  max_calls={args.max_calls} max_tokens={args.max_tokens}",
        flush=True,
    )

    budget = GlobalBudget(max_calls=args.max_calls, max_tokens=args.max_tokens)
    cache_dir = out / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = str(cache_dir / f"c_{uuid.uuid4()}")

    timeout = args.http_timeout_seconds
    client_a = make_client(
        args.mode, cache_prefix + "_stage_a.jsonl", budget, "stage_a", timeout,
    )
    client_b = make_client(
        args.mode, cache_prefix + "_stage_b.jsonl", budget, "stage_b", timeout,
    )
    client_ans_a = make_answer_client(
        args.mode, cache_prefix + "_answer_a.jsonl", budget, "answer_a", timeout,
    )
    client_ans_b = make_answer_client(
        args.mode, cache_prefix + "_answer_b.jsonl", budget, "answer_b", timeout,
    )

    if args.stage_a_execution == "batched":
        batched_verifier = PromptBatchedEvidenceEdgeVerifier(
            client=client_a, model_id=args.model, provider=args.provider,
            prompt_version=args.stage_a_prompt_version,
        )
        stage_a_runner = BatchedControlledReTraceLLM(
            edge_verifier=batched_verifier, client=client_a,
        )
        verifier_hash = batched_verifier._template_hash
    else:
        per_belief_verifier = PromptEvidenceEdgeVerifier(
            client=client_a, model_id=args.model, provider=args.provider,
            prompt_version=args.stage_a_prompt_version,
        )
        stage_a_runner = ControlledReTraceLLM(
            edge_verifier=per_belief_verifier, client=client_a,
        )
        verifier_hash = per_belief_verifier._template_hash

    stage_b_runner = DirectJudgeLLM(
        client=client_b, model_id=args.model, provider=args.provider,
    )
    gen_a = PromptAnswerGenerator(client_ans_a)
    gen_b = PromptAnswerGenerator(client_ans_b)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    done = 0
    started = time.time()

    for persona, sessions, questions in selected:
        for question in questions:
            qid = str(question.get("question_id"))
            view, view_meta = build_oracle_diagnostic_view(
                args.period, persona, question, sessions,
            )
            row: dict[str, Any] = {
                "period": args.period,
                "persona": persona,
                "question_id": qid,
                "question": view.query,
                "evaluation": question.get("evaluation", {}),
                "view_meta": view_meta,
                "stage_a": {},
                "stage_b": {},
            }

            _progress("STAGE_A_START", persona, qid, budget, started)
            try:
                res_a = stage_a_runner.run(view)
                _progress("STAGE_A_DONE", persona, qid, budget, started)
                basis_a = [
                    b for b in view.candidate_beliefs
                    if b.belief_id in res_a.authorized_belief_ids
                ]
                _progress("ANSWER_A_START", persona, qid, budget, started)
                ans_a = gen_a.generate_answer(
                    view.query, list(basis_a),
                    model_id=args.model, provider=args.provider,
                )
                _progress("ANSWER_A_DONE", persona, qid, budget, started)
                row["stage_a"] = {
                    "authorized_belief_ids": list(res_a.authorized_belief_ids),
                    "excluded_belief_ids": list(res_a.excluded_belief_ids),
                    "answer": ans_a,
                    "provenance": res_a.provenance,
                }
            except Exception as exc:
                msg = f"Stage A {persona}/{qid}: {type(exc).__name__}: {exc}"
                row["stage_a"] = {"error": msg}
                errors.append(msg)
                _progress("STAGE_A_ERROR", persona, qid, budget, started)

            _progress("STAGE_B_START", persona, qid, budget, started)
            try:
                res_b = stage_b_runner.judge(view)
                _progress("STAGE_B_DONE", persona, qid, budget, started)
                basis_b = [
                    b for b in view.candidate_beliefs
                    if b.belief_id in res_b.authorized_belief_ids
                ]
                _progress("ANSWER_B_START", persona, qid, budget, started)
                ans_b = gen_b.generate_answer(
                    view.query, list(basis_b),
                    model_id=args.model, provider=args.provider,
                )
                _progress("ANSWER_B_DONE", persona, qid, budget, started)
                row["stage_b"] = {
                    "authorized_belief_ids": list(res_b.authorized_belief_ids),
                    "excluded_belief_ids": list(res_b.excluded_belief_ids),
                    "answer": ans_b,
                    "provenance": res_b.provenance,
                }
            except Exception as exc:
                msg = f"Stage B {persona}/{qid}: {type(exc).__name__}: {exc}"
                row["stage_b"] = {"error": msg}
                errors.append(msg)
                _progress("STAGE_B_ERROR", persona, qid, budget, started)

            rows.append(row)
            done += 1
            print(
                f"[QUESTION_DONE] {done}/{total_questions} "
                f"persona={persona} qid={qid} "
                f"calls={budget.calls}/{budget.max_calls} "
                f"tokens={budget.tokens}/{budget.max_tokens} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    report_path, manifest_path = _write_report(
        out, rows, errors, args, budget, verifier_hash, gen_a.template_hash,
        personas,
    )
    print(
        f"[DONE] report={report_path} manifest={manifest_path} "
        f"errors={len(errors)} "
        f"calls={budget.calls} tokens={budget.tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()
