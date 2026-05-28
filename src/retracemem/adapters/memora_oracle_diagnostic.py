from __future__ import annotations

import json
import hashlib
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retracemem.adapters.memora_adapter import MemoraAdapter
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.evaluation.manifest import RunConfiguration, RunManifest, compute_file_sha256
from retracemem.generation.answer_generator import PromptAnswerGenerator
from retracemem.methods.batched_controlled_retrace import BatchedControlledReTraceLLM
from retracemem.methods.contracts import SharedCandidateView
from retracemem.methods.controlled_retrace import ControlledReTraceLLM
from retracemem.methods.directjudge import DirectJudgeLLM
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.budget import BudgetWrappedProvider, GlobalBudget
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.schemas import BeliefNode, EvidenceNode
from retracemem.verifier.prompt_batched_evidence_edge_verifier import PromptBatchedEvidenceEdgeVerifier
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier

DISCLAIMER = (
    "MEMORA ORACLE-CONDITIONED AUTHORIZATION DIAGNOSTIC ONLY — "
    "NOT OFFICIAL END-TO-END MEMORA RESULT."
)
DIAGNOSTIC_SCORING_LABELS = {
    "diagnostic_only": True,
    "oracle_conditioned_candidates": True,
    "official_end_to_end_result": False,
    "official_fama_score": None,
    "scoring_type": "oracle_conditioned_authorization_metrics",
}
PERSONAS = (
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
)


@dataclass(frozen=True)
class MemoraDiagnosticConfig:
    mode: str = "replay"
    reference_root: str = "reference/Memora"
    period: str = "weekly"
    persona: str = "academic_researcher"
    all_personas: bool = False
    limit_questions: int = 5
    provider: str = "siliconflow"
    model: str = "deepseek-ai/DeepSeek-V4-Pro"
    stage_a_execution: str = "batched"
    stage_a_prompt_version: str = "evidence_edge_prediction_batch_v1"
    max_calls: int = 20
    max_tokens: int = 200000
    http_timeout_seconds: float = 120.0
    output_dir: str = "outputs/memora_oracle_diag"


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


def evidence_text_for_sessions(
    sessions: list[dict[str, Any]], session_ids: set[int],
) -> str:
    seen_texts: set[str] = set()
    parts: list[str] = []
    for session in sessions:
        sid = session.get("session_id")
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = -1
        if sid_int not in session_ids:
            continue
        turns: list[str] = []
        for turn in session.get("conversation", []):
            if isinstance(turn, dict):
                speaker = turn.get("speaker") or turn.get("role") or "unknown"
                message = turn.get("message") or turn.get("content") or ""
                if message:
                    turns.append(f"{speaker}: {message}")
        if turns:
            block = f"session {sid}: " + "\n".join(turns)
            if block not in seen_texts:
                seen_texts.add(block)
                parts.append(block)
    return "\n\n".join(parts)


def build_oracle_diagnostic_view(
    period: str, persona: str, question: dict[str, Any],
    sessions: list[dict[str, Any]],
) -> tuple[SharedCandidateView, dict[str, Any]]:
    qid = str(question.get("question_id"))
    memory_items = flatten_values(question.get("memory_evidence", {}))
    forgetting_items = flatten_values(question.get("forgetting_evidence", {}))
    session_ids = {
        int(item.get("session_id"))
        for item in memory_items + forgetting_items
        if str(item.get("session_id", "")).isdigit()
    }
    context_text = evidence_text_for_sessions(sessions, session_ids)
    if not context_text:
        context_text = json.dumps(
            {"memory_evidence": question.get("memory_evidence"),
             "forgetting_evidence": question.get("forgetting_evidence")},
            ensure_ascii=False,
        )
    evidence = EvidenceNode(
        evidence_id=f"memora:{period}:{persona}:{qid}:evidence",
        session_id=f"{period}:{persona}",
        timestamp=question.get("question_date"),
        text=context_text,
        source_dataset="memora",
        source_pointer=f"{period}/{persona}/{qid}",
    )
    beliefs: list[BeliefNode] = []
    for idx, item in enumerate(memory_items):
        val = str(item.get("value") or "").strip()
        if val:
            beliefs.append(BeliefNode(
                belief_id=f"b:{period}:{persona}:{qid}:memory:{idx}",
                proposition=val,
                source_evidence_ids=(evidence.evidence_id,),
                confidence=1.0,
                metadata={"memora_role": "memory_presence",
                          "session_id": item.get("session_id"),
                          "question_date": question.get("question_date")},
            ))
    for idx, item in enumerate(forgetting_items):
        val = str(item.get("value") or "").strip()
        if val:
            beliefs.append(BeliefNode(
                belief_id=f"b:{period}:{persona}:{qid}:forget:{idx}",
                proposition=val,
                source_evidence_ids=(evidence.evidence_id,),
                confidence=1.0,
                metadata={"memora_role": "forgetting_absence",
                          "session_id": item.get("session_id"),
                          "question_date": question.get("question_date")},
            ))
    if not beliefs:
        beliefs.append(BeliefNode(
            belief_id=f"b:{period}:{persona}:{qid}:fallback",
            proposition=str(question.get("question") or ""),
            source_evidence_ids=(evidence.evidence_id,),
            confidence=0.5,
            metadata={"memora_role": "fallback",
                      "question_date": question.get("question_date")},
        ))
    view = SharedCandidateView(
        instance_id=f"memora:{period}:{persona}:{qid}",
        query_id=f"memora:{period}:{persona}:{qid}",
        query=str(question.get("question") or ""),
        evidence_context=(evidence,),
        new_evidence=evidence,
        candidate_beliefs=tuple(beliefs),
        candidate_replacement_beliefs=(),
        candidate_conditions_by_belief=tuple((b.belief_id, ()) for b in beliefs),
        dependency_edges_by_belief=tuple((b.belief_id, ()) for b in beliefs),
    )
    view_meta = {
        "candidate_belief_count": len(beliefs),
        "evidence_chars": len(context_text),
        "selected_sessions": len(session_ids),
    }
    return view, view_meta


def run_memora_oracle_diagnostic(config: MemoraDiagnosticConfig) -> tuple[Path, Path, dict[str, Any]]:
    out = Path(config.output_dir)
    if config.mode == "live-dev" and (
        (out / "memora_development_report.json").exists()
        or (out / "memora_development_manifest.json").exists()
    ):
        raise SystemExit("Refusing to overwrite existing live output directory")
    out.mkdir(parents=True, exist_ok=True)

    adapter = MemoraAdapter(config.reference_root)
    personas = list(PERSONAS if config.all_personas else (config.persona,))
    selected: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for persona in personas:
        sessions = adapter.load_sessions(config.period, persona)
        questions = adapter.load_evaluation_questions(
            config.period, persona,
        )[: max(config.limit_questions, 0)]
        if questions:
            selected.append((persona, sessions, questions))

    total_questions = sum(len(qs) for _, _, qs in selected)
    print(
        f"[PLAN] {DISCLAIMER}\n"
        f"  period={config.period} personas={len(selected)} "
        f"questions={total_questions}\n"
        f"  provider={config.provider} model={config.model}\n"
        f"  stage_a_execution={config.stage_a_execution} "
        f"timeout={config.http_timeout_seconds}s\n"
        f"  max_calls={config.max_calls} max_tokens={config.max_tokens}",
        flush=True,
    )

    budget = GlobalBudget(max_calls=config.max_calls, max_tokens=config.max_tokens)
    cache_dir = out / "caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = str(cache_dir / f"c_{uuid.uuid4()}")

    timeout = config.http_timeout_seconds
    client_a = make_client(
        config.mode, cache_prefix + "_stage_a.jsonl", budget, "stage_a", timeout,
    )
    client_b = make_client(
        config.mode, cache_prefix + "_stage_b.jsonl", budget, "stage_b", timeout,
    )
    client_ans_a = make_answer_client(
        config.mode, cache_prefix + "_answer_a.jsonl", budget, "answer_a", timeout,
    )
    client_ans_b = make_answer_client(
        config.mode, cache_prefix + "_answer_b.jsonl", budget, "answer_b", timeout,
    )

    if config.stage_a_execution == "batched":
        batched_verifier = PromptBatchedEvidenceEdgeVerifier(
            client=client_a, model_id=config.model, provider=config.provider,
            prompt_version=config.stage_a_prompt_version,
        )
        stage_a_runner = BatchedControlledReTraceLLM(
            edge_verifier=batched_verifier, client=client_a,
        )
        verifier_hash = batched_verifier._template_hash
    else:
        per_belief_verifier = PromptEvidenceEdgeVerifier(
            client=client_a, model_id=config.model, provider=config.provider,
            prompt_version=config.stage_a_prompt_version,
        )
        stage_a_runner = ControlledReTraceLLM(
            edge_verifier=per_belief_verifier, client=client_a,
        )
        verifier_hash = per_belief_verifier._template_hash

    stage_b_runner = DirectJudgeLLM(
        client=client_b, model_id=config.model, provider=config.provider,
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
                config.period, persona, question, sessions,
            )
            row: dict[str, Any] = {
                "period": config.period,
                "persona": persona,
                "question_id": qid,
                "question": view.query,
                "evaluation": question.get("evaluation", {}),
                "view_meta": view_meta,
                "candidate_roles": candidate_roles(view.candidate_beliefs),
                "stage_a": {},
                "stage_b": {},
            }

            progress("STAGE_A_START", persona, qid, budget, started)
            try:
                res_a = stage_a_runner.run(view)
                progress("STAGE_A_DONE", persona, qid, budget, started)
                basis_a = [
                    b for b in view.candidate_beliefs
                    if b.belief_id in res_a.authorized_belief_ids
                ]
                progress("ANSWER_A_START", persona, qid, budget, started)
                ans_a = gen_a.generate_answer(
                    view.query, list(basis_a),
                    model_id=config.model, provider=config.provider,
                )
                progress("ANSWER_A_DONE", persona, qid, budget, started)
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
                progress("STAGE_A_ERROR", persona, qid, budget, started)

            progress("STAGE_B_START", persona, qid, budget, started)
            try:
                res_b = stage_b_runner.judge(view)
                progress("STAGE_B_DONE", persona, qid, budget, started)
                basis_b = [
                    b for b in view.candidate_beliefs
                    if b.belief_id in res_b.authorized_belief_ids
                ]
                progress("ANSWER_B_START", persona, qid, budget, started)
                ans_b = gen_b.generate_answer(
                    view.query, list(basis_b),
                    model_id=config.model, provider=config.provider,
                )
                progress("ANSWER_B_DONE", persona, qid, budget, started)
                row["stage_b"] = {
                    "authorized_belief_ids": list(res_b.authorized_belief_ids),
                    "excluded_belief_ids": list(res_b.excluded_belief_ids),
                    "answer": ans_b,
                    "verdict_statuses": {
                        verdict.belief_id: verdict.status.value
                        for verdict in res_b.verdicts
                    },
                    "provenance": res_b.provenance,
                }
            except Exception as exc:
                msg = f"Stage B {persona}/{qid}: {type(exc).__name__}: {exc}"
                row["stage_b"] = {"error": msg}
                errors.append(msg)
                progress("STAGE_B_ERROR", persona, qid, budget, started)

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

    report_path, manifest_path = write_report(
        out, rows, errors, config, budget, verifier_hash, gen_a.template_hash,
        personas, elapsed_seconds=time.time() - started,
    )
    summary = {
        "errors": len(errors),
        "calls": budget.calls,
        "tokens": budget.tokens,
    }
    return report_path, manifest_path, summary


def candidate_roles(candidate_beliefs: tuple[BeliefNode, ...]) -> dict[str, str]:
    return {
        belief.belief_id: str(belief.metadata.get("memora_role") or "")
        for belief in candidate_beliefs
    }


def compute_stage_authorization_metrics(rows: list[dict[str, Any]], stage_key: str) -> dict[str, Any]:
    memory_total = 0
    memory_authorized = 0
    forgetting_total = 0
    forgetting_excluded = 0
    uncertain_count = 0
    scored_rows = 0
    execution_errors = 0

    for row in rows:
        stage = row.get(stage_key, {})
        if not isinstance(stage, dict) or stage.get("error"):
            execution_errors += 1
            continue
        roles = row.get("candidate_roles", {})
        if not isinstance(roles, dict):
            continue
        authorized = set(stage.get("authorized_belief_ids", []))
        excluded = set(stage.get("excluded_belief_ids", []))
        if not authorized and not excluded:
            continue
        scored_rows += 1
        uncertain_ids = uncertain_belief_ids(stage)
        for belief_id, role in roles.items():
            if role == "memory_presence":
                memory_total += 1
                if belief_id in authorized:
                    memory_authorized += 1
            elif role == "forgetting_absence":
                forgetting_total += 1
                if belief_id in excluded:
                    forgetting_excluded += 1
            else:
                continue
            if belief_id in uncertain_ids:
                uncertain_count += 1

    memory_accuracy = safe_ratio(memory_authorized, memory_total)
    forgetting_accuracy = safe_ratio(forgetting_excluded, forgetting_total)
    total_scored = memory_total + forgetting_total
    total_correct = memory_authorized + forgetting_excluded
    balanced_values = [
        value for value in (memory_accuracy, forgetting_accuracy)
        if value is not None
    ]
    return {
        **DIAGNOSTIC_SCORING_LABELS,
        "stage": stage_key,
        "scored_rows": scored_rows,
        "execution_errors": execution_errors,
        "memory_presence_total": memory_total,
        "memory_presence_authorized": memory_authorized,
        "memory_preservation_accuracy": memory_accuracy,
        "forgetting_absence_total": forgetting_total,
        "forgetting_absence_excluded": forgetting_excluded,
        "forgetting_suppression_accuracy": forgetting_accuracy,
        "overall_item_accuracy": safe_ratio(total_correct, total_scored),
        "balanced_authorization_accuracy": (
            sum(balanced_values) / len(balanced_values)
            if len(balanced_values) == 2 else None
        ),
        "uncertain_count": uncertain_count,
        "uncertain_rate": safe_ratio(uncertain_count, total_scored),
    }


def uncertain_belief_ids(stage: dict[str, Any]) -> set[str]:
    uncertain: set[str] = set()
    provenance = stage.get("provenance", {})
    if isinstance(provenance, dict):
        statuses = provenance.get("fine_grained_statuses", {})
        if isinstance(statuses, dict):
            uncertain.update(
                str(belief_id)
                for belief_id, status in statuses.items()
                if str(status).upper() in {"UNRESOLVED", "UNCERTAIN"}
            )
    verdict_statuses = stage.get("verdict_statuses", {})
    if isinstance(verdict_statuses, dict):
        uncertain.update(
            str(belief_id)
            for belief_id, status in verdict_statuses.items()
            if str(status).upper() == "UNCERTAIN"
        )
    return uncertain


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_authorization_diagnostic_metrics(
    rows: list[dict[str, Any]],
    errors: list[str],
    budget: GlobalBudget,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    metrics = {
        **DIAGNOSTIC_SCORING_LABELS,
        "errors": len(errors),
        "global_budget": budget.summary(),
        "stage_a": compute_stage_authorization_metrics(rows, "stage_a"),
        "stage_b": compute_stage_authorization_metrics(rows, "stage_b"),
    }
    if elapsed_seconds is not None:
        metrics["elapsed_seconds"] = elapsed_seconds
    return metrics


def progress(
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


def write_report(
    out: Path, rows: list[dict[str, Any]], errors: list[str],
    config: MemoraDiagnosticConfig, budget: GlobalBudget, verifier_hash: str, answer_hash: str,
    personas: list[str],
    elapsed_seconds: float | None = None,
) -> tuple[Path, Path]:
    diagnostic_metrics = compute_authorization_diagnostic_metrics(
        rows, errors, budget, elapsed_seconds=elapsed_seconds,
    )
    report = {
        "disclaimer": DISCLAIMER,
        "diagnostic_only": True,
        "oracle_conditioned_candidates": True,
        "official_end_to_end_result": False,
        "official_fama_score": None,
        "candidate_source": "memora_evaluation_annotations",
        "scoring": diagnostic_metrics,
        "scoring_type": "oracle_conditioned_authorization_metrics",
        "mode": config.mode,
        "stage_a_execution": config.stage_a_execution,
        "provider": config.provider,
        "model": config.model,
        "stage_a_prompt_version": config.stage_a_prompt_version,
        "http_timeout_seconds": config.http_timeout_seconds,
        "period": config.period,
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
        Path(config.reference_root) / "data" / config.period / p
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
            provider_name=config.provider,
            model_id=config.model,
            temperature=0.0,
            prompt_hashes={"stage_a": verifier_hash, "answer": answer_hash},
            cache_path=str(out / "caches"),
            dataset_checksum=data_checksum,
            metadata={
                "memora_upstream_sha": memora_sha,
                "period": config.period,
                "personas": personas,
                "questions_executed": len(rows),
                "oracle_conditioned_candidates": True,
                "official_end_to_end_result": False,
                "official_fama_score": None,
                "scoring_type": "oracle_conditioned_authorization_metrics",
                "stage_a_execution": config.stage_a_execution,
            },
        ),
        aggregate_cost=budget.summary(),
        instance_count=len(rows),
        output_path=str(report_path),
        errors_or_retries=[{"error": e} for e in errors],
        metadata={
            "diagnostic_only": True,
            "oracle_conditioned_candidates": True,
            "official_end_to_end_result": False,
            "official_fama_score": None,
            "scoring_type": "oracle_conditioned_authorization_metrics",
        },
    )
    manifest_path = out / "memora_development_manifest.json"
    manifest.save(str(manifest_path))
    return report_path, manifest_path
