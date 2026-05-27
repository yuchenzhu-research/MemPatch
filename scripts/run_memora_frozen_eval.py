#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retracemem.adapters.memora_adapter import MemoraAdapter
from retracemem.backends.retrace_backend import ReTraceBackend
from retracemem.backends.retrieval_baseline import RetrievalBaselineBackend
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation import evaluation_record_from_backend_output, write_jsonl
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.extraction.manual_fixture_extractor import ManualFixtureExtractor
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.retrieval.candidate_retriever import SimpleOverlapRetriever
from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType
from retracemem.verifier.heuristic_verifier import HeuristicRelationVerifier
from retracemem.verifier.prompt_verifier import PromptRelationVerifier


DEFAULT_OUTPUT = "outputs/memora/{method}_frozen_eval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Memora Frozen Evaluation.")
    parser.add_argument("--reference-root", default="reference/Memora")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--method", choices=("retrieval_baseline", "retrace"), default="retrace")
    parser.add_argument("--output", default=None)
    parser.add_argument("--disable-ledger", action="store_true")
    parser.add_argument("--disable-gate", action="store_true")
    parser.add_argument("--disable-temporal", action="store_true")
    args = parser.parse_args()

    adapter = MemoraAdapter(args.reference_root)
    roots = adapter.discover_data_roots()
    output = args.output or DEFAULT_OUTPUT.format(method=args.method)

    if not roots:
        print(f"No Memora data roots found under {args.reference_root}; nothing to run.")
        write_jsonl([], output)
        print("records_written: 0")
        return

    root = roots[0]
    period = str(root["period"])
    persona_id = str(root["persona_id"])
    sessions = adapter.load_sessions(period, persona_id)
    questions = adapter.load_evaluation_questions(period, persona_id)[: max(args.limit, 0)]

    if not sessions or not questions:
        print(f"No Memora sessions/questions loaded for {period}/{persona_id}; nothing to run.")
        write_jsonl([], output)
        print("records_written: 0")
        return

    # Backend Setup
    if args.method == "retrieval_baseline":
        backend: Any = RetrievalBaselineBackend()
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        cache = JSONLCache("artifacts/cache/memora_eval_cache.jsonl")
        cost_accountant = CostAccounting()

        if api_key:
            provider_client = MockLLMProvider(default_response="{}")
            client = CachedLLMClient(cache, provider_client, cost_accountant)
            verifier: Any = PromptRelationVerifier(client=client)
        else:
            verifier = HeuristicRelationVerifier()
            client = None

        # Fallback extractor that turns evidence text into a Belief propositional fact
        class MemoraExtractor(ManualFixtureExtractor):
            def extract(self, evidence: EpisodicEvidence) -> list[Belief]:
                res = super().extract(evidence)
                if not res and evidence.text.strip():
                    return [
                        Belief(id=f"belief_{evidence.id}", proposition=evidence.text, supported_by=[evidence.id])
                    ]
                return res

        extractor = MemoraExtractor()

        # Using SimpleOverlapRetriever for N-session scaling performance
        backend = ReTraceBackend(
            extractor=extractor,
            verifier=verifier,
            retriever=SimpleOverlapRetriever(),
            client=client,
            disable_ledger=args.disable_ledger,
            disable_gate=args.disable_gate,
            disable_temporal=args.disable_temporal,
        )

    user_id = f"{period}:{persona_id}"
    backend.reset_user(user_id)

    # Ingest sessions
    for session in sessions:
        backend.ingest_session(
            user_id,
            _session_for_backend(session),
            metadata={"period": period, "persona_id": persona_id},
        )

    # Answer queries
    records = []
    for question_index, question in enumerate(questions, start=1):
        query = str(question.get("question") or "")
        if not query.strip():
            continue
        question_id = str(question.get("question_id") or f"question_{question_index:04d}")
        retrieved = backend.search(user_id, query, limit=10)
        answer = backend.answer(user_id, query, retrieved, metadata={"question_id": question_id})
        records.append(
            evaluation_record_from_backend_output(
                query_id=question_id,
                method=args.method,
                retrieved=retrieved,
                answer=answer,
            )
        )

    write_jsonl(records, output)
    print(f"data_root: {root['root']}")
    print(f"sessions_loaded: {len(sessions)}")
    print(f"questions_loaded: {len(questions)}")
    print(f"records_written: {len(records)}")
    print(f"output_path: {output}")


def _session_for_backend(session: dict[str, Any]) -> dict[str, Any]:
    messages = []
    conversation = session.get("conversation")
    if isinstance(conversation, list):
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            content = turn.get("message") or turn.get("content") or turn.get("text")
            if not isinstance(content, str) or not content.strip():
                continue
            role = str(turn.get("speaker") or turn.get("role") or "")
            messages.append({"role": role, "content": content})

    full_text = " ".join(msg["content"] for msg in messages)

    return {
        "id": str(session.get("session_id") or ""),
        "source_id": str(session.get("date") or session.get("session_id") or ""),
        "text": full_text,
        "messages": messages,
        "metadata": session.get("metadata") if isinstance(session.get("metadata"), dict) else {},
    }


if __name__ == "__main__":
    main()
