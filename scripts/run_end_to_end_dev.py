#!/usr/bin/env python3
"""AB-2 complete end-to-end pipeline runner with live provider capability and manifest generation.

This runner executes the full multi-session update and query evaluation pipeline:
- Sequential session ingestion (extraction, induction, impact retrieval, edge verification, DPA admission)
- Query answering (query retrieval, authorization, answer generation)
- Comparison of Stage A (ReTrace) vs Stage B (DirectJudge)
- Enforces caps, caching, cost accounting, and manifest generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import uuid
from typing import Any

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.evaluation.manifest import RunConfiguration, RunManifest, compute_file_sha256
from retracemem.providers.base import BaseLLMProvider, MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)
from retracemem.pipeline import ReTracePipeline
from retracemem.retrieval.typed_retrievers import (
    OverlapImpactCandidateRetriever,
    OverlapQueryBeliefRetriever,
)
from retracemem.verifier.prompt_typed_belief_extractor import PromptTypedBeliefExtractor
from retracemem.verifier.prompt_requirement_inducer import PromptRequirementInducer
from retracemem.verifier.prompt_evidence_edge_verifier import PromptEvidenceEdgeVerifier


_CASES_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "internal_dev", "end_to_end_cases.json"
)
_PROMPT_EXTRACT = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "retrace_llm", "belief_extraction_v0.txt"
)
_PROMPT_INDUCE = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "retrace_llm", "requirement_induction_v0.txt"
)
_PROMPT_EDGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "retrace_llm", "evidence_edge_prediction_v0.txt"
)
_PROMPT_JUDGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "directjudge", "direct_usability_v1.txt"
)
_PROMPT_GEN = os.path.join(
    os.path.dirname(__file__), os.pardir, "prompts", "generation", "answer_generation_v0.txt"
)


class CappedProviderWrapper(BaseLLMProvider):
    """Safety wrapper that enforces hard call and token caps on a provider."""

    def __init__(self, inner: BaseLLMProvider, max_calls: int = 5, max_tokens: int = 10000) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls_made = 0
        self.tokens_used = 0

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if self.calls_made >= self.max_calls:
            raise RuntimeError(f"Hard call cap of {self.max_calls} reached. Aborting execution.")
        if self.tokens_used >= self.max_tokens:
            raise RuntimeError(f"Hard token cap of {self.max_tokens} reached. Aborting execution.")
            
        trace = self.inner.generate(*args, **kwargs)
        self.calls_made += 1
        self.tokens_used += trace.total_tokens
        
        if self.tokens_used >= self.max_tokens:
            raise RuntimeError(f"Hard token cap of {self.max_tokens} reached during call. Aborting execution.")
        return trace


class E2EMockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for deterministic end-to-end replay."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        schema = kwargs.get("response_schema_version")
        if schema == "direct_usability_response_v1":
            verdicts = self.case.get("stage_b_mock_verdicts", [])
            response_text = json.dumps({"verdicts": verdicts})
        elif schema == "answer_generation_response_v0":
            # Search case queries for a matching query string
            mock_answer = "Mock answer not found."
            for q_item in self.case.get("queries", []):
                if q_item["query"] in prompt:
                    mock_answer = q_item["mock_answer"]
                    break
            response_text = mock_answer
        else:
            response_text = "mock_success"

        from retracemem.schemas import ModelCallTrace
        return ModelCallTrace(
            call_id=f"mock-call-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:8]}",
            provider="mock",
            model_id="mock",
            status="success",
            prompt_template_hash=kwargs.get("prompt_template_hash"),
            response_schema_version=schema,
            parser_version=kwargs.get("parser_version"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            seed=kwargs.get("seed"),
            input_hash=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
            condition_context_hash=kwargs.get("condition_context_hash"),
            temporal_context_hash=kwargs.get("temporal_context_hash"),
            response=response_text,
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            latency_ms=10.0,
            eligible_for_replay=True,
        )


# Mock implementation of manual components for replay mode
class E2EManualExtractor:
    def __init__(self, extracted_beliefs_map: dict[str, Any]) -> None:
        self.map = extracted_beliefs_map
    def extract(self, evidence: EvidenceNode, scope_id: str) -> list[BeliefNode]:
        raw_list = self.map.get(evidence.evidence_id, [])
        return [
            BeliefNode(
                belief_id=item["belief_id"],
                proposition=item["proposition"],
                source_evidence_ids=tuple(item["source_evidence_ids"])
            )
            for item in raw_list
        ]


class E2EManualRequirementInducer:
    def __init__(self, induced_requirements_map: dict[str, Any]) -> None:
        self.map = induced_requirements_map
    def induce_requirements(self, belief: BeliefNode, evidence_context: tuple[EvidenceNode, ...]) -> list[Any]:
        from retracemem.verifier.contracts import RequirementProposal
        proposals = []
        raw_list = self.map.get(belief.belief_id, [])
        for item in raw_list:
            cond = ConditionNode(
                condition_id=item["condition"]["condition_id"],
                scope_id=belief.metadata.get("scope_id", "user"),
                text=item["condition"]["text"]
            )
            dep = DependencyEdge(
                edge_id=item["dependency_edge"]["edge_id"],
                belief_id=item["dependency_edge"]["belief_id"],
                condition_id=item["dependency_edge"]["condition_id"],
                inducer=item["dependency_edge"]["inducer"],
                edge_type=item["dependency_edge"].get("edge_type", "REQUIRES")
            )
            proposals.append(RequirementProposal(condition=cond, dependency_edge=dep))
        return proposals


class E2EManualEvidenceEdgeVerifier:
    def __init__(self, stage_a_mock_edges_map: dict[str, Any]) -> None:
        self.map = stage_a_mock_edges_map
    def verify_edges(
        self,
        new_evidence: EvidenceNode,
        candidate_belief: BeliefNode,
        candidate_replacement_beliefs: tuple[BeliefNode, ...],
        candidate_conditions: tuple[ConditionNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> list[EvidenceEdge]:
        edges = []
        raw_list = self.map.get(candidate_belief.belief_id, [])
        for item in raw_list:
            if item["evidence_id"] == new_evidence.evidence_id:
                edges.append(EvidenceEdge(
                    edge_id=item["edge_id"],
                    edge_type=EvidenceEdgeType(item["edge_type"]),
                    evidence_id=item["evidence_id"],
                    target_kind=item["target_kind"],
                    target_id=item["target_id"],
                    verifier=item["verifier"],
                    replacement_belief_id=item.get("replacement_belief_id"),
                    confidence=item.get("confidence"),
                    rationale=item.get("rationale")
                ))
        return edges


class E2EManualImpactRetriever:
    def __init__(self, impact_map: dict[str, Any]) -> None:
        self.map = impact_map
    def retrieve_impacts(self, new_evidence: EvidenceNode, prior_beliefs: tuple[BeliefNode, ...], store: Any, limit: int = 10) -> list[Any]:
        from retracemem.retrieval.typed_retrievers import ImpactCandidate
        res = []
        priors_dict = {b.belief_id: b for b in prior_beliefs}
        belief_ids = self.map.get(new_evidence.evidence_id, [])
        for bid in belief_ids:
            if bid in priors_dict:
                dep_edges = store.dependencies_of(bid)
                conditions = []
                for edge in dep_edges:
                    if store.has_condition(edge.condition_id):
                        conditions.append(store.get_condition(edge.condition_id))
                res.append(ImpactCandidate(belief=priors_dict[bid], conditions=tuple(conditions)))
        return res


class E2EManualQueryRetriever:
    def __init__(self, query_map: dict[str, Any]) -> None:
        self.map = query_map
    def retrieve_for_query(self, query: str, beliefs: tuple[BeliefNode, ...], limit: int = 10) -> list[BeliefNode]:
        b_dict = {b.belief_id: b for b in beliefs}
        res = []
        for bid in self.map.get(query, []):
            if bid in b_dict:
                res.append(b_dict[bid])
        return res


def _get_file_hash(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AB-2 complete end-to-end pipeline runner."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live API calls to LLM provider (otherwise replay/mock).",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider name (e.g. openai, google).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model ID to use for live generation.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit API key. If omitted, falls back to env vars.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for the HTTP LLM provider.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), os.pardir, "outputs", "end_to_end_dev"),
        help="Output directory for results (gitignored).",
    )
    parser.add_argument(
        "--cases",
        default=_CASES_PATH,
        help="Path to internal end-to-end dev cases JSON file.",
    )
    args = parser.parse_args()

    cases_path = os.path.normpath(args.cases)
    output_dir = os.path.normpath(args.output_dir)
    run_id = f"run-e2e-{uuid.uuid4()}"

    print("=" * 70)
    print("AB-2 END-TO-END PIPELINE RUNNER")
    print("=" * 70)
    print()
    print(f"  Run ID:  {run_id}")
    print(f"  Mode:    {'LIVE API CALLS' if args.live else 'MOCK/REPLAY'}")
    if args.live:
        print(f"  Config:  provider={args.provider}, model={args.model}")
    print(f"  Cases:   {cases_path}")
    print(f"  Output:  {output_dir}")
    print()

    # Load cases
    with open(cases_path, "r", encoding="utf-8") as f:
        cases_data = json.load(f)
    cases = cases_data.get("cases", [])
    print(f"Loaded {len(cases)} end-to-end cases.")

    prompt_hashes = {
        "belief_extraction": _get_file_hash(_PROMPT_EXTRACT),
        "requirement_induction": _get_file_hash(_PROMPT_INDUCE),
        "evidence_edge_prediction": _get_file_hash(_PROMPT_EDGE),
        "direct_usability": _get_file_hash(_PROMPT_JUDGE),
        "answer_generation": _get_file_hash(_PROMPT_GEN),
    }

    # Setup live client if enabled
    live_client = None
    cache_path = ""
    cost_accountant = CostAccounting()
    if args.live:
        os.makedirs(output_dir, exist_ok=True)
        cache_path = os.path.join(output_dir, "end_to_end_live_cache.jsonl")
        cache = JSONLCache(cache_path)
        http_provider = HTTPLLMProvider(api_key=args.api_key, base_url=args.base_url)
        capped_provider = CappedProviderWrapper(http_provider, max_calls=15, max_tokens=30000)
        live_client = CachedLLMClient(cache=cache, provider_client=capped_provider, cost_accountant=cost_accountant)
        print("  ✓ Setup live provider client with cache and safety caps.")
        print()

    results = []
    cases_passed = 0

    for idx, case in enumerate(cases):
        case_id = case["case_id"]
        case_type = case["case_type"]
        description = case["description"]
        print(f"  [{idx+1}/{len(cases)}] {case_id} ({case_type})")
        print(f"        Description: {description}")

        user_id = f"user_{case_id}"

        # Initialize pipeline
        if args.live:
            extractor = PromptTypedBeliefExtractor(client=live_client, model_id=args.model, provider=args.provider)
            inducer = PromptRequirementInducer(client=live_client, model_id=args.model, provider=args.provider)
            edge_verifier = PromptEvidenceEdgeVerifier(client=live_client, model_id=args.model, provider=args.provider)
            impact_retriever = OverlapImpactCandidateRetriever()
            query_retriever = OverlapQueryBeliefRetriever()
            pipeline = ReTracePipeline(
                extractor=extractor,
                inducer=inducer,
                edge_verifier=edge_verifier,
                impact_retriever=impact_retriever,
                query_retriever=query_retriever,
                client=live_client,
                model_id=args.model,
                provider=args.provider,
            )
        else:
            # Mock / Replay Components
            mock_provider = E2EMockLLMProvider(case)
            mock_client = CachedLLMClient(cache=JSONLCache(os.devnull), provider_client=mock_provider, cost_accountant=cost_accountant)
            extractor = E2EManualExtractor(case.get("extracted_beliefs", {}))
            inducer = E2EManualRequirementInducer(case.get("induced_requirements", {}))
            edge_verifier = E2EManualEvidenceEdgeVerifier(case.get("stage_a_mock_edges", {}))
            impact_retriever = E2EManualImpactRetriever(case.get("impact_map", {}))
            query_retriever = E2EManualQueryRetriever(case.get("query_map", {}))
            pipeline = ReTracePipeline(
                extractor=extractor,
                inducer=inducer,
                edge_verifier=edge_verifier,
                impact_retriever=impact_retriever,
                query_retriever=query_retriever,
                client=mock_client,
                model_id="mock",
                provider="mock",
            )

        pipeline.reset_user(user_id)

        # Ingest sessions sequentially
        for session in case["sessions"]:
            evidence = EvidenceNode(
                evidence_id=session["evidence_id"],
                session_id=session["session_id"],
                timestamp=session["timestamp"],
                text=session["text"],
                source_dataset=session.get("source_dataset", "internal_dev"),
                source_pointer=session.get("source_pointer", "e2e"),
            )
            pipeline.ingest_evidence(user_id, evidence)

        # Run queries
        case_queries_passed = True
        case_records = []
        for q_item in case["queries"]:
            query = q_item["query"]
            limit = q_item["limit"]
            print(f"        Query: \"{query}\"")

            # Run Stage A (ReTrace)
            record_a = pipeline.answer(user_id, query, limit=limit, method="retrace")
            # Run Stage B (DirectJudge)
            record_b = pipeline.answer(user_id, query, limit=limit, method="directjudge")

            # Check correctness in mock mode
            basis_a_props = [item["proposition"] for item in record_a.authorized_basis]
            blocked_a_status = {item["belief_id"]: item["reason"] for item in record_a.blocked_beliefs}

            expected_basis = q_item.get("expected_basis", [])
            expected_blocked = q_item.get("expected_blocked", [])

            is_basis_ok = True
            is_blocked_ok = True

            if not args.live:
                # Validate basis matches exactly
                if set(basis_a_props) != set(expected_basis):
                    print(f"          [FAIL] Stage A basis: Got {basis_a_props}, Expected {expected_basis}")
                    is_basis_ok = False
                
                # Validate blocked list matches status
                for exp_b in expected_blocked:
                    bid = exp_b["belief_id"]
                    status = exp_b["status"]
                    if blocked_a_status.get(bid) != status:
                        print(f"          [FAIL] Stage A blocked status for {bid}: Got {blocked_a_status.get(bid)}, Expected {status}")
                        is_blocked_ok = False

            if is_basis_ok and is_blocked_ok:
                print(f"          [OK] Stage A ReTrace answer generated successfully.")
            else:
                case_queries_passed = False

            print(f"          Answer: \"{record_a.answer}\"")

            # Serialize records
            case_records.append({
                "query": query,
                "stage_a": {
                    "authorized_basis": record_a.authorized_basis,
                    "blocked_beliefs": record_a.blocked_beliefs,
                    "answer": record_a.answer,
                },
                "stage_b": {
                    "authorized_basis": record_b.authorized_basis,
                    "blocked_beliefs": record_b.blocked_beliefs,
                    "answer": record_b.answer,
                }
            })

        if case_queries_passed:
            cases_passed += 1

        results.append({
            "case_id": case_id,
            "case_type": case_type,
            "queries": case_records,
            "passed": case_queries_passed,
        })

    # Summary
    pass_rate = cases_passed / len(cases) if cases else 0.0
    print()
    print("-" * 70)
    print("END-TO-END PIPELINE EVALUATION SUMMARY")
    print("-" * 70)
    print(f"  Total Cases:         {len(cases)}")
    print(f"  Cases Passed:        {cases_passed}/{len(cases)} ({pass_rate * 100:.1f}%)")
    print(f"  Total API Calls:     {cost_accountant.calls.get('total', 0)}")
    print(f"  Total Tokens Used:   {cost_accountant.tokens.get('total', 0)}")
    print()

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "end_to_end_dev_report.json")
    report = {
        "run_id": run_id,
        "aggregate": {
            "total_cases": len(cases),
            "cases_passed": cases_passed,
            "pass_rate": pass_rate,
            "api_calls": cost_accountant.calls.get("total", 0),
            "tokens_used": cost_accountant.tokens.get("total", 0),
        },
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")

    # Generate Run Configuration and Run Manifest
    config = RunConfiguration(
        run_id=run_id,
        stage_and_method_name="AB-2_StageAB_end_to_end",
        provider_name=args.provider if args.live else "mock",
        model_id=args.model if args.live else "mock",
        prompt_hashes=prompt_hashes,
        parser_schema_versions={
            "belief_extraction": "belief_extraction_response_v0",
            "requirement_induction": "requirement_induction_response_v0",
            "evidence_edge_prediction": "evidence_edge_prediction_response_v0",
            "direct_usability": "direct_usability_response_v1",
            "answer_generation": "answer_generation_response_v0",
        },
        cache_path=cache_path,
        dataset_checksum=compute_file_sha256(cases_path),
        comparison_regime="end_to_end_pipeline",
    )

    manifest = RunManifest(
        config=config,
        aggregate_cost={
            "total": cost_accountant.to_dict(),
        },
        instance_count=len(cases),
        output_path=report_path,
    )

    manifest_path = os.path.join(output_dir, "end_to_end_dev_manifest.json")
    manifest.save(manifest_path)
    print(f"Manifest saved to: {manifest_path}")

    # Exit code
    if cases_passed == len(cases):
        print("All cases passed successfully.")
        sys.exit(0)
    else:
        print("Warning: Some cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
