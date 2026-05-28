from __future__ import annotations

import json
import os
from retracemem.schemas import EvidenceNode, BeliefNode
from retracemem.pipeline import ReTracePipeline
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers.cached_client import CachedLLMClient
from scripts.run_end_to_end_dev import (
    E2EMockLLMProvider,
    E2EManualExtractor,
    E2EManualRequirementInducer,
    E2EManualEvidenceEdgeVerifier,
    E2EManualImpactRetriever,
    E2EManualQueryRetriever,
)


def test_end_to_end_pipeline_mock_cases() -> None:
    cases_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "internal_dev", "end_to_end_cases.json"
    )
    with open(cases_path, "r", encoding="utf-8") as f:
        cases_data = json.load(f)
    cases = cases_data["cases"]

    for case in cases:
        user_id = f"test_e2e_{case['case_id']}"
        mock_provider = E2EMockLLMProvider(case)
        mock_client = CachedLLMClient(cache=JSONLCache(os.devnull), provider_client=mock_provider)
        
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

        # Ingest sessions
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

        # Verify query and expected basis
        for q_item in case["queries"]:
            query = q_item["query"]
            limit = q_item["limit"]
            record_a = pipeline.answer(user_id, query, limit=limit, method="retrace")
            
            basis_a_props = [item["proposition"] for item in record_a.authorized_basis]
            blocked_a_status = {item["belief_id"]: item["reason"] for item in record_a.blocked_beliefs}

            expected_basis = q_item.get("expected_basis", [])
            expected_blocked = q_item.get("expected_blocked", [])

            assert set(basis_a_props) == set(expected_basis)
            for exp_b in expected_blocked:
                assert blocked_a_status.get(exp_b["belief_id"]) == exp_b["status"]
            
            assert record_a.answer == q_item["mock_answer"]
