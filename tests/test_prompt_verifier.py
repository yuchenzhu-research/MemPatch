from __future__ import annotations

import json
import os
import pathlib
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.extraction.manual_fixture_extractor import ManualFixtureExtractor
from retracemem.providers.base import MockLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.retrieval.candidate_retriever import MockCandidateRetriever, SimpleOverlapRetriever
from retracemem.schemas import Belief, EpisodicEvidence, RelationType
from retracemem.verifier.prompt_verifier import PromptRelationVerifier


def _mock_cached_client(
    cache_dir: pathlib.Path,
    responses: dict[str, str] | None = None,
    default_response: str = "mocked response",
    status: str = "success",
) -> CachedLLMClient:
    cache_path = str(cache_dir / "temp_cache_test.jsonl")
    cache = JSONLCache(cache_path=cache_path)
    mock_provider = MockLLMProvider(responses=responses, default_response=default_response, status=status)
    return CachedLLMClient(cache=cache, provider_client=mock_provider, cost_accountant=CostAccounting())


def test_prompt_verifier_success(tmp_path: pathlib.Path) -> None:
    ev = EpisodicEvidence(id="ev_1", timestamp="2026-05-27T00:00:00Z", text="I broke my leg.", source_id="s1")
    belief = Belief(id="b_1", proposition="I commute by bicycle.", supported_by=["ev_0"])

    response_data = {
        "relation": "BLOCK",
        "target_belief": None,
        "condition": "cycling ability",
        "rationale": "A broken leg prevents bicycle commuting.",
        "confidence": 0.95,
    }
    response_str = json.dumps(response_data)

    client = _mock_cached_client(tmp_path, default_response=response_str)
    verifier = PromptRelationVerifier(client=client)

    prediction = verifier.verify(ev, belief)

    assert prediction.relation == RelationType.BLOCK
    assert prediction.condition == "cycling ability"
    assert prediction.confidence == 0.95
    assert prediction.rationale == "A broken leg prevents bicycle commuting."
    assert prediction.evidence_id == ev.id
    assert prediction.belief_id == belief.id


def test_prompt_verifier_fail_closed_on_invalid_json(tmp_path: pathlib.Path) -> None:
    ev = EpisodicEvidence(id="ev_1", timestamp="2026-05-27T00:00:00Z", text="I broke my leg.", source_id="s1")
    belief = Belief(id="b_1", proposition="I commute by bicycle.", supported_by=["ev_0"])

    client = _mock_cached_client(tmp_path, default_response="Not a JSON string at all!")
    verifier = PromptRelationVerifier(client=client)

    prediction = verifier.verify(ev, belief)

    assert prediction.relation == RelationType.UNCERTAIN
    assert "Parse failure" in prediction.rationale
    assert prediction.confidence == 0.0


def test_prompt_verifier_fail_closed_on_api_error(tmp_path: pathlib.Path) -> None:
    ev = EpisodicEvidence(id="ev_1", timestamp="2026-05-27T00:00:00Z", text="I broke my leg.", source_id="s1")
    belief = Belief(id="b_1", proposition="I commute by bicycle.", supported_by=["ev_0"])

    client = _mock_cached_client(tmp_path, status="failure")
    verifier = PromptRelationVerifier(client=client)

    prediction = verifier.verify(ev, belief)

    assert prediction.relation == RelationType.UNCERTAIN
    assert "LLM API failure" in prediction.rationale
    assert prediction.confidence == 0.0


def test_simple_overlap_retriever() -> None:
    ev = EpisodicEvidence(id="ev_1", timestamp="2026-05-27T00:00:00Z", text="I broke my leg.", source_id="s1")
    b1 = Belief(id="b_1", proposition="User commonly rides a bicycle.", supported_by=["ev_0"])
    b2 = Belief(id="b_2", proposition="User broke a leg hiking.", supported_by=["ev_0"])
    b3 = Belief(id="b_3", proposition="User likes spicy food.", supported_by=["ev_0"])

    retriever = SimpleOverlapRetriever()
    candidates = retriever.retrieve_candidates(ev, [b1, b2, b3])

    assert len(candidates) == 1
    assert candidates[0].id == "b_2"


def test_mock_candidate_retriever() -> None:
    ev = EpisodicEvidence(id="ev_1", timestamp="2026-05-27T00:00:00Z", text="I broke my leg.", source_id="s1")
    b1 = Belief(id="b_1", proposition="User rides bicycle.", supported_by=["ev_0"])
    b2 = Belief(id="b_2", proposition="User likes spicy food.", supported_by=["ev_0"])

    retriever = MockCandidateRetriever()
    retriever.register("ev_1", [b1])

    candidates = retriever.retrieve_candidates(ev, [b1, b2])
    assert len(candidates) == 1
    assert candidates[0].id == "b_1"


def test_manual_fixture_extractor() -> None:
    ev = EpisodicEvidence(id="ev_1", timestamp="2026-05-27T00:00:00Z", text="I broke my leg.", source_id="s1")
    b = Belief(id="b_1", proposition="User broke their leg", supported_by=["ev_1"])

    extractor = ManualFixtureExtractor()
    extractor.register("ev_1", [b])

    beliefs = extractor.extract(ev)
    assert len(beliefs) == 1
    assert beliefs[0].proposition == "User broke their leg"
    assert beliefs[0].supported_by == ["ev_1"]
