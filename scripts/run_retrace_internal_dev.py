#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

# Ensure src/ is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retracemem.backends.retrace_backend import ReTraceBackend
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.evaluation.cost_accounting import CostAccounting
from retracemem.extraction.manual_fixture_extractor import ManualFixtureExtractor
from retracemem.providers.base import BaseLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.schemas import Belief, ModelCallTrace, RelationPrediction, RelationType
from retracemem.verifier.prompt_verifier import PromptRelationVerifier


class CaseMockProvider(BaseLLMProvider):
    """Dynamic mock provider that returns correct labels based on analyzed belief target."""

    def __init__(self, old_text: str, expected_rel: str, condition: str, prot_texts: list[str]) -> None:
        self.old_text = old_text
        self.expected_rel = expected_rel
        self.condition = condition
        self.prot_texts = prot_texts

    def generate(
        self,
        prompt: str,
        model_id: str,
        provider: str,
        model_revision_or_api_version: str | None = None,
        prompt_template_hash: str | None = None,
        response_schema_version: str | None = None,
        parser_version: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        condition_context_hash: str | None = None,
        temporal_context_hash: str | None = None,
        eligible_for_replay: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ModelCallTrace:
        rel = "NONE"
        c_val = None

        if self.old_text in prompt:
            rel = self.expected_rel
            c_val = self.condition if self.condition else None

        target_b = None
        if rel == "SUPERSEDE":
            target_b = "new_superseding_belief_placeholder"

        mock_data = {
            "relation": rel,
            "target_belief": target_b,
            "condition": c_val,
            "rationale": "Dynamic mock verifier output",
            "confidence": 0.99,
        }

        return ModelCallTrace(
            call_id="mock-call",
            provider=provider,
            model_id=model_id,
            status="success",
            response=json.dumps(mock_data),
            latency_ms=5.0,
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            eligible_for_replay=eligible_for_replay,
        )


def main() -> None:
    print("==================================================")
    print("Running ReTrace Internal Dev End-to-End Prototype")
    print("==================================================")

    dev_jsonl_path = "data/boundary_audit/boundary_audit_dev.jsonl"
    if not os.path.exists(dev_jsonl_path):
        print(f"Error: Dev dataset {dev_jsonl_path} not found.")
        sys.exit(1)

    cases = []
    with open(dev_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"Loaded {len(cases)} cases from {dev_jsonl_path}.\n")

    correct_obsolete_blocked = 0
    total_obsolete_expected_blocked = 0
    preserved_protected_beliefs = 0
    total_protected_beliefs = 0

    cache_path = "artifacts/cache/internal_dev_run_cache.jsonl"
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass

    for case in cases:
        case_id = case["case_id"]
        old_belief_text = case["old_belief"]
        new_evidence_text = case["new_evidence"]
        query = case["query"]
        expected_relation = case["expected_relation"]
        expected_authorized = case["expected_authorized"]
        condition = case.get("condition") or ""
        protected_list = case.get("protected_beliefs", [])

        cache = JSONLCache(cache_path=cache_path)
        mock_provider = CaseMockProvider(old_belief_text, expected_relation, condition, protected_list)
        cost_accountant = CostAccounting()
        client = CachedLLMClient(cache, mock_provider, cost_accountant)

        extractor = ManualFixtureExtractor()

        old_belief_id = f"belief_old_{case_id}"
        evidence_old_id = f"ev_old_{case_id}"
        evidence_new_id = f"ev_new_{case_id}"

        # Setup extraction mappings
        extractor.register(
            evidence_old_id,
            [Belief(id=old_belief_id, proposition=old_belief_text, supported_by=[evidence_old_id])],
        )

        protected_belief_ids = []
        for idx, prot_text in enumerate(protected_list):
            prot_id = f"belief_prot_{case_id}_{idx}"
            protected_belief_ids.append(prot_id)
            existing_beliefs = extractor.fixtures.get(evidence_old_id, [])
            existing_beliefs.append(Belief(id=prot_id, proposition=prot_text, supported_by=[evidence_old_id]))
            extractor.fixtures[evidence_old_id] = existing_beliefs

        if expected_relation == "SUPERSEDE":
            new_belief_id = f"belief_new_{case_id}"
            extractor.register(
                evidence_new_id,
                [
                    Belief(
                        id=new_belief_id,
                        proposition="The user has updated preference or location.",
                        supported_by=[evidence_new_id],
                    )
                ],
            )

        # Retrieve all beliefs for verification candidate list in prototype
        class PrototypeRetriever:
            def retrieve_candidates(self, new_ev, all_b):
                return all_b

        # Setup verifier and backend
        verifier = PromptRelationVerifier(client=client)

        if expected_relation == "SUPERSEDE":
            new_belief_id = f"belief_new_{case_id}"
            original_verify = verifier.verify

            def patched_verify(new_ev, cand_b, ctx=None):
                pred = original_verify(new_ev, cand_b, ctx)
                if pred.relation == RelationType.SUPERSEDE:
                    import dataclasses

                    return dataclasses.replace(pred, target_belief_id=new_belief_id)
                return pred

            verifier.verify = patched_verify

        backend = ReTraceBackend(
            extractor=extractor, verifier=verifier, retriever=PrototypeRetriever(), client=client
        )

        user_id = f"user_{case_id}"
        backend.reset_user(user_id)

        # Pre-populate dependency relation
        if condition:
            backend.stores[user_id].add_relation(
                RelationPrediction(
                    relation=RelationType.REQUIRED_BY,
                    evidence_id=evidence_old_id,
                    belief_id=old_belief_id,
                    condition=condition,
                )
            )

        # Ingest session 1
        backend.ingest_session(
            user_id=user_id,
            session={
                "id": evidence_old_id,
                "timestamp": "2026-05-27T01:00:00Z",
                "text": f"Pre-existing context: {old_belief_text}. " + " ".join(protected_list),
                "source_id": "session_1",
            },
        )

        # Ingest session 2 (triggers verification)
        backend.ingest_session(
            user_id=user_id,
            session={
                "id": evidence_new_id,
                "timestamp": "2026-05-27T02:00:00Z",
                "text": new_evidence_text,
                "source_id": "session_2",
            },
        )

        # Verify query basis
        basis = backend.search(user_id=user_id, query=query, limit=10)
        authorized_ids = {item["belief_id"] for item in basis}

        is_old_belief_authorized = old_belief_id in authorized_ids

        if not expected_authorized:
            total_obsolete_expected_blocked += 1
            if not is_old_belief_authorized:
                correct_obsolete_blocked += 1
        else:
            # If expected to be authorized, checking it is indeed authorized
            total_obsolete_expected_blocked += 1
            if is_old_belief_authorized:
                correct_obsolete_blocked += 1

        # Check protected beliefs (must remain authorized)
        protected_ok = True
        for prot_id in protected_belief_ids:
            total_protected_beliefs += 1
            if prot_id in authorized_ids:
                preserved_protected_beliefs += 1
            else:
                protected_ok = False

        status_str = "SUCCESS" if (is_old_belief_authorized == expected_authorized and protected_ok) else "FAIL"
        print(f"Case {case_id} [{case['bucket']}]: {status_str}")
        print(f"  Old Belief: {old_belief_text}")
        print(f"  New Evidence: {new_evidence_text}")
        print(f"  Expected Auth: {expected_authorized}, Got: {is_old_belief_authorized}")
        print(f"  Protected preserved: {protected_ok}")
        print("-" * 50)

    # Summary
    block_success_rate = (
        (correct_obsolete_blocked / total_obsolete_expected_blocked) if total_obsolete_expected_blocked > 0 else 1.0
    )
    pbp_rate = (
        (preserved_protected_beliefs / total_protected_beliefs) if total_protected_beliefs > 0 else 1.0
    )

    print("\n==================================================")
    print("Evaluation Summary")
    print("==================================================")
    print(
        f"Correct Authorization Rate: {block_success_rate * 100:.1f}% ({correct_obsolete_blocked}/{total_obsolete_expected_blocked})"
    )
    print(
        f"Protected Belief Preservation Rate (PBP): {pbp_rate * 100:.1f}% ({preserved_protected_beliefs}/{total_protected_beliefs})"
    )
    print("==================================================")

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass

    if block_success_rate == 1.0 and pbp_rate == 1.0:
        print("All Dev Cases Passed Cleanly!")
        sys.exit(0)
    else:
        print("Warning: Some cases failed to meet expected authorization status.")
        sys.exit(1)


if __name__ == "__main__":
    main()
