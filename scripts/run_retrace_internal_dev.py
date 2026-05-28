#!/usr/bin/env python3
"""ReTrace internal development runner — typed fixture edition.

Reads ``data/boundary_audit/boundary_audit_dev.jsonl`` and runs each case
through the canonical typed pipeline using only deterministic offline
components. No external API calls or legacy flat-relation types are used.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retracemem.pipeline import ReTracePipeline
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)
from retracemem.extraction.typed_extractor import ManualTypedBeliefExtractor
from retracemem.verifier.requirement_inducer import ManualRequirementInducer
from retracemem.verifier.contracts import RequirementProposal
from retracemem.verifier.evidence_edge_verifier import ManualEvidenceEdgeVerifier
from retracemem.retrieval.typed_retrievers import (
    ManualImpactCandidateRetriever,
    ManualQueryBeliefRetriever,
)


def _build_case_pipeline(case: dict) -> tuple[ReTracePipeline, str, str, str, list[str]]:
    """Build a typed pipeline for a single boundary-audit case.

    Returns (pipeline, user_id, ev_old_id, ev_new_id, protected_ids).
    """
    case_id = case["case_id"]
    old_belief_text = case["old_belief"]
    new_evidence_text = case["new_evidence"]
    query = case["query"]
    expected_relation = case["expected_relation"]
    condition_text = case.get("condition") or ""
    protected_list = case.get("protected_beliefs", [])

    user_id = f"user_{case_id}"
    old_belief_id = f"belief_old_{case_id}"
    ev_old_id = f"ev_old_{case_id}"
    ev_new_id = f"ev_new_{case_id}"

    # Beliefs extracted from old evidence
    old_beliefs: list[BeliefNode] = [
        BeliefNode(
            belief_id=old_belief_id,
            proposition=old_belief_text,
            source_evidence_ids=(ev_old_id,),
        )
    ]
    protected_ids: list[str] = []
    for idx, prot_text in enumerate(protected_list):
        prot_id = f"belief_prot_{case_id}_{idx}"
        protected_ids.append(prot_id)
        old_beliefs.append(
            BeliefNode(
                belief_id=prot_id,
                proposition=prot_text,
                source_evidence_ids=(ev_old_id,),
            )
        )

    # Beliefs extracted from new evidence
    new_beliefs: list[BeliefNode] = []
    if expected_relation == "SUPERSEDE":
        new_belief_id = f"belief_new_{case_id}"
        new_beliefs.append(
            BeliefNode(
                belief_id=new_belief_id,
                proposition=new_evidence_text,
                source_evidence_ids=(ev_new_id,),
            )
        )

    extractor = ManualTypedBeliefExtractor({
        ev_old_id: old_beliefs,
        ev_new_id: new_beliefs,
    })

    # Requirements (BLOCK / CONDITION buckets)
    proposals: list[RequirementProposal] = []
    if condition_text:
        cond_id = f"cond_{case_id}"
        condition = ConditionNode(condition_id=cond_id, scope_id=user_id, text=condition_text)
        dep = DependencyEdge(
            edge_id=f"dep_{case_id}",
            belief_id=old_belief_id,
            condition_id=cond_id,
            inducer="manual_dev_runner",
        )
        proposals.append(RequirementProposal(condition=condition, dependency_edge=dep))
    inducer = ManualRequirementInducer(proposals)

    # Evidence edges from new evidence
    edge_verifier = ManualEvidenceEdgeVerifier()
    if expected_relation == "BLOCK":
        cond_id = f"cond_{case_id}"
        block_edge = EvidenceEdge(
            edge_id=f"edge_block_{case_id}",
            edge_type=EvidenceEdgeType.BLOCKS,
            evidence_id=ev_new_id,
            target_kind="condition",
            target_id=cond_id,
            verifier="manual_dev_runner",
        )
        edge_verifier.register(block_edge, belief_id=old_belief_id)
    elif expected_relation == "SUPERSEDE":
        new_belief_id = f"belief_new_{case_id}"
        super_edge = EvidenceEdge(
            edge_id=f"edge_super_{case_id}",
            edge_type=EvidenceEdgeType.SUPERSEDES,
            evidence_id=ev_new_id,
            target_kind="belief",
            target_id=old_belief_id,
            verifier="manual_dev_runner",
            replacement_belief_id=new_belief_id,
        )
        edge_verifier.register(super_edge, belief_id=old_belief_id)
    elif expected_relation == "UNCERTAIN":
        uncertain_edge = EvidenceEdge(
            edge_id=f"edge_uncertain_{case_id}",
            edge_type=EvidenceEdgeType.UNCERTAIN,
            evidence_id=ev_new_id,
            target_kind="belief",
            target_id=old_belief_id,
            verifier="manual_dev_runner",
        )
        edge_verifier.register(uncertain_edge, belief_id=old_belief_id)
    # CONDITION / NONE: no evidence edges emitted

    # Impact retriever: new evidence impacts old belief (except NONE)
    impact_map: dict[str, list[str]] = {}
    if expected_relation in ("BLOCK", "SUPERSEDE", "UNCERTAIN", "CONDITION"):
        impact_map[ev_new_id] = [old_belief_id]

    # Query retriever: query returns old belief + protected beliefs
    all_query_ids = [old_belief_id] + protected_ids
    query_retriever = ManualQueryBeliefRetriever({query: all_query_ids})

    pipeline = ReTracePipeline.for_development_fixture(
        extractor=extractor,
        inducer=inducer,
        edge_verifier=edge_verifier,
        impact_retriever=ManualImpactCandidateRetriever(impact_map),
        query_retriever=query_retriever,
    )

    return pipeline, user_id, ev_old_id, ev_new_id, protected_ids


def main() -> None:
    print("==================================================")
    print("Running ReTrace Internal Dev (Typed Fixtures)")
    print("==================================================")

    dev_jsonl_path = "data/boundary_audit/boundary_audit_dev.jsonl"
    if not os.path.exists(dev_jsonl_path):
        print(f"Error: Dev dataset {dev_jsonl_path} not found.")
        sys.exit(1)

    cases: list[dict] = []
    with open(dev_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"Loaded {len(cases)} cases from {dev_jsonl_path}.\n")

    correct_auth = 0
    total_auth = 0
    preserved_protected = 0
    total_protected = 0

    for case in cases:
        case_id = case["case_id"]
        old_belief_text = case["old_belief"]
        new_evidence_text = case["new_evidence"]
        query = case["query"]
        expected_authorized = case["expected_authorized"]

        pipeline, user_id, ev_old_id, ev_new_id, protected_ids = _build_case_pipeline(case)
        old_belief_id = f"belief_old_{case_id}"

        pipeline.reset_user(user_id)

        # Ingest session 1 (establishes old belief + protected beliefs)
        pipeline.ingest_evidence(
            user_id,
            EvidenceNode(
                evidence_id=ev_old_id,
                session_id="session_1",
                timestamp="2026-05-27T01:00:00Z",
                text=f"Pre-existing context: {old_belief_text}",
                source_dataset="boundary_audit",
                source_pointer=case_id,
            ),
        )

        # Ingest session 2 (new evidence triggers verification)
        pipeline.ingest_evidence(
            user_id,
            EvidenceNode(
                evidence_id=ev_new_id,
                session_id="session_2",
                timestamp="2026-05-27T02:00:00Z",
                text=new_evidence_text,
                source_dataset="boundary_audit",
                source_pointer=case_id,
            ),
        )

        # Check authorization
        basis = pipeline.authorized_basis(user_id, query, limit=10)
        authorized_ids = {item["belief_id"] for item in basis}
        is_old_authorized = old_belief_id in authorized_ids

        total_auth += 1
        if is_old_authorized == expected_authorized:
            correct_auth += 1

        # Check protected beliefs
        protected_ok = True
        for prot_id in protected_ids:
            total_protected += 1
            if prot_id in authorized_ids:
                preserved_protected += 1
            else:
                protected_ok = False

        status_str = "SUCCESS" if (is_old_authorized == expected_authorized and protected_ok) else "FAIL"
        print(f"Case {case_id} [{case['bucket']}]: {status_str}")
        print(f"  Old Belief: {old_belief_text}")
        print(f"  New Evidence: {new_evidence_text}")
        print(f"  Expected Auth: {expected_authorized}, Got: {is_old_authorized}")
        print(f"  Protected preserved: {protected_ok}")
        print("-" * 50)

    # Summary
    auth_rate = (correct_auth / total_auth) if total_auth > 0 else 1.0
    pbp_rate = (preserved_protected / total_protected) if total_protected > 0 else 1.0

    print("\n==================================================")
    print("Evaluation Summary")
    print("==================================================")
    print(f"Correct Authorization Rate: {auth_rate * 100:.1f}% ({correct_auth}/{total_auth})")
    print(f"Protected Belief Preservation Rate (PBP): {pbp_rate * 100:.1f}% ({preserved_protected}/{total_protected})")
    print("==================================================")

    if auth_rate == 1.0 and pbp_rate == 1.0:
        print("All Dev Cases Passed Cleanly!")
        sys.exit(0)
    else:
        print("Warning: Some cases failed to meet expected authorization status.")
        sys.exit(1)


if __name__ == "__main__":
    main()
