from __future__ import annotations

import pytest
from retracemem.extraction.typed_extractor import ManualTypedBeliefExtractor
from retracemem.schemas import BeliefNode, EvidenceNode


def test_valid_grounded_extraction():
    evidence = EvidenceNode(
        evidence_id="ev_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice moved to SF.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    belief = BeliefNode(
        belief_id="belief_1",
        proposition="Alice resides in SF.",
        source_evidence_ids=("ev_1",),
    )
    
    extractor = ManualTypedBeliefExtractor(mappings={"ev_1": [belief]})
    results = extractor.extract(evidence, scope_id="user_alice")
    assert results == [belief]


def test_rejected_ungrounded_belief():
    evidence = EvidenceNode(
        evidence_id="ev_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice moved to SF.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    belief = BeliefNode(
        belief_id="belief_1",
        proposition="Alice resides in SF.",
        source_evidence_ids=("ev_other_unrelated",),
    )
    
    extractor = ManualTypedBeliefExtractor(mappings={"ev_1": [belief]})
    with pytest.raises(ValueError, match="Grounding violation"):
        extractor.extract(evidence, scope_id="user_alice")


def test_multiple_extracted_candidate_new_beliefs():
    evidence = EvidenceNode(
        evidence_id="ev_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice moved to SF and bought a bicycle.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    belief_1 = BeliefNode(
        belief_id="belief_1",
        proposition="Alice resides in SF.",
        source_evidence_ids=("ev_1",),
    )
    belief_2 = BeliefNode(
        belief_id="belief_2",
        proposition="Alice owns a bicycle.",
        source_evidence_ids=("ev_1",),
    )
    
    extractor = ManualTypedBeliefExtractor(mappings={"ev_1": [belief_1, belief_2]})
    results = extractor.extract(evidence, scope_id="user_alice")
    assert len(results) == 2
    assert belief_1 in results
    assert belief_2 in results


def test_missing_or_empty_scope_id():
    evidence = EvidenceNode(
        evidence_id="ev_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice moved to SF.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    belief = BeliefNode(
        belief_id="belief_1",
        proposition="Alice resides in SF.",
        source_evidence_ids=("ev_1",),
    )
    
    extractor = ManualTypedBeliefExtractor(mappings={"ev_1": [belief]})
    with pytest.raises(ValueError, match="scope_id is required"):
        extractor.extract(evidence, scope_id="")
