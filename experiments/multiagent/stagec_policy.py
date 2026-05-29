from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
from retracemem.authorization import EvidenceProposalBatch
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    ProposalPolicyOutput,
)
from experiments.multiagent.export_stagec_sft import SYSTEM_PROMPT, format_user_prompt

CANONICAL_ACTIONS = {"SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"}


class PromptTypedRevisionPolicy:
    policy_variant = "prompt"

    def build_messages(
        self,
        submission: FixedCandidateSubmission,
    ) -> Tuple[Dict[str, str], ...]:
        """Construct the prompt messages for the policy using method-visible context."""
        # Using the same formatting helper as SFT export to ensure zero discrepancy
        # Construct a fake StageCTrainingExample wrapper for formatting helper
        from experiments.multiagent.contracts import StageCTrainingExample
        fake_ex = StageCTrainingExample(
            example_id="temp_id",
            episode_id="temp_ep",
            submission_id=submission.submission_id,
            method_visible_input=submission,
            targets=(),
            split="development_only",
            domain="unknown",
            failure_type="unknown",
            label_source="temporary",
        )
        user_content = format_user_prompt(fake_ex)
        
        return (
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        )

    def parse_response(
        self,
        response_text: str,
        *,
        example_id: str,
        submission_id: str,
    ) -> ProposalPolicyOutput:
        """Parse LLM JSON response text into ProposalPolicyOutput."""
        errors: List[str] = []
        parsing_valid = True
        edges: List[EvidenceEdge] = []
        
        cleaned_text = response_text.strip()
        
        # Try to locate JSON block if wrapped in markdown code fence
        if cleaned_text.startswith("```"):
            # Strip ```json or ``` from start and ``` from end
            lines = cleaned_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned_text)
            if not isinstance(parsed, list):
                raise ValueError("LLM response must be a JSON array of objects.")
                
            for idx, item in enumerate(parsed):
                if not isinstance(item, dict):
                    raise ValueError(f"Array item at index {idx} must be a JSON object.")
                    
                action = item.get("action_type")
                if not action:
                    raise ValueError(f"Item at index {idx} is missing required field 'action_type'.")
                    
                if action not in CANONICAL_ACTIONS:
                    raise ValueError(f"Action '{action}' at index {idx} is not in the canonical vocabulary.")
                    
                if action == "NO_REVISION":
                    continue
                    
                # Parse edge fields
                target_belief_id = item.get("target_belief_id")
                target_condition_id = item.get("target_condition_id")
                replacement_belief_id = item.get("replacement_belief_id")
                evidence_ids = item.get("evidence_ids", [])
                
                # Check target kinds
                target_kind = None
                target_id = None
                if target_belief_id:
                    target_kind = "belief"
                    target_id = target_belief_id
                elif target_condition_id:
                    target_kind = "condition"
                    target_id = target_condition_id
                else:
                    raise ValueError(f"Action '{action}' at index {idx} must specify either target_belief_id or target_condition_id.")
                    
                if action == "SUPERSEDES" and not replacement_belief_id:
                    raise ValueError(f"SUPERSEDES action at index {idx} requires replacement_belief_id.")
                    
                if not evidence_ids or not isinstance(evidence_ids, list):
                    raise ValueError(f"Action '{action}' at index {idx} requires a non-empty list of evidence_ids.")
                
                # Build canonical DPA edge
                # Use a deterministic edge ID
                edge_id = f"edge_policy_{example_id}_{idx}"
                edge = EvidenceEdge(
                    edge_id=edge_id,
                    edge_type=EvidenceEdgeType(action) if hasattr(EvidenceEdgeType, action) else action,
                    evidence_id=str(evidence_ids[0]),
                    target_kind=target_kind,
                    target_id=target_id,
                    verifier="stagec_policy",
                    replacement_belief_id=replacement_belief_id,
                    rationale=item.get("rationale", "Propose by Stage C Prompt Policy"),
                )
                edges.append(edge)
                
        except Exception as e:
            parsing_valid = False
            errors.append(f"Parsing failed: {str(e)}")
            
        proposal_batches = ()
        if parsing_valid and edges:
            proposal_batches = (
                EvidenceProposalBatch(
                    edges=tuple(edges),
                    metadata={"parser": "PromptTypedRevisionPolicy"},
                ),
            )
            
        return ProposalPolicyOutput(
            example_id=example_id,
            submission_id=submission_id,
            policy_variant=self.policy_variant,
            proposal_batches=proposal_batches,
            parsing_valid=parsing_valid,
            errors=tuple(errors),
        )
