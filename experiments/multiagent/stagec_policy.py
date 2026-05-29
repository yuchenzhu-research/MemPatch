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
        submission: FixedCandidateSubmission,
    ) -> ProposalPolicyOutput:
        """Parse LLM JSON response text into ProposalPolicyOutput."""
        errors: List[str] = []
        parsing_valid = True
        edges: List[EvidenceEdge] = []
        parsed_targets: List[TypedRevisionTarget] = []
        
        cleaned_text = response_text.strip()
        
        # Try to locate JSON block if wrapped in markdown code fence
        if cleaned_text.startswith("```"):
            lines = cleaned_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()

        # Build validation lookups
        valid_evidence_ids = {ev.evidence_id for ev in submission.evidence_context} | {submission.new_evidence_id}
        valid_candidate_belief_ids = {b.belief_id for b in submission.candidate_beliefs}
        valid_replacement_belief_ids = {b.belief_id for b in submission.candidate_replacement_beliefs}
        valid_condition_ids = set()
        for _, conds in submission.candidate_conditions_by_belief:
            for c in conds:
                valid_condition_ids.add(c.condition_id)

        deduplicated_parsed: List[TypedRevisionTarget] = []
        has_duplicates = False

        try:
            parsed = json.loads(cleaned_text)
            if not isinstance(parsed, list):
                raise ValueError("LLM response must be a JSON array of objects.")
            
            if not parsed:
                raise ValueError("Parsed action array is empty.")
                
            for idx, item in enumerate(parsed):
                if not isinstance(item, dict):
                    raise ValueError(f"Array item at index {idx} must be a JSON object.")
                    
                action = item.get("action_type")
                if not action:
                    raise ValueError(f"Item at index {idx} is missing required field 'action_type'.")
                    
                if action in ("AUTHORIZED", "BLOCKED", "SUPERSEDED", "UNRESOLVED"):
                    raise ValueError(f"Action '{action}' is a final DPA status, not a valid policy proposal action.")
                    
                if action not in CANONICAL_ACTIONS:
                    raise ValueError(f"Action '{action}' is not canonical.")
                    
                evidence_ids = item.get("evidence_ids", [])
                if not isinstance(evidence_ids, list):
                    raise ValueError(f"Action '{action}' evidence_ids must be an array.")
                if not evidence_ids:
                    raise ValueError(f"Action '{action}' requires a non-empty evidence_ids array.")
                if submission.new_evidence_id not in evidence_ids:
                    raise ValueError(f"Action '{action}' evidence_ids must explicitly include submission's new_evidence_id '{submission.new_evidence_id}'.")
                for ev_id in evidence_ids:
                    if ev_id not in valid_evidence_ids:
                        raise ValueError(f"Evidence ID '{ev_id}' in action '{action}' is not visible in submission evidence context.")
                        
                target_belief_id = item.get("target_belief_id")
                target_condition_id = item.get("target_condition_id")
                replacement_belief_id = item.get("replacement_belief_id")
                rationale = item.get("rationale", "Propose by Stage C Prompt Policy")
                
                # Semantic field validations
                if action == "NO_REVISION":
                    if target_belief_id or target_condition_id or replacement_belief_id:
                        raise ValueError("NO_REVISION action must not target any belief or condition.")
                elif action == "SUPERSEDES":
                    if not target_belief_id or not replacement_belief_id:
                        raise ValueError("SUPERSEDES action requires both target_belief_id and replacement_belief_id.")
                    if target_belief_id not in valid_candidate_belief_ids:
                        raise ValueError(f"SUPERSEDES target_belief_id '{target_belief_id}' is not in candidate beliefs.")
                    if replacement_belief_id not in valid_replacement_belief_ids:
                        raise ValueError(f"SUPERSEDES replacement_belief_id '{replacement_belief_id}' is not in candidate replacement beliefs.")
                elif action in ("BLOCKS", "RELEASES"):
                    if not target_condition_id:
                        raise ValueError(f"{action} action requires target_condition_id.")
                    if target_condition_id not in valid_condition_ids:
                        raise ValueError(f"{action} target_condition_id '{target_condition_id}' is not in candidate conditions.")
                elif action in ("UNCERTAIN", "REAFFIRMS"):
                    if not target_belief_id:
                        raise ValueError(f"{action} action requires target_belief_id.")
                    if target_belief_id not in valid_candidate_belief_ids:
                        raise ValueError(f"{action} target_belief_id '{target_belief_id}' is not in candidate beliefs.")
                
                from experiments.multiagent.contracts import TypedRevisionTarget
                target = TypedRevisionTarget(
                    submission_id=submission.submission_id,
                    action_type=action,
                    target_belief_id=target_belief_id,
                    target_condition_id=target_condition_id,
                    replacement_belief_id=replacement_belief_id,
                    rationale=rationale,
                    evidence_ids=tuple(evidence_ids),
                )
                parsed_targets.append(target)

            # Deduplication and conflict check
            seen_actions = set()
            blocked_conditions = set()
            released_conditions = set()
            
            for t in parsed_targets:
                t_key = (
                    t.action_type,
                    t.target_belief_id,
                    t.target_condition_id,
                    t.replacement_belief_id,
                    tuple(t.evidence_ids)
                )
                if t_key in seen_actions:
                    has_duplicates = True
                    continue
                seen_actions.add(t_key)
                
                if t.action_type == "BLOCKS":
                    if t.target_condition_id in released_conditions:
                        raise ValueError(f"Conflict: Condition '{t.target_condition_id}' is both BLOCKED and RELEASED.")
                    blocked_conditions.add(t.target_condition_id)
                elif t.action_type == "RELEASES":
                    if t.target_condition_id in blocked_conditions:
                        raise ValueError(f"Conflict: Condition '{t.target_condition_id}' is both BLOCKED and RELEASED.")
                    released_conditions.add(t.target_condition_id)
                    
                deduplicated_parsed.append(t)

            has_no_revision = any(t.action_type == "NO_REVISION" for t in deduplicated_parsed)
            if has_no_revision and len(deduplicated_parsed) > 1:
                raise ValueError("NO_REVISION cannot be combined with other revision actions in a single submission.")

            # Create DPA edges for canonical processing (exclude NO_REVISION)
            for idx, t in enumerate(deduplicated_parsed):
                if t.action_type == "NO_REVISION":
                    continue
                    
                target_kind = "belief" if t.target_belief_id else "condition"
                target_id = t.target_belief_id or t.target_condition_id
                
                # Build canonical DPA edge
                edge_id = f"edge_policy_{example_id}_{idx}"
                edge = EvidenceEdge(
                    edge_id=edge_id,
                    edge_type=EvidenceEdgeType(t.action_type) if hasattr(EvidenceEdgeType, t.action_type) else t.action_type,
                    evidence_id=str(t.evidence_ids[0]),
                    target_kind=target_kind,
                    target_id=target_id,
                    verifier="stagec_policy",
                    replacement_belief_id=t.replacement_belief_id,
                    rationale=t.rationale,
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
            submission_id=submission.submission_id,
            policy_variant=self.policy_variant,
            proposal_batches=proposal_batches,
            parsing_valid=parsing_valid,
            errors=tuple(errors),
            parsed_actions=tuple(deduplicated_parsed),
            metadata={"has_duplicates": has_duplicates},
        )
