from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
from retracemem.authorization import EvidenceProposalBatch
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    ProposalPolicyOutput,
    TypedRevisionTarget,
    ApprovedRevisionExemplar,
    TypedRevisionProposer,
)
from experiments.multiagent.export_stagec_sft import format_user_prompt

CANONICAL_ACTIONS = {"SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"}


class PromptTypedRevisionPolicy:
    policy_variant = "prompt"

    def __init__(self, allowed_actions: tuple[str, ...] | None = None, diagnostic_mode: bool = False) -> None:
        self.allowed_actions = allowed_actions if allowed_actions is not None else (
            "SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"
        )
        self.diagnostic_mode = diagnostic_mode

    def build_system_prompt(self) -> str:
        system_prompt = (
            "You are the ReTrace Stage C revision policy. Your task is to propose explicit "
            "typed revision proposals for multi-agent shared-memory updates.\n\n"
            "Propose revision actions only from this canonical vocabulary:\n"
            "- SUPERSEDES (requires target_belief_id and replacement_belief_id)\n"
            "- BLOCKS (requires target_condition_id)\n"
            "- RELEASES (requires target_condition_id)\n"
            "- UNCERTAIN (requires target_belief_id)\n"
            "- REAFFIRMS (requires target_belief_id)\n"
            "- NO_REVISION (requires all target/replacement IDs as null)\n\n"
            "### Silent Decision Checklist\n"
            "Before producing JSON, you must silently evaluate the following questions based on the evidence:\n"
            "1. Does the new evidence update, replace, or contradict a prior belief?\n"
            "2. Is there a candidate replacement belief directly supported by the new evidence?\n"
            "3. Does the new evidence invalidate a listed condition?\n"
            "4. Does it restore a previously blocked condition?\n"
            "5. Does it create unresolved conflict without a safe replacement?\n"
            "6. Does it independently confirm an existing belief?\n"
            "7. Is the new evidence duplicate or irrelevant?\n\n"
            "### Action Trigger Guidance\n"
            "- Emit SUPERSEDES if a candidate replacement belief is a newer/updated version of an existing candidate belief and is supported by the new evidence.\n"
            "- Emit BLOCKS if a listed condition required by a belief is contradicted or invalidated by the new evidence.\n"
            "- Emit RELEASES if a listed condition is restored or cleared by the new evidence.\n"
            "- Emit UNCERTAIN if an unresolved conflict or doubt is created for a belief, and no safe replacement belief is available.\n"
            "- Emit REAFFIRMS if the new evidence independently confirms or supports an existing belief.\n"
            "- Emit NO_REVISION only if the new evidence is duplicate, irrelevant, or does not warrant any status change.\n\n"
            "### Constraints\n"
            "- BLOCKS and RELEASES must target only listed condition IDs (target_condition_id).\n"
            "- SUPERSEDES, UNCERTAIN, and REAFFIRMS must target only listed belief IDs (target_belief_id).\n"
            "- NO_REVISION must specify target_belief_id, target_condition_id, and replacement_belief_id as null, and include the new_evidence_id in evidence_ids.\n"
        )
        if self.diagnostic_mode:
            system_prompt += (
                "- Do not output any thinking process, chain-of-thought, or markdown formatting.\n"
                '- You must return ONLY a strict JSON object with two keys: "decision_audit" and "actions".\n'
                '  "decision_audit" must follow this schema:\n'
                "  {\n"
                '    "new_evidence_role": "brief description of role of new evidence",\n'
                '    "prior_replacement_relation": "brief relation between candidate beliefs and replacements",\n'
                '    "condition_effect": "brief effect on listed condition anchors",\n'
                '    "conflict_state": "brief assessment of memory conflict",\n'
                '    "selected_action_types": ["action_type1", ...],\n'
                '    "rejected_action_types": {"action_type2": "rejection reason", ...}\n'
                "  }\n"
                '  "actions" must be a list of action objects as defined in the examples.\n\n'
                "### Output Schema (Diagnostic Mode)\n"
                "{\n"
                '  "decision_audit": {\n'
                '    "new_evidence_role": "...",\n'
                '    "prior_replacement_relation": "...",\n'
                '    "condition_effect": "...",\n'
                '    "conflict_state": "...",\n'
                '    "selected_action_types": [...],\n'
                '    "rejected_action_types": {...}\n'
                "  },\n"
                '  "actions": [\n'
                "    ... (action objects) ...\n"
                "  ]\n"
                "}\n"
            )
        else:
            system_prompt += (
                "- Do not output any thinking process, chain-of-thought, markdown formatting, or final DPA status. Return ONLY a strict JSON array of objects.\n\n"
                "### Examples\n"
                "Example 1: SUPERSEDES\n"
                '[\n  {\n    "action_type": "SUPERSEDES",\n    "target_belief_id": "b_old",\n'
                '    "target_condition_id": null,\n    "replacement_belief_id": "b_new",\n'
                '    "rationale": "New evidence replaces the old address.",\n'
                '    "evidence_ids": ["ev_new"]\n  }\n]\n\n'
                "Example 2: BLOCKS\n"
                '[\n  {\n    "action_type": "BLOCKS",\n    "target_belief_id": null,\n'
                '    "target_condition_id": "c_limit",\n    "replacement_belief_id": null,\n'
                '    "rationale": "Injury invalidates active physical status.",\n'
                '    "evidence_ids": ["ev_new"]\n  }\n]\n\n'
                "Example 3: SUPERSEDES + BLOCKS (compositional)\n"
                '[\n  {\n    "action_type": "SUPERSEDES",\n    "target_belief_id": "b_1",\n'
                '    "target_condition_id": null,\n    "replacement_belief_id": "b_2",\n'
                '    "rationale": "New schedule supersedes previous.",\n'
                '    "evidence_ids": ["ev_new"]\n  },\n  {\n    "action_type": "BLOCKS",\n'
                '    "target_belief_id": null,\n    "target_condition_id": "c_2",\n'
                '    "replacement_belief_id": null,\n    "rationale": "Schedule conflict blocks the location condition.",\n'
                '    "evidence_ids": ["ev_new"]\n  }\n]\n\n'
                "Example 4: UNCERTAIN\n"
                '[\n  {\n    "action_type": "UNCERTAIN",\n    "target_belief_id": "b_1",\n'
                '    "target_condition_id": null,\n    "replacement_belief_id": null,\n'
                '    "rationale": "Evidence indicates the status might have changed but details are unclear.",\n'
                '    "evidence_ids": ["ev_new"]\n  }\n]\n\n'
                "Example 5: REAFFIRMS\n"
                '[\n  {\n    "action_type": "REAFFIRMS",\n    "target_belief_id": "b_1",\n'
                '    "target_condition_id": null,\n    "replacement_belief_id": null,\n'
                '    "rationale": "Evidence confirms the user is still at the same office.",\n'
                '    "evidence_ids": ["ev_new"]\n  }\n]\n\n'
                "Example 6: NO_REVISION\n"
                '[\n  {\n    "action_type": "NO_REVISION",\n    "target_belief_id": null,\n'
                '    "target_condition_id": null,\n    "replacement_belief_id": null,\n'
                '    "rationale": "The evidence is duplicate or does not change any belief.",\n'
                '    "evidence_ids": ["ev_new"]\n  }\n]'
            )
        return system_prompt

    def build_messages(
        self,
        submission: FixedCandidateSubmission,
    ) -> Tuple[Dict[str, str], ...]:
        """Construct the prompt messages for the policy using method-visible context."""
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

        # Build deterministic candidate contrast block
        ne_node = next((e for e in submission.evidence_context if e.evidence_id == submission.new_evidence_id), None)
        new_evidence_text = f'"{ne_node.text}" (ID: {ne_node.evidence_id})' if ne_node else f"(ID: {submission.new_evidence_id})"

        beliefs_str = "\n".join(f"  - {b.belief_id}: \"{b.proposition}\"" for b in submission.candidate_beliefs) or "  - (none)"
        replacements_str = "\n".join(f"  - {b.belief_id}: \"{b.proposition}\"" for b in submission.candidate_replacement_beliefs) or "  - (none)"
        
        cond_parts = []
        for bid, conds in submission.candidate_conditions_by_belief:
            for c in conds:
                cond_parts.append(f"  - [{bid}] {c.condition_id}: \"{c.text}\"")
        conditions_str = "\n".join(cond_parts) or "  - (none)"
        
        contrast_block = (
            "\n\n### Deterministic Candidate Contrast Block\n"
            f"- New Evidence to Evaluate: {new_evidence_text}\n\n"
            "- Prior Candidate Beliefs:\n"
            f"{beliefs_str}\n\n"
            "- Candidate Replacement Beliefs:\n"
            f"{replacements_str}\n\n"
            "- Condition Anchors:\n"
            f"{conditions_str}"
        )
        user_content = user_content + contrast_block
        
        return (
            {"role": "system", "content": self.build_system_prompt()},
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
        
        from retracemem.multiagent.utils import extract_first_json_array, extract_json_object

        valid_evidence_ids = {ev.evidence_id for ev in submission.evidence_context} | {submission.new_evidence_id}
        valid_candidate_belief_ids = {b.belief_id for b in submission.candidate_beliefs}
        valid_replacement_belief_ids = {b.belief_id for b in submission.candidate_replacement_beliefs}
        valid_condition_ids = set()
        for _, conds in submission.candidate_conditions_by_belief:
            for c in conds:
                valid_condition_ids.add(c.condition_id)

        deduplicated_parsed: List[TypedRevisionTarget] = []
        has_duplicates = False
        decision_audit = None

        try:
            if self.diagnostic_mode:
                try:
                    obj = extract_json_object(response_text)
                except Exception as e:
                    raise ValueError(f"Diagnostic mode requires a JSON object with 'decision_audit' and 'actions'. Error: {e}")
                if not isinstance(obj, dict):
                    raise ValueError("Diagnostic mode output must be a JSON object.")
                if "actions" not in obj:
                    raise ValueError("Diagnostic mode output must contain 'actions' key.")
                if "decision_audit" not in obj:
                    raise ValueError("Diagnostic mode output must contain 'decision_audit' key.")
                decision_audit = obj["decision_audit"]
                parsed = obj["actions"]
            else:
                parsed = extract_first_json_array(response_text)

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
                    
                if action not in self.allowed_actions:
                    raise ValueError(f"Action '{action}' is not allowed in current vocabulary configuration.")
                    
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

            for idx, t in enumerate(deduplicated_parsed):
                if t.action_type == "NO_REVISION":
                    continue
                    
                target_kind = "belief" if t.target_belief_id else "condition"
                target_id = t.target_belief_id or t.target_condition_id
                
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
            
        meta = {"has_duplicates": has_duplicates}
        if decision_audit is not None:
            meta["decision_audit"] = decision_audit

        return ProposalPolicyOutput(
            example_id=example_id,
            submission_id=submission.submission_id,
            policy_variant=self.policy_variant,
            proposal_batches=proposal_batches,
            parsing_valid=parsing_valid,
            errors=tuple(errors),
            parsed_actions=tuple(deduplicated_parsed),
            metadata=meta,
        )


class ClosedAPIZeroShotProposer(TypedRevisionProposer):
    proposer_name = "closed_api_zero_shot"
    policy_variant = "zero_shot"

    def __init__(
        self,
        provider_kind: str | None = None,
        model_id: str | None = None,
        client: Any = None,
        allowed_actions: tuple[str, ...] | None = None,
        diagnostic_mode: bool = False,
    ) -> None:
        self.provider_kind = provider_kind
        self.model_id = model_id
        self.client = client
        self.allowed_actions = allowed_actions or ("SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION")
        self.diagnostic_mode = diagnostic_mode
        self._policy = PromptTypedRevisionPolicy(
            allowed_actions=self.allowed_actions,
            diagnostic_mode=self.diagnostic_mode,
        )

    def propose(
        self,
        submission: FixedCandidateSubmission,
        *,
        exemplars: tuple[ApprovedRevisionExemplar, ...] = (),
    ) -> ProposalPolicyOutput:
        # Zero-shot ignores exemplars
        messages = self._policy.build_messages(submission)
        system_text = messages[0]["content"]
        user_text = messages[1]["content"]
        full_prompt = f"System:\n{system_text}\n\nUser:\n{user_text}"

        is_live = self.provider_kind is not None and self.provider_kind != "mock"
        if is_live:
            if self.client is None or self.model_id is None:
                raise ValueError(
                    f"Live mode requires client and model_id to be specified. "
                    f"Got client={self.client}, model_id={self.model_id}"
                )
            trace = self.client.generate(
                prompt=full_prompt,
                model_id=self.model_id,
                provider=self.provider_kind,
            )
            response_text = trace.response or "[]"
        else:
            if self._policy.diagnostic_mode:
                response_text = json.dumps({
                    "decision_audit": {
                        "new_evidence_role": "Mock new evidence role",
                        "prior_replacement_relation": "Mock relation",
                        "condition_effect": "Mock condition effect",
                        "conflict_state": "Mock conflict state",
                        "selected_action_types": ["NO_REVISION"],
                        "rejected_action_types": {}
                    },
                    "actions": [{
                        "action_type": "NO_REVISION",
                        "target_belief_id": None,
                        "target_condition_id": None,
                        "replacement_belief_id": None,
                        "rationale": "Mock default action",
                        "evidence_ids": [submission.new_evidence_id]
                    }]
                })
            else:
                response_text = json.dumps([{
                    "action_type": "NO_REVISION",
                    "target_belief_id": None,
                    "target_condition_id": None,
                    "replacement_belief_id": None,
                    "rationale": "Mock default action",
                    "evidence_ids": [submission.new_evidence_id]
                }])

        out = self._policy.parse_response(
            response_text,
            example_id=f"ex_{submission.submission_id}",
            submission=submission,
        )
        from dataclasses import replace
        new_metadata = dict(out.metadata)
        new_metadata["prompt"] = full_prompt
        new_metadata["raw_response"] = response_text
        return replace(out, metadata=new_metadata)


class ClosedAPIZeroShotConstrainedProposer(TypedRevisionProposer):
    proposer_name = "closed_api_zero_shot_constrained"
    policy_variant = "zero_shot_constrained"

    def __init__(
        self,
        provider_kind: str | None = None,
        model_id: str | None = None,
        client: Any = None,
        allowed_actions: tuple[str, ...] | None = None,
        diagnostic_mode: bool = False,
    ) -> None:
        self.provider_kind = provider_kind
        self.model_id = model_id
        self.client = client
        self.allowed_actions = allowed_actions or ("SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION")
        self.diagnostic_mode = diagnostic_mode

    def build_system_prompt(self, candidates: list[dict[str, Any]]) -> str:
        prompt = (
            "You are the ReTrace Stage C revision policy with constrained action affordances.\n"
            "Your task is to select one or more candidate action IDs from the provided candidate list.\n\n"
            "Constraints:\n"
            "- You MUST choose only candidate IDs from the list provided in the user message.\n"
            "- You MUST NOT invent any new IDs.\n"
            "- If no revision is warranted, select ONLY 'act_no_revision'.\n"
            "- You cannot combine 'act_no_revision' with any other action ID.\n"
            "- If you select multiple candidate actions, they must be compatible and not duplicate each other.\n\n"
        )
        if self.diagnostic_mode:
            prompt += (
                "### Diagnostic Mode Output Format\n"
                "You must return ONLY a strict JSON object (no markdown, no extra explanation) with the following structure:\n"
                "{\n"
                '  "decision_audit": {\n'
                '    "new_evidence_role": "brief description of role of new evidence",\n'
                '    "prior_replacement_relation": "brief relation between candidate beliefs and replacements",\n'
                '    "condition_effect": "brief effect on listed condition anchors",\n'
                '    "conflict_state": "brief assessment of memory conflict",\n'
                '    "selected_action_types": ["action_type1", ...],\n'
                '    "rejected_action_types": {"action_type2": "rejection reason", ...}\n'
                "  },\n"
                '  "selected_candidate_action_ids": ["act_id_1", ...],\n'
                '  "rejection_reasons": {\n'
                '    "act_id_2": "short reason why this was rejected",\n'
                "    ...\n"
                "  }\n"
                "}\n"
            )
        else:
            prompt += (
                "### Output Format\n"
                "You must return ONLY a strict JSON object (no markdown, no extra explanation) with the following structure:\n"
                "{\n"
                '  "selected_candidate_action_ids": ["act_id_1", ...],\n'
                '  "rejection_reasons": {\n'
                '    "act_id_2": "short reason why this was rejected",\n'
                "    ...\n"
                "  }\n"
                "}\n"
            )
        return prompt

    def build_user_prompt(
        self,
        submission: FixedCandidateSubmission,
        candidates: list[dict[str, Any]],
    ) -> str:
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

        ne_node = next((e for e in submission.evidence_context if e.evidence_id == submission.new_evidence_id), None)
        new_evidence_text = f'"{ne_node.text}" (ID: {ne_node.evidence_id})' if ne_node else f"(ID: {submission.new_evidence_id})"

        beliefs_str = "\n".join(f"  - {b.belief_id}: \"{b.proposition}\"" for b in submission.candidate_beliefs) or "  - (none)"
        replacements_str = "\n".join(f"  - {b.belief_id}: \"{b.proposition}\"" for b in submission.candidate_replacement_beliefs) or "  - (none)"
        
        cond_parts = []
        for bid, conds in submission.candidate_conditions_by_belief:
            for c in conds:
                cond_parts.append(f"  - [{bid}] {c.condition_id}: \"{c.text}\"")
        conditions_str = "\n".join(cond_parts) or "  - (none)"
        
        contrast_block = (
            "\n\n### Deterministic Candidate Contrast Block\n"
            f"- New Evidence to Evaluate: {new_evidence_text}\n\n"
            "- Prior Candidate Beliefs:\n"
            f"{beliefs_str}\n\n"
            "- Candidate Replacement Beliefs:\n"
            f"{replacements_str}\n\n"
            "- Condition Anchors:\n"
            f"{conditions_str}"
        )
        user_content = user_content + contrast_block

        cand_parts = []
        for c in candidates:
            cand_parts.append(
                f"  - candidate_action_id: {c['candidate_action_id']}\n"
                f"    action_type: {c['action_type']}\n"
                f"    target_belief_id: {c['target_belief_id']}\n"
                f"    target_condition_id: {c['target_condition_id']}\n"
                f"    replacement_belief_id: {c['replacement_belief_id']}\n"
                f"    why_candidate: {c['why_candidate']}"
            )
        candidates_str = "\n\n".join(cand_parts)

        user_content += (
            "\n\n### Available Candidate Action Affordances\n"
            "Choose only from the following candidates:\n\n"
            f"{candidates_str}"
        )
        return user_content

    def propose(
        self,
        submission: FixedCandidateSubmission,
        *,
        exemplars: tuple[ApprovedRevisionExemplar, ...] = (),
    ) -> ProposalPolicyOutput:
        from retracemem.multiagent.utils import build_candidate_actions
        candidates = build_candidate_actions(submission)

        system_prompt = self.build_system_prompt(candidates)
        user_prompt = self.build_user_prompt(submission, candidates)
        full_prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"

        is_live = self.provider_kind is not None and self.provider_kind != "mock"
        if is_live:
            if self.client is None or self.model_id is None:
                raise ValueError(
                    f"Live mode requires client and model_id to be specified. "
                    f"Got client={self.client}, model_id={self.model_id}"
                )
            trace = self.client.generate(
                prompt=full_prompt,
                model_id=self.model_id,
                provider=self.provider_kind,
            )
            response_text = trace.response or "{}"
        else:
            if self.diagnostic_mode:
                response_text = json.dumps({
                    "decision_audit": {
                        "new_evidence_role": "Mock new evidence role",
                        "prior_replacement_relation": "Mock relation",
                        "condition_effect": "Mock condition effect",
                        "conflict_state": "Mock conflict state",
                        "selected_action_types": ["NO_REVISION"],
                        "rejected_action_types": {}
                    },
                    "selected_candidate_action_ids": ["act_no_revision"],
                    "rejection_reasons": {
                        c["candidate_action_id"]: "Mock reject"
                        for c in candidates if c["candidate_action_id"] != "act_no_revision"
                    }
                })
            else:
                response_text = json.dumps({
                    "selected_candidate_action_ids": ["act_no_revision"],
                    "rejection_reasons": {
                        c["candidate_action_id"]: "Mock reject"
                        for c in candidates if c["candidate_action_id"] != "act_no_revision"
                    }
                })

        out = self.parse_response(
            response_text,
            example_id=f"ex_{submission.submission_id}",
            submission=submission,
            candidates=candidates,
        )
        from dataclasses import replace
        new_metadata = dict(out.metadata)
        new_metadata["prompt"] = full_prompt
        new_metadata["raw_response"] = response_text
        return replace(out, metadata=new_metadata)

    def parse_response(
        self,
        response_text: str,
        *,
        example_id: str,
        submission: FixedCandidateSubmission,
        candidates: list[dict[str, Any]],
    ) -> ProposalPolicyOutput:
        from retracemem.multiagent.utils import extract_json_object
        from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
        from retracemem.authorization import EvidenceProposalBatch

        errors: list[str] = []
        parsing_valid = True
        edges: list[EvidenceEdge] = []
        parsed_targets: list[TypedRevisionTarget] = []

        candidate_map = {c["candidate_action_id"]: c for c in candidates}
        decision_audit = None

        try:
            obj = extract_json_object(response_text)
            if not isinstance(obj, dict):
                raise ValueError("Response must be a JSON object.")

            if "decision_audit" in obj:
                decision_audit = obj["decision_audit"]

            if "selected_candidate_action_ids" not in obj:
                raise ValueError("Response missing required key 'selected_candidate_action_ids'.")

            selected_ids = obj["selected_candidate_action_ids"]
            if not isinstance(selected_ids, list):
                raise ValueError("'selected_candidate_action_ids' must be an array.")

            if not selected_ids:
                raise ValueError("'selected_candidate_action_ids' must not be empty.")

            # Verify no invented IDs
            for cid in selected_ids:
                if cid not in candidate_map:
                    raise ValueError(f"Invented candidate action ID '{cid}' is not allowed.")

            # Verify NO_REVISION combined check
            if "act_no_revision" in selected_ids and len(selected_ids) > 1:
                raise ValueError("NO_REVISION cannot be combined with other revision actions in a single submission.")

            # Map back to typed actions
            for idx, cid in enumerate(selected_ids):
                c = candidate_map[cid]
                action = c["action_type"]

                # Check vocabulary restriction
                if action not in self.allowed_actions:
                    raise ValueError(f"Action '{action}' is not allowed in current vocabulary configuration.")

                evidence_ids = c["evidence_ids"]
                target_belief_id = c["target_belief_id"]
                target_condition_id = c["target_condition_id"]
                replacement_belief_id = c["replacement_belief_id"]
                rationale = c["why_candidate"]

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

            # Build DPA edges
            for idx, t in enumerate(parsed_targets):
                if t.action_type == "NO_REVISION":
                    continue

                target_kind = "belief" if t.target_belief_id else "condition"
                target_id = t.target_belief_id or t.target_condition_id

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
                    metadata={"parser": "ClosedAPIZeroShotConstrainedProposer"},
                ),
            )

        meta = {"has_duplicates": False}
        if decision_audit is not None:
            meta["decision_audit"] = decision_audit

        return ProposalPolicyOutput(
            example_id=example_id,
            submission_id=submission.submission_id,
            policy_variant=self.policy_variant,
            proposal_batches=proposal_batches,
            parsing_valid=parsing_valid,
            errors=tuple(errors),
            parsed_actions=tuple(parsed_targets),
            metadata=meta,
        )


class ClosedAPIICLProposer(TypedRevisionProposer):
    proposer_name = "closed_api_icl"
    policy_variant = "icl"

    def __init__(
        self,
        provider_kind: str | None = None,
        model_id: str | None = None,
        client: Any = None,
        allowed_actions: tuple[str, ...] | None = None,
        top_k: int = 1,
    ) -> None:
        self.provider_kind = provider_kind
        self.model_id = model_id
        self.client = client
        self.allowed_actions = allowed_actions or ("SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION")
        self.top_k = top_k
        self._policy = PromptTypedRevisionPolicy(allowed_actions=self.allowed_actions)

    def retrieve_exemplars(
        self,
        submission: FixedCandidateSubmission,
        exemplars: tuple[ApprovedRevisionExemplar, ...],
    ) -> tuple[ApprovedRevisionExemplar, ...]:
        if not exemplars:
            return ()
        query_tokens = set(submission.query.lower().split())
        scored: list[tuple[float, ApprovedRevisionExemplar]] = []
        for ex in exemplars:
            ex_tokens = set(ex.method_visible_input.query.lower().split())
            intersection = query_tokens & ex_tokens
            union = query_tokens | ex_tokens
            score = len(intersection) / len(union) if union else 0.0
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(ex for score, ex in scored[:self.top_k])

    def propose(
        self,
        submission: FixedCandidateSubmission,
        *,
        exemplars: tuple[ApprovedRevisionExemplar, ...] = (),
    ) -> ProposalPolicyOutput:
        selected_exs = self.retrieve_exemplars(submission, exemplars)
        messages = self._policy.build_messages(submission)
        system_text = messages[0]["content"]
        user_text = messages[1]["content"]

        icl_context = ""
        for ex in selected_exs:
            from experiments.multiagent.contracts import StageCTrainingExample
            fake_ex = StageCTrainingExample(
                example_id=ex.exemplar_id,
                episode_id=ex.source_episode_id,
                submission_id=ex.method_visible_input.submission_id,
                method_visible_input=ex.method_visible_input,
                targets=(),
                split="development_only",
                domain=ex.domain,
                failure_type=ex.failure_type,
                label_source="temporary",
            )
            targets_list = []
            for t in ex.approved_typed_actions:
                targets_list.append({
                    "action_type": t.action_type,
                    "target_belief_id": t.target_belief_id,
                    "target_condition_id": t.target_condition_id,
                    "replacement_belief_id": t.replacement_belief_id,
                    "rationale": t.rationale,
                    "evidence_ids": list(t.evidence_ids),
                })
            targets_json = json.dumps(targets_list, indent=2)
            icl_context += f"Example input:\n{format_user_prompt(fake_ex)}\n\nExample Output:\n{targets_json}\n\n---\n\n"

        if icl_context:
            system_text += f"\n\nHere are some examples of expected revision decisions:\n\n{icl_context}"

        full_prompt = f"System:\n{system_text}\n\nUser:\n{user_text}"
        is_live = self.provider_kind is not None and self.provider_kind != "mock"
        if is_live:
            if self.client is None or self.model_id is None:
                raise ValueError(
                    f"Live mode requires client and model_id to be specified. "
                    f"Got client={self.client}, model_id={self.model_id}"
                )
            trace = self.client.generate(
                prompt=full_prompt,
                model_id=self.model_id,
                provider=self.provider_kind,
            )
            response_text = trace.response or "[]"
        else:
            response_text = json.dumps([{
                "action_type": "NO_REVISION",
                "target_belief_id": None,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "rationale": "Mock default action",
                "evidence_ids": [submission.new_evidence_id]
            }])

        out = self._policy.parse_response(
            response_text,
            example_id=f"ex_{submission.submission_id}",
            submission=submission,
        )
        from dataclasses import replace
        new_metadata = dict(out.metadata)
        new_metadata["prompt"] = full_prompt
        new_metadata["raw_response"] = response_text
        return replace(out, metadata=new_metadata)


class OpenModelPromptProposer(TypedRevisionProposer):
    proposer_name = "open_model_prompt"
    policy_variant = "open_prompt"

    def __init__(
        self,
        provider_kind: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.provider_kind = provider_kind
        self.model_id = model_id

    def propose(
        self,
        submission: FixedCandidateSubmission,
        *,
        exemplars: tuple[ApprovedRevisionExemplar, ...] = (),
    ) -> ProposalPolicyOutput:
        return ProposalPolicyOutput(
            example_id=f"ex_{submission.submission_id}",
            submission_id=submission.submission_id,
            policy_variant=self.policy_variant,
            proposal_batches=(),
            parsing_valid=True,
            errors=(),
            parsed_actions=(),
            metadata={"placeholder": True},
        )


class OpenModelLoRAProposer(TypedRevisionProposer):
    proposer_name = "open_model_lora"
    policy_variant = "open_lora"

    def __init__(
        self,
        provider_kind: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.provider_kind = provider_kind
        self.model_id = model_id

    def propose(
        self,
        submission: FixedCandidateSubmission,
        *,
        exemplars: tuple[ApprovedRevisionExemplar, ...] = (),
    ) -> ProposalPolicyOutput:
        return ProposalPolicyOutput(
            example_id=f"ex_{submission.submission_id}",
            submission_id=submission.submission_id,
            policy_variant=self.policy_variant,
            proposal_batches=(),
            parsing_valid=True,
            errors=(),
            parsed_actions=(),
            metadata={"placeholder": True},
        )
