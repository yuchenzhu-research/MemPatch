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

    def __init__(self, allowed_actions: tuple[str, ...] | None = None) -> None:
        self.allowed_actions = allowed_actions if allowed_actions is not None else (
            "SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"
        )

    def build_system_prompt(self) -> str:
        actions_str = ""
        for act in self.allowed_actions:
            if act == "SUPERSEDES":
                actions_str += "- SUPERSEDES (requires replacement_belief_id)\n"
            else:
                actions_str += f"- {act}\n"
        
        system_prompt = (
            "You are the ReTrace Stage C revision policy. Your task is to propose explicit "
            "typed revision proposals for multi-agent shared-memory updates. "
            "Propose revision actions only from this canonical vocabulary:\n"
            f"{actions_str}\n"
            "Constraints:\n"
            "- BLOCKS and RELEASES must target only listed condition IDs (target_condition_id).\n"
            "- SUPERSEDES, UNCERTAIN, and REAFFIRMS must target only listed belief IDs (target_belief_id).\n"
            "- NO_REVISION must specify target_belief_id, target_condition_id, and replacement_belief_id as null, and include the new_evidence_id in evidence_ids.\n\n"
            "Return your response as a strict JSON array of objects with the following fields:\n"
            "- action_type (string)\n"
            "- target_belief_id (string or null)\n"
            "- target_condition_id (string or null)\n"
            "- replacement_belief_id (string or null)\n"
            "- rationale (string)\n"
            "- evidence_ids (array of strings)\n\n"
            "Example format for NO_REVISION:\n"
            '[\n  {\n    "action_type": "NO_REVISION",\n    "target_belief_id": null,\n'
            '    "target_condition_id": null,\n    "replacement_belief_id": null,\n'
            '    "rationale": "No evidence-grounded revision is warranted.",\n'
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
        
        from retracemem.multiagent.utils import extract_first_json_array

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


class ClosedAPIZeroShotProposer(TypedRevisionProposer):
    proposer_name = "closed_api_zero_shot"
    policy_variant = "zero_shot"

    def __init__(
        self,
        provider_kind: str | None = None,
        model_id: str | None = None,
        client: Any = None,
        allowed_actions: tuple[str, ...] | None = None,
    ) -> None:
        self.provider_kind = provider_kind
        self.model_id = model_id
        self.client = client
        self.allowed_actions = allowed_actions or ("SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION")
        self._policy = PromptTypedRevisionPolicy(allowed_actions=self.allowed_actions)

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
