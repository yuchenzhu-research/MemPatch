"""Stage C adapter / local / SFT-style typed revision proposers.

These proposers complete the Stage C (`ReTrace-Learn`) family by
turning *already-decoded* model text into a `ProposalPolicyOutput` that flows
through the exact same RevisionGate -> deterministic DPA -> commit path used by
Stage A / Stage B.

Design intent (see AGENTS.md "ReTrace-Learn Paper Training Boundary"):

* The expensive / non-deterministic part (decoding a local adapter or SFT
  checkpoint, e.g. via MLX or transformers + LoRA) happens *offline* and is
  dumped to per-submission text files.
* This module performs the deterministic, API-free evaluation step: it reads
  the decoded text, applies the canonical constrained post-validation parser
  (`PromptTypedRevisionPolicy.parse_response`), and emits typed proposal
  batches. No external network call happens here.

The proposers fail closed: a missing generation or an unparseable / ungrounded
output yields ``parsing_valid=False`` with an explicit failure reason rather
than raising, so the surrounding runner can still record the case.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

from retracemem.evaluation.multiagent.contracts import (
    ApprovedRevisionExemplar,
    FixedCandidateSubmission,
    ProposalPolicyOutput,
    TypedRevisionProposer,
)
from retracemem.proposers.typed_revision_policy import PromptTypedRevisionPolicy

CANONICAL_ACTIONS: tuple[str, ...] = (
    "SUPERSEDES",
    "BLOCKS",
    "RELEASES",
    "UNCERTAIN",
    "REAFFIRMS",
    "NO_REVISION",
)

# A generation source maps a submission to its already-decoded model text, or
# ``None`` when no generation is available for that submission.
GenerationSource = Callable[[FixedCandidateSubmission], "str | None"]


class DirectoryGenerationSource:
    """Read decoded Stage C generations from a directory of text files.

    Files are looked up by ``{submission_id}{suffix}`` (default ``.txt``). This
    matches the offline decoding workflow where an adapter / SFT checkpoint
    writes one decoded completion per submission.
    """

    def __init__(self, generations_dir: str | Path, *, suffix: str = ".txt") -> None:
        self.generations_dir = Path(generations_dir)
        self.suffix = suffix

    def __call__(self, submission: FixedCandidateSubmission) -> str | None:
        path = self.generations_dir / f"{submission.submission_id}{self.suffix}"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")


class MappingGenerationSource:
    """Serve decoded generations from an in-memory ``{submission_id: text}`` map.

    Useful for tests and the offline smoke path (no files, no API)."""

    def __init__(self, generations_by_submission: dict[str, str]) -> None:
        self.generations_by_submission = dict(generations_by_submission)

    def __call__(self, submission: FixedCandidateSubmission) -> str | None:
        return self.generations_by_submission.get(submission.submission_id)


def no_revision_generation(submission: FixedCandidateSubmission) -> str:
    """Deterministic, schema-valid NO_REVISION completion for a submission.

    This is the offline smoke generator: it produces well-formed typed-action
    JSON that passes constrained post-validation without any external model.
    It does NOT consult gold labels, so it stands in for a (trivial) proposer
    policy rather than leaking evaluation targets.
    """
    return json.dumps(
        [
            {
                "action_type": "NO_REVISION",
                "target_belief_id": None,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "rationale": "Offline smoke generation: no revision proposed.",
                "evidence_ids": [submission.new_evidence_id],
            }
        ]
    )


class MockGenerationSource:
    """Offline generation source emitting a valid NO_REVISION completion."""

    def __call__(self, submission: FixedCandidateSubmission) -> str | None:
        return no_revision_generation(submission)


class LocalAdapterReplayProposer(TypedRevisionProposer):
    """Replay pre-decoded adapter / local / SFT generations through the kernel.

    The proposer is intentionally deterministic and API-free: it consumes
    decoded text from a :class:`GenerationSource` and applies the canonical
    constrained post-validation parser to produce typed proposal batches that
    preserve the exact schema expected by the ReTrace pipeline.
    """

    proposer_name = "local_adapter_replay"

    def __init__(
        self,
        generation_source: GenerationSource,
        *,
        policy_variant: str = "adapter_replay",
        backbone_model: str | None = None,
        checkpoint_id: str | None = None,
        provider_kind: str | None = None,
        model_id: str | None = None,
        allowed_actions: tuple[str, ...] | None = None,
        constrained_postvalidation: bool = True,
    ) -> None:
        self.generation_source = generation_source
        self.policy_variant = policy_variant
        self.backbone_model = backbone_model
        self.checkpoint_id = checkpoint_id
        self.provider_kind = provider_kind
        self.model_id = model_id or backbone_model
        self.allowed_actions = allowed_actions or CANONICAL_ACTIONS
        self.constrained_postvalidation = constrained_postvalidation
        self._policy = PromptTypedRevisionPolicy(allowed_actions=self.allowed_actions)

    def _reference_prompt(self, submission: FixedCandidateSubmission) -> str:
        """The canonical Stage C policy prompt that decoding should have used.

        Recorded for reproducibility/auditing; not used to call any model here.
        """
        messages = self._policy.build_messages(submission)
        return f"System:\n{messages[0]['content']}\n\nUser:\n{messages[1]['content']}"

    def propose(
        self,
        submission: FixedCandidateSubmission,
        *,
        exemplars: tuple[ApprovedRevisionExemplar, ...] = (),
    ) -> ProposalPolicyOutput:
        reference_prompt = self._reference_prompt(submission)
        raw_text = self.generation_source(submission)

        if raw_text is None:
            failure = (
                f"No decoded generation found for submission "
                f"'{submission.submission_id}'."
            )
            return ProposalPolicyOutput(
                example_id=f"ex_{submission.submission_id}",
                submission_id=submission.submission_id,
                policy_variant=self.policy_variant,
                proposal_batches=(),
                backbone_model=self.backbone_model,
                checkpoint_id=self.checkpoint_id,
                parsing_valid=False,
                errors=(failure,),
                parsed_actions=(),
                metadata={
                    "prompt": reference_prompt,
                    "raw_response": "",
                    "first_pass_valid_json": False,
                    "first_pass_parser_error": failure,
                    "repair_triggered": False,
                    "repair_success": False,
                    "failure_reason": "missing_generation",
                    "constrained_postvalidation": self.constrained_postvalidation,
                },
            )

        out = self._policy.parse_response(
            raw_text,
            example_id=f"ex_{submission.submission_id}",
            submission=submission,
        )

        new_metadata = dict(out.metadata)
        first_pass_error = out.errors[0] if out.errors else None
        new_metadata.update(
            {
                "prompt": reference_prompt,
                "raw_response": raw_text,
                "first_pass_valid_json": out.parsing_valid,
                "first_pass_parser_error": first_pass_error,
                "repair_triggered": False,
                "repair_success": False,
                "constrained_postvalidation": self.constrained_postvalidation,
            }
        )
        if not out.parsing_valid:
            new_metadata["failure_reason"] = "parse_or_validation_error"

        return replace(
            out,
            policy_variant=self.policy_variant,
            backbone_model=self.backbone_model,
            checkpoint_id=self.checkpoint_id,
            metadata=new_metadata,
        )


def build_replay_proposer(
    *,
    generations_dir: str | Path | None = None,
    generations_by_submission: dict[str, str] | None = None,
    mock: bool = False,
    suffix: str = ".txt",
    policy_variant: str = "adapter_replay",
    backbone_model: str | None = None,
    checkpoint_id: str | None = None,
    allowed_actions: tuple[str, ...] | None = None,
    constrained_postvalidation: bool = True,
) -> LocalAdapterReplayProposer:
    """Construct a :class:`LocalAdapterReplayProposer` from one source.

    Exactly one of ``mock``, ``generations_dir`` or ``generations_by_submission``
    selects the decoded-text source.
    """
    if mock:
        source: GenerationSource = MockGenerationSource()
        if policy_variant == "adapter_replay":
            policy_variant = "mock_smoke"
    elif generations_by_submission is not None:
        source = MappingGenerationSource(generations_by_submission)
    elif generations_dir is not None:
        source = DirectoryGenerationSource(generations_dir, suffix=suffix)
    else:
        raise ValueError(
            "build_replay_proposer requires one of: mock=True, "
            "generations_dir=..., or generations_by_submission=..."
        )

    return LocalAdapterReplayProposer(
        source,
        policy_variant=policy_variant,
        backbone_model=backbone_model,
        checkpoint_id=checkpoint_id,
        allowed_actions=allowed_actions,
        constrained_postvalidation=constrained_postvalidation,
    )
