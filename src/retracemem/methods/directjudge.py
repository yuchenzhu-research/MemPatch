"""DirectJudge-LLM: Stage B matched direct-adjudication attribution baseline.

This is a **sibling method path**, not an ``EvidenceEdgeVerifier``. It
directly decides memory usability without DPA or local edge restrictions.

It must NOT import or invoke:
- RevisionGate, DefeatPathAuthorizationAlgorithm
- EvidenceEdge, DependencyEdge, EvidenceEdgeType
- RequirementInducer, EvidenceEdgeVerifier

AB-1A.5 protocol lock:
- Stage B is explicitly told which evidence is new/current (prompt v1).
- metadata fields are non-semantic and MUST NOT be consumed.
- Same fixed SharedCandidateView semantic input does NOT mean identical
  prompts. Stage A and Stage B instantiate different method interfaces.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from retracemem.methods.contracts import (
    ControlledMethodResult,
    DirectUsabilityStatus,
    DirectUsabilityVerdict,
    SharedCandidateView,
)
from retracemem.providers.cached_client import CachedLLMClient

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "prompts", "directjudge")
_PROMPT_FILE = os.path.join(_PROMPT_DIR, "direct_usability_v1.txt")
_PROMPT_VERSION = "direct_usability_v1"
_RESPONSE_SCHEMA_VERSION = "direct_usability_response_v1"
_PARSER_VERSION = "direct_usability_parser_v1"


def _load_prompt_template() -> str:
    path = os.path.normpath(_PROMPT_FILE)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _cost_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compute per-instance cost by subtracting cumulative snapshots."""
    tokens_before = before.get("tokens", {})
    tokens_after = after.get("tokens", {})
    calls_before = before.get("calls", {})
    calls_after = after.get("calls", {})
    return {
        "latency_ms": after.get("latency_ms", 0.0) - before.get("latency_ms", 0.0),
        "tokens": {
            "prompt": tokens_after.get("prompt", 0) - tokens_before.get("prompt", 0),
            "completion": tokens_after.get("completion", 0) - tokens_before.get("completion", 0),
            "total": tokens_after.get("total", 0) - tokens_before.get("total", 0),
        },
        "calls": {
            k: calls_after.get(k, 0) - calls_before.get(k, 0)
            for k in set(calls_after) | set(calls_before)
        },
        "cache_hits": after.get("cache_hits", 0) - before.get("cache_hits", 0),
        "cache_misses": after.get("cache_misses", 0) - before.get("cache_misses", 0),
    }


def _canonicalize_belief_id(returned_id: Any, valid_belief_ids: set[str]) -> str:
    if not isinstance(returned_id, str) or not returned_id:
        raise ValueError(f"Verdict item missing belief_id: {returned_id!r}")
    if returned_id in valid_belief_ids:
        return returned_id
    matches = [
        valid_id for valid_id in valid_belief_ids
        if returned_id.startswith(valid_id) and returned_id != valid_id
    ]
    if len(matches) == 1:
        suffix = returned_id[len(matches[0]):]
        if suffix and (
            suffix[0].isspace()
            or suffix.startswith((':', '-', '—', '–', '"', "'", '(', '[', '{'))
        ):
            return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"DirectJudge returned ambiguous belief_id prefix '{returned_id}'. "
            f"Matches: {sorted(matches)}"
        )
    raise ValueError(
        f"DirectJudge returned verdict for unknown belief_id '{returned_id}'. "
        f"Valid: {valid_belief_ids}"
    )


class DirectJudgeLLM:
    """Stage B: matched same-model direct usability adjudication.

    Consumes ``SharedCandidateView`` and returns per-belief
    ``DirectUsabilityVerdict`` values plus a ``ControlledMethodResult``.
    """

    def __init__(
        self,
        client: CachedLLMClient,
        model_id: str = "gemini-pro",
        provider: str = "google",
        model_revision_or_api_version: str | None = None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.model_revision_or_api_version = model_revision_or_api_version
        self._template = _load_prompt_template()
        self._template_hash = hashlib.sha256(self._template.encode("utf-8")).hexdigest()

    def judge(self, view: SharedCandidateView) -> ControlledMethodResult:
        """Run direct usability adjudication on the shared candidate view."""
        cost_before = self.client.cost_accountant.to_dict()

        # Render new_evidence explicitly for model visibility
        ne = view.new_evidence
        evidence_str = "\n".join(
            f"  - [{e.evidence_id}] (session: {e.session_id}, "
            f"timestamp: {e.timestamp}, "
            f"source: {e.source_dataset}/{e.source_pointer}) "
            f"\"{e.text}\""
            for e in view.evidence_context
            if e.evidence_id != ne.evidence_id
        ) or "  (none beyond current/new evidence)"
        beliefs_str = "\n".join(
            f"  - {b.belief_id}: \"{b.proposition}\""
            for b in view.candidate_beliefs
        ) or "  (none)"
        replacements_str = "\n".join(
            f"  - {b.belief_id}: \"{b.proposition}\""
            for b in view.candidate_replacement_beliefs
        ) or "  (none)"
        conditions_str_parts: list[str] = []
        for bid, conds in view.candidate_conditions_by_belief:
            for c in conds:
                conditions_str_parts.append(f"  - [{bid}] {c.condition_id}: \"{c.text}\"")
        conditions_str = "\n".join(conditions_str_parts) or "  (none)"

        prompt = self._template.replace("{query}", view.query)
        prompt = prompt.replace("{new_evidence_id}", ne.evidence_id)
        prompt = prompt.replace("{new_evidence_session_id}", ne.session_id)
        prompt = prompt.replace("{new_evidence_timestamp}", ne.timestamp or "")
        prompt = prompt.replace("{new_evidence_source_dataset}", ne.source_dataset)
        prompt = prompt.replace("{new_evidence_source_pointer}", ne.source_pointer)
        prompt = prompt.replace("{new_evidence_text}", ne.text)
        prompt = prompt.replace("{evidence_context}", evidence_str)
        prompt = prompt.replace("{candidate_beliefs}", beliefs_str)
        prompt = prompt.replace("{candidate_replacement_beliefs}", replacements_str)
        prompt = prompt.replace("{candidate_conditions}", conditions_str)

        trace = self.client.generate(
            prompt=prompt,
            model_id=self.model_id,
            provider=self.provider,
            prompt_template_hash=self._template_hash,
            response_schema_version=_RESPONSE_SCHEMA_VERSION,
            parser_version=_PARSER_VERSION,
            temperature=0.0,
            metadata={
                "instance_id": view.instance_id,
                "query_id": view.query_id,
            },
        )

        if trace.status != "success" or not trace.response:
            raise ValueError(
                f"DirectJudge failed: status={trace.status}, "
                f"error={trace.error_message}"
            )

        valid_belief_ids = {b.belief_id for b in view.candidate_beliefs}
        verdicts = self._parse(trace.response, valid_belief_ids, trace.call_id)

        authorized = tuple(v.belief_id for v in verdicts if v.status == DirectUsabilityStatus.USABLE)
        excluded = tuple(v.belief_id for v in verdicts if v.status != DirectUsabilityStatus.USABLE)

        cost_after = self.client.cost_accountant.to_dict()
        instance_cost = _cost_delta(cost_before, cost_after)

        return ControlledMethodResult(
            method_name="directjudge_llm",
            instance_id=view.instance_id,
            query_id=view.query_id,
            authorized_belief_ids=authorized,
            excluded_belief_ids=excluded,
            verdicts=tuple(verdicts),
            model_call_trace_ids=(trace.call_id,),
            cost=instance_cost,
            provenance={
                "prompt_version": _PROMPT_VERSION,
                "model_id": self.model_id,
                "provider": self.provider,
                "model_revision_or_api_version": self.model_revision_or_api_version,
                "view_fingerprint": view.view_fingerprint,
            },
        )

    def _parse(
        self,
        response: str,
        valid_belief_ids: set[str],
        call_id: str,
    ) -> list[DirectUsabilityVerdict]:
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        if not isinstance(data, dict) or "verdicts" not in data:
            raise ValueError(f"Invalid DirectJudge response: missing 'verdicts' key")

        seen_ids: set[str] = set()
        verdicts: list[DirectUsabilityVerdict] = []
        for item in data["verdicts"]:
            bid = item.get("belief_id")
            status_str = item.get("status", "").upper()
            rationale = item.get("rationale", "")

            if not bid:
                raise ValueError(f"Verdict item missing belief_id: {item}")
            bid = _canonicalize_belief_id(bid, valid_belief_ids)
            if bid in seen_ids:
                raise ValueError(
                    f"DirectJudge returned duplicate verdict for belief_id '{bid}'"
                )
            seen_ids.add(bid)

            try:
                status = DirectUsabilityStatus(status_str)
            except ValueError:
                raise ValueError(f"Unknown DirectUsabilityStatus '{status_str}': {item}")

            confidence = item.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                    raise ValueError(
                        f"Confidence must be a number in [0.0, 1.0], got {confidence!r}: {item}"
                    )

            verdict = DirectUsabilityVerdict(
                belief_id=bid,
                status=status,
                rationale=rationale,
                model_call_trace_id=call_id,
                confidence=confidence,
            )
            verdicts.append(verdict)

        missing = valid_belief_ids - seen_ids
        if missing:
            raise ValueError(
                f"DirectJudge omitted verdicts for candidate belief(s): {sorted(missing)}"
            )

        return verdicts
