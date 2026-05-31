"""Stage B DirectJudge-API: prompt rendering, parsing, and per-episode run.

Stage B predicts final usability status directly, with no typed actions, no
RevisionGate, and no DPA.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from retracemem.methods.contracts import DirectUsabilityStatus
from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateSubmission,
    FixedCandidateInputEpisode,
    FixedCandidateGoldRecord,
)
from retracemem.evaluation.multiagent.metrics import _STATUS_MAP_A_TO_COMPARABLE

_PROMPT_JUDGE_FILE = str(
    Path(__file__).resolve().parents[4] / "prompts" / "directjudge" / "direct_usability_v1.txt"
)


def load_direct_judge_template() -> str:
    with open(_PROMPT_JUDGE_FILE, "r", encoding="utf-8") as f:
        return f.read()


def render_direct_judge_prompt(template: str, sub: FixedCandidateSubmission) -> str:
    """Helper to render directjudge prompt matching DirectJudgeLLM.judge exactly."""
    ne = next(e for e in sub.evidence_context if e.evidence_id == sub.new_evidence_id)
    evidence_str = "\n".join(
        f"  - [{e.evidence_id}] (session: {e.session_id}, "
        f"timestamp: {e.timestamp}, "
        f"source: {e.source_dataset}/{e.source_pointer}) "
        f"\"{e.text}\""
        for e in sub.evidence_context
        if e.evidence_id != ne.evidence_id
    ) or "  (none beyond current/new evidence)"
    beliefs_str = "\n".join(
        f"  - {b.belief_id}: \"{b.proposition}\""
        for b in sub.candidate_beliefs
    ) or "  (none)"
    replacements_str = "\n".join(
        f"  - {b.belief_id}: \"{b.proposition}\""
        for b in sub.candidate_replacement_beliefs
    ) or "  (none)"
    conditions_str_parts: list[str] = []
    for bid, conds in sub.candidate_conditions_by_belief:
        for c in conds:
            conditions_str_parts.append(f"  - [{bid}] {c.condition_id}: \"{c.text}\"")
    conditions_str = "\n".join(conditions_str_parts) or "  (none)"

    prompt = template.replace("{query}", sub.query)
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
    return prompt


from retracemem.multiagent.utils import canonicalize_belief_id_with_type


def parse_direct_judge_response(
    response: str,
    valid_belief_ids: set[str],
) -> list[dict[str, Any]]:
    """Parse DirectJudge JSON response."""
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    data = json.loads(text)
    if not isinstance(data, dict) or "verdicts" not in data:
        raise ValueError("Invalid DirectJudge response: missing 'verdicts' key")
    
    verdicts = []
    for item in data["verdicts"]:
        bid = item.get("belief_id")
        status_str = item.get("status", "").upper()
        rationale = item.get("rationale", "")
        confidence = item.get("confidence")
        
        if not bid:
            raise ValueError("Verdict item missing belief_id")
        
        canonical_bid, applied, match_type = canonicalize_belief_id_with_type(bid, valid_belief_ids)
        if match_type == "failed":
            raise ValueError(f"Verdict references unknown belief ID '{bid}' which failed canonicalization")
        
        status = DirectUsabilityStatus(status_str).value
        verdicts.append({
            "raw_belief_id": bid,
            "canonical_belief_id": canonical_bid,
            "canonicalization_applied": applied,
            "canonicalization_type": match_type,
            "status": status,
            "rationale": rationale,
            "confidence": confidence,
        })
    
    missing = valid_belief_ids - {v["canonical_belief_id"] for v in verdicts}
    if missing:
        raise ValueError(f"DirectJudge response omitted verdicts for belief(s): {missing}")
        
    return verdicts



def run_directjudge_on_episode(
    episode: FixedCandidateInputEpisode,
    gold: FixedCandidateGoldRecord,
    client: Any,
    model: str | None,
    provider: str | None,
    dry_run: bool = False,
    mock: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Runs Stage B (Direct Judge flow) on a single episode."""
    direct_judge_template = load_direct_judge_template()
    ep_b_raw_outputs = []
    ep_b_parsed_verdicts = []
    
    strict_stage_b_final_statuses = {}
    canonical_stage_b_final_statuses = {}

    for sub in episode.submissions:
        # Build direct judge prompt
        prompt_b = render_direct_judge_prompt(direct_judge_template, sub)
        
        raw_response_b = ""
        parse_err_b = None
        parsed_verdicts = []

        if dry_run:
            raw_response_b = "DRY RUN MOCK RESPONSE"
            parsed_verdicts = []
        elif not mock and client is not None and model is not None and provider is not None:
            try:
                trace_b = client.generate(
                    prompt=prompt_b,
                    model_id=model,
                    provider=provider,
                    temperature=0.0,
                    seed=42,
                )
                if trace_b.status == "success" and trace_b.response:
                    raw_response_b = trace_b.response
                else:
                    parse_err_b = trace_b.error_message or "Unknown model call failure"
            except Exception as ex:
                parse_err_b = f"{type(ex).__name__}: {ex}"
        else:
            # Mock default usability verdict based on gold snapshot
            verdicts_mock = []
            for b in sub.candidate_beliefs:
                status_raw = gold.gold_snapshot.belief_statuses.get(b.belief_id, "AUTHORIZED")
                status_mapped = _STATUS_MAP_A_TO_COMPARABLE.get(status_raw, "UNCERTAIN")
                verdicts_mock.append({
                    "belief_id": b.belief_id,
                    "status": status_mapped,
                    "rationale": "Mock correct status from gold",
                    "confidence": 1.0
                })
            for b in sub.candidate_replacement_beliefs:
                status_raw = gold.gold_snapshot.belief_statuses.get(b.belief_id, "AUTHORIZED")
                status_mapped = _STATUS_MAP_A_TO_COMPARABLE.get(status_raw, "UNCERTAIN")
                verdicts_mock.append({
                    "belief_id": b.belief_id,
                    "status": status_mapped,
                    "rationale": "Mock correct status from gold",
                    "confidence": 1.0
                })
            raw_response_b = json.dumps({"verdicts": verdicts_mock})

        # Parse verdicts
        if not parse_err_b and not dry_run:
            try:
                valid_belief_ids = {b.belief_id for b in sub.candidate_beliefs} | {
                    b.belief_id for b in sub.candidate_replacement_beliefs
                }
                parsed_verdicts = parse_direct_judge_response(raw_response_b, valid_belief_ids)
            except Exception as ex:
                parse_err_b = str(ex)

        if parse_err_b:
            # Fail cleanly by leaving parsed_verdicts as empty list
            parsed_verdicts = []

        # Cumulative usability verdict update
        for verdict in parsed_verdicts:
            canonical_bid = verdict["canonical_belief_id"]
            canonical_stage_b_final_statuses[canonical_bid] = verdict["status"]
            if not verdict["canonicalization_applied"]:
                strict_stage_b_final_statuses[canonical_bid] = verdict["status"]

        ep_b_raw_outputs.append({
            "submission_id": sub.submission_id,
            "prompt": prompt_b,
            "raw_response": raw_response_b,
            "parse_error": parse_err_b,
        })
        ep_b_parsed_verdicts.append({
            "submission_id": sub.submission_id,
            "verdicts": parsed_verdicts,
            "parse_error": parse_err_b,
        })

    return ep_b_raw_outputs, ep_b_parsed_verdicts, strict_stage_b_final_statuses, canonical_stage_b_final_statuses


