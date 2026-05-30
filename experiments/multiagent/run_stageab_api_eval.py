#!/usr/bin/env python3
"""Stage A vs Stage B API Evaluation Runner for ReTrace ICLR 2027 Paper 1.

Stage A (Decomposition):
- API model predicts typed revision actions.
- RevisionGate checks and deterministic DPA computes final status.

Stage B (Direct Judge):
- Same API model directly predicts usability status.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv

# Ensure src and repo root are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=False)

from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
)
from retracemem.authorization import EvidenceProposalBatch
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.multiagent.commit import commit_submission_sequence
from retracemem.multiagent.contracts import SubagentMemorySubmission
from retracemem.methods.contracts import DirectUsabilityStatus

from experiments.multiagent.dev_expansion import generate_expanded_episodes
from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    TypedRevisionTarget,
    FixedCandidateGoldRecord,
    FixedCandidateInputEpisode,
)
from experiments.multiagent.stagec_policy import ClosedAPIZeroShotProposer
from experiments.multiagent.run_stagec_adapter_eval import rename_submission

_STATUS_MAP_A_TO_COMPARABLE = {
    "AUTHORIZED": "USABLE",
    "BLOCKED": "NOT_USABLE",
    "SUPERSEDED": "NOT_USABLE",
    "UNRESOLVED": "UNCERTAIN",
}

# Prompt helper for Stage B direct usability matching directjudge logic
_PROMPT_JUDGE_FILE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "prompts", "directjudge", "direct_usability_v1.txt"
)


def load_direct_judge_template() -> str:
    with open(_PROMPT_JUDGE_FILE, "r", encoding="utf-8") as f:
        return f.read()


def rename_string(s: str | None, old_ns: str, new_ns: str) -> str | None:
    if s is None:
        return None
    return s.replace(old_ns, new_ns)


def rename_episode_and_gold(
    episode: FixedCandidateInputEpisode,
    gold: FixedCandidateGoldRecord,
) -> tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]:
    """Rename namespace of an episode and its gold records."""
    old_ns = episode.episode_id
    new_ns = f"{old_ns}__heldout_base"

    # Rename submissions
    renamed_subs = tuple(rename_submission(s, old_ns, new_ns) for s in episode.submissions)

    # Rename downstream tasks
    renamed_tasks = []
    for t in episode.downstream_tasks:
        renamed_tasks.append(
            t.__class__(
                task_id=rename_string(t.task_id, old_ns, new_ns),
                query=rename_string(t.query, old_ns, new_ns),
                expected_answer_or_action=rename_string(t.expected_answer_or_action, old_ns, new_ns),
                relevant_belief_ids=tuple(rename_string(bid, old_ns, new_ns) for bid in t.relevant_belief_ids),
                protected_belief_ids=tuple(rename_string(bid, old_ns, new_ns) for bid in t.protected_belief_ids),
                metadata=t.metadata,
            )
        )

    # Rename gold record
    renamed_belief_statuses = {}
    for bid, status in gold.gold_snapshot.belief_statuses.items():
        renamed_belief_statuses[rename_string(bid, old_ns, new_ns)] = status

    renamed_gold_snapshot = gold.gold_snapshot.__class__(
        belief_statuses=renamed_belief_statuses,
        required_authorized_belief_ids=tuple(rename_string(bid, old_ns, new_ns) for bid in gold.gold_snapshot.required_authorized_belief_ids),
        forbidden_authorized_belief_ids=tuple(rename_string(bid, old_ns, new_ns) for bid in gold.gold_snapshot.forbidden_authorized_belief_ids),
        rationale=gold.gold_snapshot.rationale,
    )

    renamed_targets = []
    for target in gold.gold_typed_targets:
        renamed_targets.append(
            TypedRevisionTarget(
                submission_id=rename_string(target.submission_id, old_ns, new_ns),
                action_type=target.action_type,
                target_belief_id=rename_string(target.target_belief_id, old_ns, new_ns),
                target_condition_id=rename_string(target.target_condition_id, old_ns, new_ns),
                replacement_belief_id=rename_string(target.replacement_belief_id, old_ns, new_ns),
                rationale=target.rationale,
                evidence_ids=tuple(rename_string(eid, old_ns, new_ns) for eid in target.evidence_ids),
            )
        )

    renamed_episode = episode.__class__(
        episode_id=new_ns,
        domain=episode.domain,
        failure_type_public_or_controlled=episode.failure_type_public_or_controlled,
        subagent_roles=episode.subagent_roles,
        submissions=renamed_subs,
        downstream_tasks=tuple(renamed_tasks),
        stress_factors=episode.stress_factors,
        split=episode.split,
        protocol_mode=episode.protocol_mode,
        proposal_source=episode.proposal_source,
        metadata=episode.metadata,
    )

    renamed_gold = gold.__class__(
        episode_id=new_ns,
        gold_snapshot=renamed_gold_snapshot,
        gold_typed_targets=tuple(renamed_targets),
        failure_type=gold.failure_type,
        metadata=gold.metadata,
    )

    return renamed_episode, renamed_gold


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


def check_grounding_error_stage_a(action: dict[str, Any], sub: FixedCandidateSubmission) -> bool:
    """Return True if there is a grounding error in Stage A action."""
    action_type = action.get("action_type")
    if action_type == "NO_REVISION":
        return False
        
    target_belief_id = action.get("target_belief_id")
    target_condition_id = action.get("target_condition_id")
    replacement_belief_id = action.get("replacement_belief_id")
    evidence_ids = action.get("evidence_ids") or []

    # Grounding check for target belief/condition
    valid_belief_ids = {b.belief_id for b in sub.candidate_beliefs}
    valid_replacement_ids = {b.belief_id for b in sub.candidate_replacement_beliefs}
    valid_condition_ids = {
        c.condition_id
        for _, conds in sub.candidate_conditions_by_belief
        for c in conds
    }
    valid_evidence_ids = {e.evidence_id for e in sub.evidence_context} | {sub.new_evidence_id}

    if action_type in ("SUPERSEDES", "UNCERTAIN", "REAFFIRMS"):
        if not target_belief_id or target_belief_id not in valid_belief_ids:
            return True
        if action_type == "SUPERSEDES":
            if not replacement_belief_id or replacement_belief_id not in valid_replacement_ids:
                return True
    elif action_type in ("BLOCKS", "RELEASES"):
        if not target_condition_id or target_condition_id not in valid_condition_ids:
            return True
            
    for eid in evidence_ids:
        if eid not in valid_evidence_ids:
            return True

    return False


def check_grounding_error_stage_b(verdict: dict[str, Any], valid_belief_ids: set[str]) -> bool:
    """Return True if Stage B verdict contains an invalid raw belief_id."""
    bid = verdict.get("raw_belief_id")
    return bid not in valid_belief_ids


def compute_stage_a_action_metrics(
    pred_actions: list[dict[str, Any]],
    gold_actions: tuple[TypedRevisionTarget, ...],
    sub: FixedCandidateSubmission,
) -> dict[str, float]:
    """Compute fine-grained metrics for Stage A proposal action match."""
    valid_json = 1.0  # Assumes parsed if we reached here
    
    # Grounding checks
    target_grounding_correct = 0
    total_pred = len(pred_actions)
    for act in pred_actions:
        if not check_grounding_error_stage_a(act, sub):
            target_grounding_correct += 1
    target_grounding = target_grounding_correct / total_pred if total_pred > 0 else 1.0

    # Match exact actions
    def canonical_action_tuple(a: dict[str, Any] | TypedRevisionTarget) -> tuple:
        if isinstance(a, dict):
            evs = tuple(sorted(a.get("evidence_ids") or []))
            return (
                a.get("action_type"),
                a.get("target_belief_id"),
                a.get("target_condition_id"),
                a.get("replacement_belief_id"),
                evs,
            )
        else:
            evs = tuple(sorted(a.evidence_ids))
            return (
                a.action_type,
                a.target_belief_id,
                a.target_condition_id,
                a.replacement_belief_id,
                evs,
            )

    pred_tuples = {canonical_action_tuple(a) for a in pred_actions}
    gold_tuples = {canonical_action_tuple(a) for a in gold_actions}
    exact_action_match = 1.0 if pred_tuples == gold_tuples else 0.0

    # Action type and evidence grounding
    action_type_match_correct = 0
    evidence_grounding_correct = 0
    matched_count = 0

    for pred_act in pred_actions:
        pred_target = pred_act.get("target_belief_id") or pred_act.get("target_condition_id")
        pred_action_type = pred_act.get("action_type")
        
        # Match NO_REVISION
        if pred_action_type == "NO_REVISION":
            for gold_act in gold_actions:
                if gold_act.action_type == "NO_REVISION":
                    matched_count += 1
                    action_type_match_correct += 1
                    pred_evs = set(pred_act.get("evidence_ids") or [])
                    gold_evs = set(gold_act.evidence_ids)
                    if pred_evs == gold_evs:
                         evidence_grounding_correct += 1
                    break
            continue

        if not pred_target:
            continue
        # Find matching gold action by target
        for gold_act in gold_actions:
            gold_target = gold_act.target_belief_id or gold_act.target_condition_id
            if pred_target == gold_target:
                matched_count += 1
                if pred_act.get("action_type") == gold_act.action_type:
                    action_type_match_correct += 1
                
                pred_evs = set(pred_act.get("evidence_ids") or [])
                gold_evs = set(gold_act.evidence_ids)
                if pred_evs == gold_evs:
                    evidence_grounding_correct += 1
                break

    if not pred_actions and not gold_actions:
        action_type_match = 1.0
        evidence_grounding = 1.0
    else:
        action_type_match = action_type_match_correct / max(len(pred_actions), len(gold_actions))
        evidence_grounding = evidence_grounding_correct / max(len(pred_actions), len(gold_actions))

    return {
        "valid_json": valid_json,
        "action_type_match": action_type_match,
        "target_grounding": target_grounding,
        "evidence_grounding": evidence_grounding,
        "exact_action_match": exact_action_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A vs Stage B API Evaluation Runner")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--live", action="store_true", help="Enable live API calls")
    mode_group.add_argument("--dry-run", action="store_true", help="Dry run prompt generation and dataset checking")
    mode_group.add_argument("--mock", action="store_true", help="Explicit mock replay mode")
    
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases to evaluate")
    parser.add_argument("--resume", action="store_true", help="Resume from cached files")
    parser.add_argument("--provider", default="siliconflow", help="API Provider name")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3", help="Model ID")
    parser.add_argument("--api-key", default=None, help="Explicit API key")
    parser.add_argument("--base-url", default=None, help="Explicit base URL")
    parser.add_argument("--output-dir", default="outputs/runs/stageab_dev70", help="Output directory")
    args = parser.parse_args()

    if args.output_dir.startswith("artifacts/"):
        print("\n⚠ WARNING: output_dir starts with 'artifacts/'. Generated artifacts in artifacts/ directory should not be committed to GitHub under paper governance policy. Please consider using 'outputs/runs/' instead.\n")

    print("=" * 80)
    print("STAGE A VS STAGE B API EVALUATION RUNNER")
    print("=" * 80)
    print(f"Mode: {'LIVE API' if args.live else 'DRY RUN' if args.dry_run else 'MOCK REPLAY'}")
    print(f"Provider: {args.provider}, Model: {args.model}")
    print(f"Output Directory: {args.output_dir}")
    print()

    # Load 70 Dev Expansion Cases
    ep_gold_pairs = generate_expanded_episodes()
    print(f"Successfully loaded {len(ep_gold_pairs)} episodes from dev_expansion.")

    # Apply Namespace replacement for _v5 cases
    processed_cases: list[tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]] = []
    for ep, gold in ep_gold_pairs:
        if ep.episode_id.endswith("_v5"):
            ep_renamed, gold_renamed = rename_episode_and_gold(ep, gold)
            processed_cases.append((ep_renamed, gold_renamed))
        else:
            processed_cases.append((ep, gold))

    if args.max_cases is not None:
        processed_cases = processed_cases[:args.max_cases]
        print(f"Restricted to first {len(processed_cases)} cases via --max-cases.")

    # Initialize directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize live client if live
    client = None
    if args.live:
        api_key = args.api_key or os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("Live mode requires API key to be set via --api-key or SILICONFLOW_API_KEY env var.")
        if not args.provider or args.provider == "mock":
            raise ValueError("Live mode requires a valid non-mock --provider.")
        if not args.model:
            raise ValueError("Live mode requires a valid --model ID.")

        cache_file = output_path / "api_cache.jsonl"
        cache = JSONLCache(str(cache_file))
        http_provider = HTTPLLMProvider(api_key=api_key, base_url=args.base_url)
        client = CachedLLMClient(cache=cache, provider_client=http_provider)
        print(f"✓ Initialized live API client with cache at: {cache_file}")

    # Resume capability: load already processed cases
    stage_a_raw_rows = []
    stage_b_raw_rows = []
    stage_a_parsed_rows = []
    stage_b_parsed_rows = []
    dpa_trace_rows = []

    resumed_episodes = set()
    if args.resume:
        a_raw_file = output_path / "stage_a_raw.jsonl"
        b_raw_file = output_path / "stage_b_raw.jsonl"
        a_parsed_file = output_path / "stage_a_parsed.jsonl"
        b_parsed_file = output_path / "stage_b_parsed.jsonl"
        dpa_traces_file = output_path / "dpa_traces.jsonl"

        # If files exist, load them and figure out what is done
        if a_raw_file.exists() and b_raw_file.exists():
            print("✓ Resuming. Found existing raw/parsed result files. Loading...")
            
            # Read existing rows to resume
            def load_jsonl(p: Path):
                rows = []
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
                return rows

            try:
                stage_a_raw_rows = load_jsonl(a_raw_file)
                stage_b_raw_rows = load_jsonl(b_raw_file)
                stage_a_parsed_rows = load_jsonl(a_parsed_file)
                stage_b_parsed_rows = load_jsonl(b_parsed_file)
                dpa_trace_rows = load_jsonl(dpa_traces_file)

                # Collect episodes that are fully finished in both Stage A and Stage B
                finished_a = {r["episode_id"] for r in stage_a_parsed_rows}
                finished_b = {r["episode_id"] for r in stage_b_parsed_rows}
                resumed_episodes = finished_a & finished_b
                print(f"✓ Resumed {len(resumed_episodes)} fully completed cases.")
            except Exception as e:
                print(f"⚠ Failed to resume cleanly: {e}. Will run from scratch.")
                resumed_episodes = set()
                stage_a_raw_rows = []
                stage_b_raw_rows = []
                stage_a_parsed_rows = []
                stage_b_parsed_rows = []
                dpa_trace_rows = []

    # Policy setup for Stage A Proposing
    if args.live:
        proposer_a = ClosedAPIZeroShotProposer(
            provider_kind=args.provider,
            model_id=args.model,
            client=client,
        )
    elif args.mock:
        proposer_a = ClosedAPIZeroShotProposer(
            provider_kind="mock",
            model_id=None,
            client=None,
        )
    else: # dry-run
        proposer_a = ClosedAPIZeroShotProposer(
            provider_kind="mock",
            model_id=None,
            client=None,
        )
    direct_judge_template = load_direct_judge_template()

    # Main Execution Loop
    for idx, (episode, gold) in enumerate(processed_cases):
        ep_id = episode.episode_id
        if ep_id in resumed_episodes:
            print(f"[{idx+1}/{len(processed_cases)}] Skipping (resumed) {ep_id}")
            continue

        print(f"[{idx+1}/{len(processed_cases)}] Evaluating {ep_id} ({episode.failure_type_public_or_controlled})")

        # -------------------------------------------------------------
        # STAGE A (Decomposition Flow)
        # -------------------------------------------------------------
        subagent_subs = []
        ep_a_raw_outputs = []
        ep_a_parsed_actions = []
        ep_a_parse_errors = []

        for sub in episode.submissions:
            proposal_output = proposer_a.propose(sub)

            raw_response = proposal_output.metadata.get("raw_response", "")
            full_prompt = proposal_output.metadata.get("prompt", "")
            parse_err = "\n".join(proposal_output.errors) if proposal_output.errors else None

            parsed_actions = []
            if proposal_output.parsing_valid:
                for t in proposal_output.parsed_actions:
                    parsed_actions.append({
                        "action_type": t.action_type,
                        "target_belief_id": t.target_belief_id,
                        "target_condition_id": t.target_condition_id,
                        "replacement_belief_id": t.replacement_belief_id,
                        "rationale": t.rationale,
                        "evidence_ids": list(t.evidence_ids),
                    })

            proposal_batches = proposal_output.proposal_batches
            proposal_edges = []
            for batch in proposal_batches:
                for edge in batch.edges:
                    proposal_edges.append({
                        "edge_id": edge.edge_id,
                        "edge_type": edge.edge_type.value if hasattr(edge.edge_type, "value") else str(edge.edge_type),
                        "evidence_id": edge.evidence_id,
                        "target_kind": edge.target_kind,
                        "target_id": edge.target_id,
                        "replacement_belief_id": edge.replacement_belief_id,
                        "rationale": edge.rationale,
                    })

            # Build SubagentMemorySubmission with predicted proposal batches
            subagent_sub = SubagentMemorySubmission(
                submission_id=sub.submission_id,
                producer_id=sub.producer_id,
                producer_role=sub.producer_role,
                parent_snapshot_id=sub.parent_snapshot_id,
                observed_at=sub.observed_at,
                instance_id=sub.instance_id,
                query_id=sub.query_id,
                query=sub.query,
                evidence_context=sub.evidence_context,
                new_evidence_id=sub.new_evidence_id,
                candidate_beliefs=sub.candidate_beliefs,
                candidate_replacement_beliefs=sub.candidate_replacement_beliefs,
                candidate_conditions_by_belief=sub.candidate_conditions_by_belief,
                dependency_edges_by_belief=sub.dependency_edges_by_belief,
                proposal_batches=proposal_batches,
                task_id=sub.task_id,
                metadata=sub.metadata,
            )
            subagent_subs.append(subagent_sub)

            ep_a_raw_outputs.append({
                "submission_id": sub.submission_id,
                "prompt": full_prompt,
                "raw_response": raw_response,
                "parse_error": parse_err,
            })
            ep_a_parsed_actions.append({
                "submission_id": sub.submission_id,
                "actions": parsed_actions,
                "proposal_edges": proposal_edges,
                "parse_error": parse_err,
            })

        # Run Stage A sequence commit to get final DPA statuses
        seq_res = commit_submission_sequence(tuple(subagent_subs), final_snapshot_evaluation=True)
        final_dpa_statuses = seq_res.final_belief_statuses

        # Write Stage A logs
        stage_a_raw_rows.append({
            "episode_id": ep_id,
            "submissions": ep_a_raw_outputs,
        })
        stage_a_parsed_rows.append({
            "episode_id": ep_id,
            "submissions": ep_a_parsed_actions,
            "final_belief_statuses": final_dpa_statuses,
        })
        dpa_trace_rows.append({
            "episode_id": ep_id,
            "dpa_trace": seq_res.trace,
            "final_belief_statuses": final_dpa_statuses,
        })

        # -------------------------------------------------------------
        # STAGE B (Direct Judge Flow)
        # -------------------------------------------------------------
        ep_b_raw_outputs = []
        ep_b_parsed_verdicts = []
        ep_b_parse_errors = []
        
        strict_stage_b_final_statuses = {}
        canonical_stage_b_final_statuses = {}

        for sub in episode.submissions:
            # Build direct judge prompt
            prompt_b = render_direct_judge_prompt(direct_judge_template, sub)
            
            raw_response_b = ""
            parse_err_b = None
            parsed_verdicts = []

            if args.dry_run:
                raw_response_b = "DRY RUN MOCK RESPONSE"
                parsed_verdicts = []
            elif args.live and client is not None:
                try:
                    trace_b = client.generate(
                        prompt=prompt_b,
                        model_id=args.model,
                        provider=args.provider,
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
                # Mock default usability verdict
                verdicts_mock = []
                for b in sub.candidate_beliefs:
                    verdicts_mock.append({
                        "belief_id": b.belief_id,
                        "status": "USABLE",
                        "rationale": "Mock default usable status",
                        "confidence": 1.0
                    })
                raw_response_b = json.dumps({"verdicts": verdicts_mock})

            # Parse verdicts
            if not parse_err_b:
                try:
                    valid_belief_ids = {b.belief_id for b in sub.candidate_beliefs}
                    parsed_verdicts = parse_direct_judge_response(raw_response_b, valid_belief_ids)
                except Exception as ex:
                    parse_err_b = str(ex)

            if parse_err_b:
                ep_b_parse_errors.append(parse_err_b)
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

        # Write Stage B logs
        stage_b_raw_rows.append({
            "episode_id": ep_id,
            "submissions": ep_b_raw_outputs,
        })
        stage_b_parsed_rows.append({
            "episode_id": ep_id,
            "submissions": ep_b_parsed_verdicts,
            "strict_final_belief_statuses": strict_stage_b_final_statuses,
            "canonicalized_final_belief_statuses": canonical_stage_b_final_statuses,
            "final_belief_statuses": canonical_stage_b_final_statuses,
        })

    # Save outputs if not dry-run
    if not args.dry_run:
        def save_jsonl(p: Path, data: list):
            with open(p, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row) + "\n")

        save_jsonl(output_path / "stage_a_raw.jsonl", stage_a_raw_rows)
        save_jsonl(output_path / "stage_b_raw.jsonl", stage_b_raw_rows)
        save_jsonl(output_path / "stage_a_parsed.jsonl", stage_a_parsed_rows)
        save_jsonl(output_path / "stage_b_parsed.jsonl", stage_b_parsed_rows)
        save_jsonl(output_path / "dpa_traces.jsonl", dpa_trace_rows)
        print("✓ Wrote all raw, parsed, and trace files to output folder.")

    # -------------------------------------------------------------
    # METRICS CALCULATION
    # -------------------------------------------------------------
    print("\nCalculating metrics...")

    # Build maps of final results for easy indexing
    stage_a_results_map = {r["episode_id"]: r for r in stage_a_parsed_rows}
    stage_b_results_map = {r["episode_id"]: r for r in stage_b_parsed_rows}
    stage_a_raw_map = {r["episode_id"]: r for r in stage_a_raw_rows}
    stage_b_raw_map = {r["episode_id"]: r for r in stage_b_raw_rows}

    # Case Breakdown Table Setup
    failure_breakdown_rows = []

    # Overall counters for global metrics
    # Metric format: { "stage_a": { ... }, "stage_b": { ... } }
    stage_metrics = {
        "stage_a": {
            "correct_beliefs": 0,
            "total_beliefs": 0,
            "over_updates": 0,
            "not_usable_total": 0,
            "under_updates": 0,
            "usable_total": 0,
            "uncertainty_errors": 0,
            "grounding_errors": 0,
            "valid_outputs": 0,
            "total_outputs": 0,
        },
        "stage_b_strict": {
            "correct_beliefs": 0,
            "total_beliefs": 0,
            "over_updates": 0,
            "not_usable_total": 0,
            "under_updates": 0,
            "usable_total": 0,
            "uncertainty_errors": 0,
        },
        "stage_b_canonicalized": {
            "correct_beliefs": 0,
            "total_beliefs": 0,
            "over_updates": 0,
            "not_usable_total": 0,
            "under_updates": 0,
            "usable_total": 0,
            "uncertainty_errors": 0,
        },
        "stage_b_common": {
            "grounding_errors": 0,
            "valid_outputs": 0,
            "total_outputs": 0,
        }
    }

    # Stage A Action-level metrics counters
    action_metrics_counters = {
        "valid_json": [],
        "action_type_match": [],
        "target_grounding": [],
        "evidence_grounding": [],
        "exact_action_match": [],
        "no_revision_match": [],
    }

    # Stage B Canonicalization counters
    total_stage_b_verdicts = 0
    canonicalized_stage_b_verdicts = 0
    fuzzy_stage_b_verdicts = 0

    for ep, gold in processed_cases:
        ep_id = episode.episode_id if ep.episode_id.endswith("__heldout_base") else ep.episode_id
        # Map correctly due to rename potential
        ep_id = ep.episode_id

        gold_statuses = gold.gold_snapshot.belief_statuses
        
        # Pull Results
        res_a = stage_a_results_map.get(ep_id, {})
        res_b = stage_b_results_map.get(ep_id, {})
        raw_a = stage_a_raw_map.get(ep_id, {})
        raw_b = stage_b_raw_map.get(ep_id, {})

        pred_a_statuses = res_a.get("final_belief_statuses", {})
        strict_pred_b_statuses = res_b.get("strict_final_belief_statuses", {})
        canonical_pred_b_statuses = res_b.get("canonicalized_final_belief_statuses", {})

        # Compute Action Metrics for Stage A only per submission
        for s_parsed in res_a.get("submissions", []):
            sub_id = s_parsed["submission_id"]
            orig_sub = next(s for s in ep.submissions if s.submission_id == sub_id)
            pred_actions_for_sub = s_parsed.get("actions", [])
            gold_actions_for_sub = tuple(t for t in gold.gold_typed_targets if t.submission_id == sub_id)
            
            act_metrics = compute_stage_a_action_metrics(pred_actions_for_sub, gold_actions_for_sub, orig_sub)
            for k, v in act_metrics.items():
                action_metrics_counters[k].append(v)

        # Grounding errors Stage A
        has_grounding_err_a = False
        for s_parsed in res_a.get("submissions", []):
            sub_id = s_parsed["submission_id"]
            orig_sub = next(s for s in ep.submissions if s.submission_id == sub_id)
            for act in s_parsed.get("actions", []):
                if check_grounding_error_stage_a(act, orig_sub):
                    has_grounding_err_a = True
                    break

        # Grounding errors Stage B
        has_grounding_err_b = False
        for s_parsed in res_b.get("submissions", []):
            sub_id = s_parsed["submission_id"]
            orig_sub = next(s for s in ep.submissions if s.submission_id == sub_id)
            valid_belief_ids = {b.belief_id for b in orig_sub.candidate_beliefs}
            for verd in s_parsed.get("verdicts", []):
                if check_grounding_error_stage_b(verd, valid_belief_ids):
                    has_grounding_err_b = True
                    break

        # Collect Stage B canonicalization stats
        for s_parsed in res_b.get("submissions", []):
            for verd in s_parsed.get("verdicts", []):
                total_stage_b_verdicts += 1
                if verd.get("canonicalization_applied"):
                    canonicalized_stage_b_verdicts += 1
                if verd.get("canonicalization_type") == "fuzzy":
                    fuzzy_stage_b_verdicts += 1

        # Output Validity (Parse Errors)
        has_parse_error_a = any(s.get("parse_error") is not None for s in raw_a.get("submissions", []))
        has_parse_error_b = any(s.get("parse_error") is not None for s in raw_b.get("submissions", []))

        stage_metrics["stage_a"]["total_outputs"] += len(ep.submissions)
        stage_metrics["stage_a"]["valid_outputs"] += sum(1 for s in raw_a.get("submissions", []) if s.get("parse_error") is None)
        
        stage_metrics["stage_b_common"]["total_outputs"] += len(ep.submissions)
        stage_metrics["stage_b_common"]["valid_outputs"] += sum(1 for s in raw_b.get("submissions", []) if s.get("parse_error") is None)

        if has_grounding_err_a:
            stage_metrics["stage_a"]["grounding_errors"] += 1
        if has_grounding_err_b:
            stage_metrics["stage_b_common"]["grounding_errors"] += 1

        # Evaluate final statuses
        ep_correct_a = 0
        ep_correct_b_strict = 0
        ep_correct_b_canonical = 0
        total_beliefs_in_ep = len(gold_statuses)

        for bid, gold_status in gold_statuses.items():
            gold_comp = _STATUS_MAP_A_TO_COMPARABLE.get(gold_status, "UNCERTAIN")
            
            # Stage A Mapping
            actual_a_raw = pred_a_statuses.get(bid, "UNRESOLVED")
            actual_a_comp = _STATUS_MAP_A_TO_COMPARABLE.get(actual_a_raw, "UNCERTAIN")

            # Stage B Mapping (strict & canonicalized)
            actual_b_strict = strict_pred_b_statuses.get(bid, "UNCERTAIN")
            actual_b_canonical = canonical_pred_b_statuses.get(bid, "UNCERTAIN")

            # Final status accuracy counters for Stage A
            stage_metrics["stage_a"]["total_beliefs"] += 1
            if actual_a_comp == gold_comp:
                stage_metrics["stage_a"]["correct_beliefs"] += 1
                ep_correct_a += 1

            # Strict Stage B
            stage_metrics["stage_b_strict"]["total_beliefs"] += 1
            if actual_b_strict == gold_comp:
                stage_metrics["stage_b_strict"]["correct_beliefs"] += 1
                ep_correct_b_strict += 1

            # Canonicalized Stage B
            stage_metrics["stage_b_canonicalized"]["total_beliefs"] += 1
            if actual_b_canonical == gold_comp:
                stage_metrics["stage_b_canonicalized"]["correct_beliefs"] += 1
                ep_correct_b_canonical += 1

            # Over update (Stale propagation)
            if gold_comp == "NOT_USABLE":
                stage_metrics["stage_a"]["not_usable_total"] += 1
                if actual_a_comp == "USABLE":
                    stage_metrics["stage_a"]["over_updates"] += 1

                stage_metrics["stage_b_strict"]["not_usable_total"] += 1
                if actual_b_strict == "USABLE":
                    stage_metrics["stage_b_strict"]["over_updates"] += 1

                stage_metrics["stage_b_canonicalized"]["not_usable_total"] += 1
                if actual_b_canonical == "USABLE":
                    stage_metrics["stage_b_canonicalized"]["over_updates"] += 1

            # Under update
            if gold_comp == "USABLE":
                stage_metrics["stage_a"]["usable_total"] += 1
                if actual_a_comp != "USABLE":
                    stage_metrics["stage_a"]["under_updates"] += 1

                stage_metrics["stage_b_strict"]["usable_total"] += 1
                if actual_b_strict != "USABLE":
                    stage_metrics["stage_b_strict"]["under_updates"] += 1

                stage_metrics["stage_b_canonicalized"]["usable_total"] += 1
                if actual_b_canonical != "USABLE":
                    stage_metrics["stage_b_canonicalized"]["under_updates"] += 1

            # Uncertainty error rate
            if gold_comp == "UNCERTAIN":
                if actual_a_comp != "UNCERTAIN":
                    stage_metrics["stage_a"]["uncertainty_errors"] += 1
                if actual_b_strict != "UNCERTAIN":
                    stage_metrics["stage_b_strict"]["uncertainty_errors"] += 1
                if actual_b_canonical != "UNCERTAIN":
                    stage_metrics["stage_b_canonicalized"]["uncertainty_errors"] += 1
            else:
                if actual_a_comp == "UNCERTAIN":
                    stage_metrics["stage_a"]["uncertainty_errors"] += 1
                if actual_b_strict == "UNCERTAIN":
                    stage_metrics["stage_b_strict"]["uncertainty_errors"] += 1
                if actual_b_canonical == "UNCERTAIN":
                    stage_metrics["stage_b_canonicalized"]["uncertainty_errors"] += 1

        # Failure breakdown CSV row
        failure_breakdown_rows.append({
            "episode_id": ep_id,
            "failure_type": ep.failure_type_public_or_controlled,
            "domain": ep.domain,
            "total_beliefs": total_beliefs_in_ep,
            "correct_beliefs_a": ep_correct_a,
            "correct_beliefs_b_strict": ep_correct_b_strict,
            "correct_beliefs_b_canonicalized": ep_correct_b_canonical,
            "accuracy_a": ep_correct_a / total_beliefs_in_ep if total_beliefs_in_ep > 0 else 1.0,
            "accuracy_b_strict": ep_correct_b_strict / total_beliefs_in_ep if total_beliefs_in_ep > 0 else 1.0,
            "accuracy_b_canonicalized": ep_correct_b_canonical / total_beliefs_in_ep if total_beliefs_in_ep > 0 else 1.0,
            "has_parse_error_a": has_parse_error_a,
            "has_parse_error_b": has_parse_error_b,
            "has_grounding_error_a": has_grounding_err_a,
            "has_grounding_error_b": has_grounding_err_b,
        })

    # Global aggregate metric calculation
    def calc_rate(num, den):
        return num / den if den > 0 else 0.0

    global_metrics = {
        "stage_a": {
            "final_status_accuracy": calc_rate(stage_metrics["stage_a"]["correct_beliefs"], stage_metrics["stage_a"]["total_beliefs"]),
            "over_update_rate": calc_rate(stage_metrics["stage_a"]["over_updates"], stage_metrics["stage_a"]["not_usable_total"]),
            "stale_propagation_rate": calc_rate(stage_metrics["stage_a"]["over_updates"], stage_metrics["stage_a"]["not_usable_total"]),
            "under_update_rate": calc_rate(stage_metrics["stage_a"]["under_updates"], stage_metrics["stage_a"]["usable_total"]),
            "uncertainty_error_rate": calc_rate(stage_metrics["stage_a"]["uncertainty_errors"], stage_metrics["stage_a"]["total_beliefs"]),
            "grounding_error_rate": calc_rate(stage_metrics["stage_a"]["grounding_errors"], len(processed_cases)),
            "valid_output_rate": calc_rate(stage_metrics["stage_a"]["valid_outputs"], stage_metrics["stage_a"]["total_outputs"]),
            # Stage A specific metrics
            "valid_json": sum(action_metrics_counters["valid_json"]) / len(action_metrics_counters["valid_json"]) if action_metrics_counters["valid_json"] else 0.0,
            "action_type_match": sum(action_metrics_counters["action_type_match"]) / len(action_metrics_counters["action_type_match"]) if action_metrics_counters["action_type_match"] else 0.0,
            "target_grounding": sum(action_metrics_counters["target_grounding"]) / len(action_metrics_counters["target_grounding"]) if action_metrics_counters["target_grounding"] else 0.0,
            "evidence_grounding": sum(action_metrics_counters["evidence_grounding"]) / len(action_metrics_counters["evidence_grounding"]) if action_metrics_counters["evidence_grounding"] else 0.0,
            "exact_action_match": sum(action_metrics_counters["exact_action_match"]) / len(action_metrics_counters["exact_action_match"]) if action_metrics_counters["exact_action_match"] else 0.0,
            "no_revision_match": sum(action_metrics_counters["no_revision_match"]) / len(action_metrics_counters["no_revision_match"]) if action_metrics_counters["no_revision_match"] else 0.0,
        },
        "stage_b": {
            "final_status_accuracy": calc_rate(stage_metrics["stage_b_canonicalized"]["correct_beliefs"], stage_metrics["stage_b_canonicalized"]["total_beliefs"]),
            "strict_final_status_accuracy": calc_rate(stage_metrics["stage_b_strict"]["correct_beliefs"], stage_metrics["stage_b_strict"]["total_beliefs"]),
            "canonicalized_final_status_accuracy": calc_rate(stage_metrics["stage_b_canonicalized"]["correct_beliefs"], stage_metrics["stage_b_canonicalized"]["total_beliefs"]),
            "canonicalization_rate": calc_rate(canonicalized_stage_b_verdicts, total_stage_b_verdicts),
            "fuzzy_canonicalization_rate": calc_rate(fuzzy_stage_b_verdicts, total_stage_b_verdicts),
            "over_update_rate": calc_rate(stage_metrics["stage_b_canonicalized"]["over_updates"], stage_metrics["stage_b_canonicalized"]["not_usable_total"]),
            "stale_propagation_rate": calc_rate(stage_metrics["stage_b_canonicalized"]["over_updates"], stage_metrics["stage_b_canonicalized"]["not_usable_total"]),
            "under_update_rate": calc_rate(stage_metrics["stage_b_canonicalized"]["under_updates"], stage_metrics["stage_b_canonicalized"]["usable_total"]),
            "uncertainty_error_rate": calc_rate(stage_metrics["stage_b_canonicalized"]["uncertainty_errors"], stage_metrics["stage_b_canonicalized"]["total_beliefs"]),
            "grounding_error_rate": calc_rate(stage_metrics["stage_b_common"]["grounding_errors"], len(processed_cases)),
            "valid_output_rate": calc_rate(stage_metrics["stage_b_common"]["valid_outputs"], stage_metrics["stage_b_common"]["total_outputs"]),
        }
    }

    print("\n" + "-" * 40)
    print("STAGE A (Decomposition + DPA) METRICS:")
    print("-" * 40)
    for k, v in global_metrics["stage_a"].items():
        print(f"  {k:28s}: {v:.4f}")

    print("\n" + "-" * 40)
    print("STAGE B (Direct usability judge) METRICS:")
    print("-" * 40)
    for k, v in global_metrics["stage_b"].items():
        print(f"  {k:28s}: {v:.4f}")
    print("-" * 40)

    # Save metrics and failure breakdown files
    if not args.dry_run:
        with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(global_metrics, f, indent=2)

        # Save failure breakdown csv
        csv_file = output_path / "failure_breakdown.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=failure_breakdown_rows[0].keys())
            writer.writeheader()
            writer.writerows(failure_breakdown_rows)

        # Save manifest.json
        # Calculate prompt template hash
        from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
        temp_policy = PromptTypedRevisionPolicy()
        sys_prompt = temp_policy.build_system_prompt()
        prompt_template_hash = hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest()

        manifest = {
            "run_identifier": "development_live_api_run / not_final_paper_result" if args.live else "development_run",
            "executed_at": datetime.datetime.now().isoformat(),
            "run_mode": "live" if args.live else "mock" if args.mock else "dry-run",
            "is_live_api_result": args.live,
            "mock_default_used": args.mock,
            "provider": args.provider,
            "model": args.model,
            "resume_mode": args.resume,
            "temperature": 0.0,
            "seed": 42,
            "decoding_parameters": {
                "temperature": 0.0,
                "seed": 42,
            },
            "cases_evaluated": len(processed_cases),
            "output_directory": args.output_dir,
            "git_commit_sha": "unknown",
            "code_commit_sha": "unknown",
            "prompt_template_hash": prompt_template_hash,
            "parser_version": "PromptTypedRevisionPolicy_v1",
            "response_schema_version": "v1_canonical",
        }
        if args.live:
            manifest["warning"] = "development_live_api_run / not_final_paper_result"

        try:
            res_git = os.popen("git rev-parse HEAD").read().strip()
            if res_git:
                manifest["git_commit_sha"] = res_git
                manifest["code_commit_sha"] = res_git
        except Exception:
            pass

        with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✓ Saved metrics, breakdown csv, and manifest to {output_path}/")

    print("\nEvaluation Completed successfully!")


if __name__ == "__main__":
    main()
