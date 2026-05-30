#!/usr/bin/env python3
"""
SiliconFlow / Industrial API Model Matrix Evaluation.
Compares DirectJudge-API, StageA-Freeform, StageA-Constrained, and StageC-ICL
across various SiliconFlow models (V3, Qwen) on dev_expansion.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import yaml
from pathlib import Path
from typing import Any

from experiments.multiagent.dev_expansion import generate_expanded_episodes
from experiments.multiagent.stagec_policy import (
    ClosedAPIZeroShotProposer,
    ClosedAPIZeroShotConstrainedProposer,
    ClosedAPIICLProposer,
)
from retracemem.providers.http_provider import HTTPLLMProvider
from retracemem.providers.cached_client import CachedLLMClient
from retracemem.cache.jsonl_cache import JSONLCache
from retracemem.multiagent.commit import commit_submission_sequence
from experiments.multiagent.run_stageab_api_eval import (
    render_direct_judge_prompt,
    load_direct_judge_template,
    parse_direct_judge_response,
    _STATUS_MAP_A_TO_COMPARABLE,
)
from experiments.multiagent.contracts import FixedCandidateSubmission

def backoff_retry_generate(client: Any, prompt: str, model_id: str, provider: str, max_retries: int = 3) -> Any:
    """Urllib request wrapper with exponential backoff for rate limits or transit errors."""
    delay = 1.0
    for attempt in range(max_retries):
        trace = client.generate(
            prompt=prompt,
            model_id=model_id,
            provider=provider,
            temperature=0.0,
            seed=42,
        )
        if trace.status == "success":
            return trace
        # Exponential backoff on rate limit or other failures
        print(f"  [Attempt {attempt+1}/{max_retries}] API call failure: {trace.error_message}. Retrying in {delay}s...")
        time.sleep(delay)
        delay *= 2
    # If all failed, return the final failed trace
    return trace

def main() -> None:
    parser = argparse.ArgumentParser(description="SiliconFlow Model Matrix Evaluation Runner")
    parser.add_argument("--config", default="configs/model_matrix.siliconflow.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Dry run prompt checking and mockup metrics calculation")
    parser.add_argument("--max-cases", type=int, default=None, help="Force limit number of cases evaluated")
    parser.add_argument("--api-key", default=None, help="Explicit SILICONFLOW_API_KEY")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    provider = config.get("provider", "siliconflow")
    api_url = config.get("api_url", "https://api.siliconflow.cn/v1/chat/completions")
    models = config.get("models", [])
    methods = config.get("methods", [])
    max_cases = args.max_cases or config.get("max_cases")

    print("=" * 80)
    print("SILICONFLOW INDUSTRIAL API MODEL MATRIX EVALUATOR")
    print("=" * 80)
    print(f"Dry-run Mode: {args.dry_run}")
    print(f"Models: {models}")
    print(f"Methods: {methods}")
    print(f"Max Cases: {max_cases}")
    print("-" * 80)

    # Load Cases
    ep_gold_pairs = generate_expanded_episodes()
    if max_cases is not None:
        ep_gold_pairs = ep_gold_pairs[:max_cases]
    print(f"Loaded {len(ep_gold_pairs)} test episodes from dev_expansion.")

    # API Setup
    api_key = args.api_key or os.getenv("SILICONFLOW_API_KEY")
    if not args.dry_run and not api_key:
        raise ValueError("SILICONFLOW_API_KEY is required for live API runs. Pass via --api-key or environment variable.")

    # Initialize Output Dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "outputs/runs/matrix_eval")) / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "matrix_api_cache.jsonl"
    cache = JSONLCache(str(cache_file))
    http_provider = HTTPLLMProvider(api_key=api_key, base_url=api_url)
    client = CachedLLMClient(cache=cache, provider_client=http_provider)

    direct_judge_template = load_direct_judge_template()

    # Matrix results structure: results[model][method] = { ...metrics }
    matrix_results: dict[str, dict[str, Any]] = {}

    for model in models:
        matrix_results[model] = {}
        for method in methods:
            print(f"\nEvaluating Model: {model} | Method: {method} ...")
            
            correct_beliefs = 0
            total_beliefs = 0
            over_updates = 0
            not_usable_total = 0
            under_updates = 0
            usable_total = 0
            uncertainty_errors = 0
            parser_errors = 0

            # Proposer setup if Stage A/C
            proposer = None
            prov_kind = "mock" if args.dry_run else provider
            
            if method == "StageA-Freeform":
                proposer = ClosedAPIZeroShotProposer(provider_kind=prov_kind, model_id=model, client=client)
            elif method == "StageA-Constrained":
                proposer = ClosedAPIZeroShotConstrainedProposer(provider_kind=prov_kind, model_id=model, client=client)
            elif method == "StageC-ICL":
                # Mock exemplars for SFT/ICL comparison
                proposer = ClosedAPIICLProposer(
                    provider_kind=prov_kind,
                    model_id=model,
                    client=client,
                    allow_fallback_to_zeroshot=True,
                )

            for ep, gold in ep_gold_pairs:
                gold_statuses = gold.gold_snapshot.belief_statuses
                
                # Setup output statuses placeholders
                pred_statuses: dict[str, str] = {}

                # -------------------------------------------------------------
                # Branch 1: Direct Usability Judge (DirectJudge-API)
                # -------------------------------------------------------------
                if method == "DirectJudge-API":
                    canonical_stage_b_final_statuses = {}
                    for sub in ep.submissions:
                        prompt_b = render_direct_judge_prompt(direct_judge_template, sub)
                        parsed_verdicts = []
                        parse_err_b = None

                        if args.dry_run:
                            # Mock exact gold verdict to verify validation pipeline
                            for b in sub.candidate_beliefs:
                                status_raw = gold_statuses.get(b.belief_id, "AUTHORIZED")
                                status_mapped = _STATUS_MAP_A_TO_COMPARABLE.get(status_raw, "UNCERTAIN")
                                parsed_verdicts.append({
                                    "canonical_belief_id": b.belief_id,
                                    "status": status_mapped,
                                    "canonicalization_applied": False,
                                })
                        else:
                            try:
                                trace = backoff_retry_generate(client, prompt_b, model, provider)
                                if trace.status == "success" and trace.response:
                                    valid_belief_ids = {b.belief_id for b in sub.candidate_beliefs} | {
                                        b.belief_id for b in sub.candidate_replacement_beliefs
                                    }
                                    parsed_verdicts = parse_direct_judge_response(trace.response, valid_belief_ids)
                                else:
                                    parse_err_b = trace.error_message
                            except Exception as ex:
                                parse_err_b = str(ex)

                        if parse_err_b:
                            parser_errors += 1
                        else:
                            for verdict in parsed_verdicts:
                                canonical_bid = verdict["canonical_belief_id"]
                                canonical_stage_b_final_statuses[canonical_bid] = verdict["status"]

                    # Map DirectJudge outputs to comparable statuses
                    for bid in gold_statuses:
                        pred_statuses[bid] = canonical_stage_b_final_statuses.get(bid, "UNCERTAIN")

                # -------------------------------------------------------------
                # Branch 2: Edge Proposing + DPA Authorization (Stage A / Stage C)
                # -------------------------------------------------------------
                else:
                    subagent_subs = []
                    for sub in ep.submissions:
                        if args.dry_run:
                            # Generate dynamic mock targets based on gold labels for dry-run
                            mock_targets = []
                            for target in gold.gold_typed_targets:
                                if target.submission_id == sub.submission_id:
                                    mock_targets.append(target)
                            if not mock_targets:
                                from experiments.multiagent.contracts import TypedRevisionTarget
                                mock_targets.append(TypedRevisionTarget(
                                    submission_id=sub.submission_id,
                                    action_type="NO_REVISION",
                                    target_belief_id=None,
                                    target_condition_id=None,
                                    replacement_belief_id=None,
                                    rationale="Mock correct target (NO_REVISION)",
                                    evidence_ids=(sub.new_evidence_id,)
                                ))
                            
                            from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
                            from retracemem.authorization import EvidenceProposalBatch
                            from experiments.multiagent.contracts import ProposalPolicyOutput
                            
                            edges = []
                            for idx_edge, t in enumerate(mock_targets):
                                if t.action_type == "NO_REVISION":
                                    continue
                                target_kind = "belief" if t.target_belief_id else "condition"
                                target_id = t.target_belief_id or t.target_condition_id
                                edges.append(EvidenceEdge(
                                    edge_id=f"edge_policy_ex_{sub.submission_id}_{idx_edge}",
                                    edge_type=EvidenceEdgeType(t.action_type) if hasattr(EvidenceEdgeType, t.action_type) else t.action_type,
                                    evidence_id=str(t.evidence_ids[0]),
                                    target_kind=target_kind,
                                    target_id=target_id,
                                    verifier="stagec_policy",
                                    replacement_belief_id=t.replacement_belief_id,
                                    rationale=t.rationale,
                                ))
                            proposal_batches = ()
                            if edges:
                                proposal_batches = (
                                    EvidenceProposalBatch(edges=tuple(edges), metadata={"parser": "PromptTypedRevisionPolicy"}),
                                )
                            proposal_output = ProposalPolicyOutput(
                                example_id=f"ex_{sub.submission_id}",
                                submission_id=sub.submission_id,
                                policy_variant="mock_gold",
                                proposal_batches=proposal_batches,
                                parsing_valid=True,
                                errors=(),
                                parsed_actions=tuple(mock_targets),
                                metadata={},
                            )
                        else:
                            proposal_output = proposer.propose(sub)

                        if not proposal_output.parsing_valid:
                            parser_errors += 1

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
                            proposal_batches=proposal_output.proposal_batches,
                        )
                        subagent_subs.append(subagent_sub)

                    seq_res = commit_submission_sequence(tuple(subagent_subs), final_snapshot_evaluation=True)
                    pred_statuses = seq_res.final_belief_statuses

                # -------------------------------------------------------------
                # Metrics Calculation
                # -------------------------------------------------------------
                for bid, gold_status in gold_statuses.items():
                    gold_comp = _STATUS_MAP_A_TO_COMPARABLE.get(gold_status, "UNCERTAIN")
                    actual_raw = pred_statuses.get(bid, "UNRESOLVED")
                    actual_comp = _STATUS_MAP_A_TO_COMPARABLE.get(actual_raw, "UNCERTAIN")

                    total_beliefs += 1
                    if actual_comp == gold_comp:
                        correct_beliefs += 1

                    # Over update
                    if gold_comp == "NOT_USABLE":
                        not_usable_total += 1
                        if actual_comp == "USABLE":
                            over_updates += 1

                    # Under update
                    if gold_comp == "USABLE":
                        usable_total += 1
                        if actual_comp != "USABLE":
                            under_updates += 1

                    # Uncertainty error rate
                    if gold_comp == "UNCERTAIN":
                        if actual_comp != "UNCERTAIN":
                            uncertainty_errors += 1
                    else:
                        if actual_comp == "UNCERTAIN":
                            uncertainty_errors += 1

            # Compile global metrics
            def calc_rate(num, den):
                return num / den if den > 0 else 0.0

            matrix_results[model][method] = {
                "dpa_final_status_accuracy": calc_rate(correct_beliefs, total_beliefs),
                "over_update_rate": calc_rate(over_updates, not_usable_total),
                "under_update_rate": calc_rate(under_updates, usable_total),
                "uncertainty_error_rate": calc_rate(uncertainty_errors, total_beliefs),
                "parser_error_rate": calc_rate(parser_errors, len(ep_gold_pairs)),
                "total_beliefs_evaluated": total_beliefs,
            }

            print(f"  Accuracy: {matrix_results[model][method]['dpa_final_status_accuracy']:.4f}")
            print(f"  Over Updates (Stale Propagation): {matrix_results[model][method]['over_update_rate']:.4f}")

    # Write Matrix results
    with open(output_dir / "matrix_results.json", "w", encoding="utf-8") as f:
        json.dump(matrix_results, f, indent=2)

    # Save manifest.json
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_identifier": "development_live_api_run / not_final_paper_result" if not args.dry_run else "development_run",
        "is_live_api_result": not args.dry_run,
        "mock_default_used": args.dry_run,
        "provider": provider,
        "models_evaluated": models,
        "methods_evaluated": methods,
        "output_directory": str(output_dir),
        "results": matrix_results,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Completed matrix evaluation successfully! Report saved to {output_dir}/")

if __name__ == "__main__":
    from retracemem.multiagent.contracts import SubagentMemorySubmission
    main()
