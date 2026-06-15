"""Audit critical pairs and confluence for Reference Transition Semantics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from mempatch.reference_semantics.rules import RULES, TransitionRule
from mempatch.reference_semantics.states import Configuration, MemoryState
from mempatch.reference_semantics.transition import ReferenceTransitionEngine

def simulate_with_order(r1: TransitionRule, r2: TransitionRule, order: list[TransitionRule], mids: dict[str, str]) -> dict[str, str]:
    # Initialize configuration
    memories = {}
    for role, mid in mids.items():
        status = "out_of_scope" if role == "distractor" else "current"
        memories[mid] = MemoryState(memory_id=mid, status=status, text="init", scope="stable")
        
    config = Configuration(memories=memories, t=0)
    engine = ReferenceTransitionEngine(config)
    
    # Render mock events matching each rule
    for idx, rule in enumerate(order):
        # We craft a mock event text containing the rule_id to guarantee match
        mock_event = {
            "event_id": f"e-mock-{idx}",
            "timestamp_order": idx + 1,
            "text": f"This event triggers [trigger:{rule.rule_id}]"
        }
        engine.config = engine.step(mock_event)
        
    return engine.config.get_status_dict()

def check_confluence(r1: TransitionRule, r2: TransitionRule, mids: dict[str, str]) -> tuple[bool, dict[str, Any]]:
    # Path A: r1 then r2
    state_a = simulate_with_order(r1, r2, [r1, r2], mids)
    # Path B: r2 then r1
    state_b = simulate_with_order(r1, r2, [r2, r1], mids)
    
    joinable = (state_a == state_b)
    details = {
        "rule_1": r1.rule_id,
        "rule_2": r2.rule_id,
        "order_1_2": state_a,
        "order_2_1": state_b,
        "joinable": joinable
    }
    return joinable, details

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit critical pairs and confluence.")
    parser.add_argument("--rules", required=True, help="YAML rules config path")
    parser.add_argument("--output", required=True, help="JSON output file path")
    args = parser.parse_args()
    
    rules_path = Path(args.rules)
    output_path = Path(args.output)
    
    if not rules_path.exists():
        print(f"Error: rules file {rules_path} does not exist", file=sys.stderr)
        sys.exit(1)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load rules from config (for audit verification)
    with rules_path.open("r") as f:
        config_data = yaml.safe_load(f)
    config_rules = config_data.get("rules", [])
    
    # Synthetic order-swap audit over every unordered pair of rules. This is
    # implementation stress coverage, not a critical-overlap confluence proof.
    mids = {
        "target": "m-target",
        "condition": "m-condition",
        "distractor": "m-distractor"
    }
    
    pairwise_order_audits = []
    non_joinable = []
    joinable_count = 0
    total_pairs = 0
    
    # Compare all pairs (r1, r2) where r1 != r2
    for i in range(len(RULES)):
        for j in range(i + 1, len(RULES)):
            r1 = RULES[i]
            r2 = RULES[j]
            
            # Since both can be triggered, do they converge?
            joinable, details = check_confluence(r1, r2, mids)
            total_pairs += 1
            pairwise_order_audits.append(details)
            
            if joinable:
                joinable_count += 1
            else:
                non_joinable.append(details)
                
    # Keep the caller-selected output path for backward compatibility.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pairwise_order_audits, f, indent=2, ensure_ascii=False)
        
    # Write summary
    summary_path = output_path.parent / "critical_pair_summary.json"
    summary = {
        "total_rules": len(RULES),
        "total_unordered_rule_pairs": total_pairs,
        "order_invariant_pairs_under_synthetic_audit": joinable_count,
        "order_sensitive_pairs_under_synthetic_audit": len(non_joinable),
        "all_synthetic_order_swaps_agree": (len(non_joinable) == 0),
        "claim_scope": "synthetic pairwise order-swap audit; not critical-pair confluence"
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    # Write non-joinable JSONL
    non_joinable_path = output_path.parent / "non_joinable_pairs.jsonl"
    with non_joinable_path.open("w", encoding="utf-8") as f:
        for item in non_joinable:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("\n--- Synthetic Rule-Pair Order Audit Summary ---")
    print(f"Unordered rule pairs analyzed: {total_pairs}")
    print(f"Order-invariant pairs: {joinable_count} ({joinable_count/total_pairs*100:.2f}%)")
    print(f"Order-sensitive pairs: {len(non_joinable)} ({len(non_joinable)/total_pairs*100:.2f}%)")
    print(f"All synthetic order swaps agree: {summary['all_synthetic_order_swaps_agree']}")

if __name__ == "__main__":
    main()
