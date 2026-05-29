from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List
from experiments.multiagent.select_prompt_smoke_examples import select_smoke_examples

def run_preflight(config_path: str, review_file: str) -> None:
    print("[*] Running Stage C Prompt Smoke Preflight Mode...")
    
    # 1. Load config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    print(f"[*] Loaded configuration from: {config_path}")
    print(json.dumps(config, indent=2))
    
    # 2. Check for approved examples
    # (select_smoke_examples will exit with 1 if no approved examples exist)
    # Under preflight mode, we can catch it to print diagnostic warnings
    try:
        selected = select_smoke_examples(review_file, confirm_live_run=True)
        print(f"[+] Found {len(selected)} approved examples for smoke preflight.")
    except SystemExit:
        print("[!] Preflight Warning: No approved examples are available in review file yet.")
        print("[!] Once human review approves examples, they will be selected.")
        selected = []
        
    # Print dry run manifest planning
    manifest = {
        "run_id_prefix": config.get("run_config", {}).get("run_id_prefix", "stagec_prompt_smoke"),
        "mode": "smoke_live_preflight",
        "provider": config.get("model_config", {}).get("provider"),
        "backbone_model": config.get("model_config", {}).get("backbone_model"),
        "approved_examples_count": len(selected),
        "scientific_status": "smoke_validation_only",
        "created_at": datetime.datetime.now().isoformat(),
    }
    print("[*] Planned Run Manifest:")
    print(json.dumps(manifest, indent=2))
    print("[+] Preflight checks complete. No API calls were made.")


def run_live(config_path: str, review_file: str, confirm_live_run: bool) -> None:
    print("[*] Attempting Stage C Prompt Smoke Live Mode...")
    
    if not confirm_live_run:
        print("Error: Live run requires explicit confirmation flag: --confirm-live-run")
        sys.exit(1)
        
    # 1. Load config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    # 2. Check placeholders
    provider = config.get("model_config", {}).get("provider")
    model = config.get("model_config", {}).get("backbone_model")
    
    if "<select_before_run>" in str(model) or "<openai" in str(provider):
        print(f"Error: Model configuration contains placeholders: provider={provider}, model={model}")
        print("Please replace placeholders in the configuration file before running live.")
        sys.exit(1)
        
    # 3. Retrieve approved examples (this will exit if 0 approved)
    selected = select_smoke_examples(review_file, confirm_live_run=True)
    
    print(f"Error: Live run provider wiring is deferred. Model: {model}. Ready examples: {len(selected)}.")
    sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage C Prompt Smoke Evaluation (Packet 4F-A)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="preflight",
        choices=["preflight", "live"],
        help="Execution mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/multiagent/configs/stagec_prompt_smoke.example.json",
        help="Path to the JSON model/run configuration"
    )
    parser.add_argument(
        "--review-file",
        type=str,
        default="outputs/stagec_dev_review_queue_70.jsonl",
        help="Path to the review JSONL queue file"
    )
    parser.add_argument(
        "--confirm-live-run",
        action="store_true",
        help="Explicitly confirm this live execution"
    )
    args = parser.parse_args()
    
    if args.mode == "preflight":
        run_preflight(args.config, args.review_file)
    elif args.mode == "live":
        run_live(args.config, args.review_file, args.confirm_live_run)
        
if __name__ == "__main__":
    main()
