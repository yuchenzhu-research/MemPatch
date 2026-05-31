#!/usr/bin/env python3
"""Build a Stage A vs Stage B failure-analysis report from a finished run dir.

Two-step workflow (Stage A and Stage B are independent methods, not a pipeline;
the runner evaluates both jointly on the same fixed-candidate cases):

    # 1) produce a run directory (live or offline)
    python3 experiments/multiagent/run_stageab_api_eval.py --live \
        --provider siliconflow --model deepseek-ai/DeepSeek-V3 \
        --constrained --max-cases 8 --output-dir outputs/runs/<run_id>

    # 2) attribute Stage A failures and emit the analysis artifacts
    python3 scripts/build_failure_analysis.py --run-dir outputs/runs/<run_id>

Writes into the run dir:
    failure_analysis.csv
    failure_analysis.md
    failure_analysis_manifest.json   (commit/branch/provider/.../env-var-name-only)

SECURITY: this script never reads, prints, or stores an API key value. The
manifest records only the *name* of the key env var and a boolean presence flag.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from retracemem.evaluation.multiagent.cases import load_eval_cases
from retracemem.evaluation.multiagent.failure_analysis import (
    BeliefAnalysis,
    build_failure_rows,
    build_manifest,
    rows_to_csv,
    rows_to_markdown,
)

DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def reconstruct_stage_b(b_record: dict) -> dict[str, dict[str, str]]:
    """Reconstruct cumulative per-belief Stage B statuses from a parsed B record.

    Mirrors the DirectJudge cumulative "last write wins" rule: every verdict
    updates the canonical map; strict only updates when no canonicalization was
    applied. Returns {belief_id: {"raw", "canonical", "strict"}}.
    """
    canonical: dict[str, str] = {}
    strict: dict[str, str] = {}
    raw: dict[str, str] = {}
    for sub in b_record.get("submissions", []):
        for v in sub.get("verdicts", []):
            cbid = v["canonical_belief_id"]
            status = v["status"]
            canonical[cbid] = status
            raw[v.get("raw_belief_id", cbid)] = status
            if not v.get("canonicalization_applied"):
                strict[cbid] = status
    merged: dict[str, dict[str, str]] = {}
    for bid in set(canonical) | set(strict) | set(raw):
        merged[bid] = {
            "raw": raw.get(bid, canonical.get(bid, "")),
            "canonical": canonical.get(bid, ""),
            "strict": strict.get(bid, ""),
        }
    return merged


def build_belief_analyses(run_dir: Path, dataset: str) -> list[BeliefAnalysis]:
    """Assemble per-(episode, belief) analysis inputs from run-dir artifacts + gold."""
    a_records = {r["episode_id"]: r for r in _read_jsonl(run_dir / "stage_a_parsed.jsonl")}
    b_records = {r["episode_id"]: r for r in _read_jsonl(run_dir / "stage_b_parsed.jsonl")}
    dpa_records = {r["episode_id"]: r for r in _read_jsonl(run_dir / "dpa_traces.jsonl")}

    gold_by_ep: dict[str, dict[str, str]] = {}
    for ep, gold in load_eval_cases(dataset=dataset):
        gold_by_ep[ep.episode_id] = dict(gold.gold_snapshot.belief_statuses)

    beliefs: list[BeliefAnalysis] = []
    for idx, (ep_id, gold_statuses) in enumerate(
        sorted((e, gold_by_ep[e]) for e in gold_by_ep if e in a_records)
    ):
        a_rec = a_records.get(ep_id, {})
        b_rec = b_records.get(ep_id, {})
        dpa_rec = dpa_records.get(ep_id, {})

        a_final = a_rec.get("final_belief_statuses", {})
        a_parse_error = any(s.get("parse_error") for s in a_rec.get("submissions", []))
        gate_decisions = dpa_rec.get("dpa_trace", {}).get("gate_decisions", [])
        b_statuses = reconstruct_stage_b(b_rec)

        # Collect Stage A actions targeting each belief.
        actions_by_belief: dict[str, list[dict]] = {}
        for sub in a_rec.get("submissions", []):
            for act in sub.get("actions", []):
                for key in ("target_belief_id", "replacement_belief_id"):
                    tgt = act.get(key)
                    if tgt:
                        actions_by_belief.setdefault(tgt, []).append(act)

        for bid, gold_status in sorted(gold_statuses.items()):
            b_st = b_statuses.get(bid, {})
            beliefs.append(BeliefAnalysis(
                case_id=f"case_{idx:03d}",
                episode_id=ep_id,
                belief_id=bid,
                gold_status=gold_status,
                a_status=a_final.get(bid),
                b_raw_verdict=b_st.get("raw") or None,
                b_canonical_verdict=b_st.get("canonical") or None,
                b_strict_verdict=b_st.get("strict") or None,
                a_actions=tuple(actions_by_belief.get(bid, [])),
                a_gate_decisions=tuple(d for d in gate_decisions if d.get("target_id") == bid),
                a_parse_error=a_parse_error and bid not in a_final,
            ))
    return beliefs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Finished Stage A/B run output directory.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Name of the API key env var (value never read).")
    parser.add_argument("--dataset", default=None, help="Override dataset; defaults to the run manifest's dataset_name.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    run_manifest = {}
    rm_path = run_dir / "manifest.json"
    if rm_path.exists():
        run_manifest = json.loads(rm_path.read_text())

    dataset = args.dataset or run_manifest.get("dataset_name", "dev_expansion")
    beliefs = build_belief_analyses(run_dir, dataset)
    rows = build_failure_rows(beliefs)

    provider_name = run_manifest.get("provider", "unknown")
    provider_mode = run_manifest.get("provider_mode") or run_manifest.get("mode")
    if not provider_mode:
        # Best-effort resolution from the registry so the manifest is informative
        # for live runs that recorded only the provider name.
        try:
            from retracemem.providers import provider_config_from_registry
            cfg = provider_config_from_registry(provider_name)
            provider_mode = cfg.mode if cfg is not None else "unknown"
        except Exception:
            provider_mode = "unknown"

    manifest = build_manifest(
        git_commit=run_manifest.get("git_commit_sha", _git("rev-parse", "HEAD")),
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        provider=provider_name,
        provider_mode=provider_mode,
        model=run_manifest.get("model", "unknown"),
        temperature=run_manifest.get("temperature", 0.0),
        constrained=run_manifest.get("constrained", False),
        cache_enabled=True,
        prompt_version=run_manifest.get("prompt_template_hash", "unknown"),
        data_split=dataset,
        max_cases=run_manifest.get("cases_evaluated"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        api_key_env=args.api_key_env,
        api_key_present=bool(os.getenv(args.api_key_env)),
        live_api_run=bool(run_manifest.get("is_live_api_result", False)),
    )

    (run_dir / "failure_analysis.csv").write_text(rows_to_csv(rows))
    (run_dir / "failure_analysis.md").write_text(rows_to_markdown(rows, manifest=manifest))
    (run_dir / "failure_analysis_manifest.json").write_text(json.dumps(manifest, indent=2))

    a_correct = sum(1 for r in rows if r["a_correct"])
    b_correct = sum(1 for r in rows if r["b_correct"])
    print(f"Wrote failure analysis for {len(rows)} beliefs to {run_dir}")
    print(f"  Stage A correct: {a_correct}/{len(rows)} | Stage B correct: {b_correct}/{len(rows)}")
    print(f"  live_api_run={manifest['live_api_run']} api_key_present={manifest['api_key_present']}")


if __name__ == "__main__":
    main()
