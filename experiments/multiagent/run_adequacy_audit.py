from __future__ import annotations

import json
import os
import sys

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src"))

from experiments.multiagent.episodes_fc_dev import get_fc_dev_episodes


def main() -> None:
    print("=" * 70)
    print("ACTION-SPACE ADEQUACY AUDIT")
    print("=" * 70)
    
    episodes = get_fc_dev_episodes()
    total = len(episodes)
    
    representable_count = 0
    multi_action_count = 0
    extensions_dist: dict[str, int] = {}
    
    for ep, gold, artifact in episodes:
        if gold.representable_by_core_actions:
            representable_count += 1
        if gold.requires_multi_action:
            multi_action_count += 1
        ext = gold.missing_extension
        extensions_dist[ext] = extensions_dist.get(ext, 0) + 1
        
    print(f"Total episodes audited:      {total}")
    print(f"Representable by core:      {representable_count}/{total} ({representable_count/total:.2%})")
    print(f"Requires multi-action:      {multi_action_count}/{total} ({multi_action_count/total:.2%})")
    print("\nMissing extensions distribution:")
    for ext, count in extensions_dist.items():
        print(f"  - {ext}: {count}")
        
    output_dir = "outputs/ablation_studies"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "adequacy_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_audited": total,
            "representable_by_core_actions_count": representable_count,
            "requires_multi_action_count": multi_action_count,
            "missing_extensions_distribution": extensions_dist,
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\n[+] Saved adequacy audit report to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
