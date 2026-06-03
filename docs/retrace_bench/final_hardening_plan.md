# ReTrace-Bench Final Hardening Plan

This document outlines the implementation steps to upgrade `ReTrace-Bench` to its final AAAI paper-facing release.

## Split Realignment
We migrate from legacy v1.0 structures and produce standard public splits in `data/retrace_bench/`:
- `main_3000_en/`
- `hard_500_en/`
- `realistic_200_en/`
- `calibration_80_en/`
- `private_hidden_200_en/` (for contamination-resistant evaluation)

## GitHub Realistic Patterns (15)
We implement systematic support for:
1. `merged_but_unreleased`
2. `closed_as_duplicate_not_fixed`
3. `docs_ahead_of_code`
4. `release_then_revert`
5. `version_scope_leakage`
6. `branch_scope_leakage`
7. `authority_conflict`
8. `ci_failed_after_claim`
9. `security_policy_override`
10. `backport_only_fix`
11. `maintainer_correction_over_user_claim`
12. `stale_comment_after_new_release`
13. `label_state_mismatch`
14. `multi_memory_coupling`
15. `negative_evidence_required`

## Difficulty Tiers (L1-L4)
- **L1**: Single event, no distractor, simple update.
- **L2**: Distractors, temporal or trust conflict, >=2 evidence events.
- **L3**: Scope/version/authority conflict, >=3 evidence events, >=2 memories, minimal evidence required.
- **L4**: Multi-memory coupling, adversarial distractors, scope/version/authority/release/rollback conflict, negative evidence, no latest-event shortcut.

## Evaluation Metric Enhancements
We extend `scorers_general.py` and `api.py` with:
- `joint_revision_success`
- `minimal_evidence_exact_match`
- `evidence_precision`
- `overcitation_rate`
- `counterevidence_recall`
- `answer_state_consistency`
- `scope_authority_accuracy`
- `latest_event_shortcut_failure_rate`

## Implementation Steps
1. **Mining & Blueprint Logic**: Create `mine_github_workflow_seeds.py` to extract seeds from public repos. Write blueprint generators (`hard_plus_blueprints.py`, `github_realistic_blueprints.py`, `adversarial_distractors.py`, `evidence_dependency_graph.py`).
2. **Generation**: Implement `generate_retrace_bench_final.py` and `render_retrace_bench_final.py` to produce datasets, manifests, and Readmes.
3. **Validation**: Update `validate_retrace_bench_dataset.py` with strict criteria (e.g. no shortcut, no hidden leakage).
4. **Baseline & Packaging**: Re-run offline baseline configurations and package the HF dataset.
5. **Results**: Run baseline prediction and outputs generation scripts.
