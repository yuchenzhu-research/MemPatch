# Quality Gates v2

To prevent dataset contamination, leakage, and synthetic artifacts, ReTrace-Bench v2 implements a staged series of Quality Gates.

## Staged Quality Gates

### Phase 1: sample_20_v2 (Toy Validation Smoke Gate)
- **Schema Validation**: All scenarios must validate successfully against the v2 schema rules.
- **Baseline Run**: The offline toy baseline must execute without errors.
- **Offline Limits**: Zero external API calls during validation.
- **No Hidden Gold**: Private hidden evaluation labels must **never** be committed.

### Phase 2: seed_100 (Core Human-Audited Gate)
To promote seed_100 to the official repository, baseline scores must meet these limits:
- **TaskSuccess**: Minimum of `0.40` on baseline models (verifies the task is solvable).
- **EvidenceGroundingF1**: Minimum of `0.40`.
- **Max StaleReuseRate**: At most `0.25`.
- **MemoryStateAccuracy**: Minimum of `0.70`.
- **Policy/Scope Leakage**: Maximum of `0.10`.
- **Human Curation**: Every scenario must have a recorded human review signature.

### Phase 3: alpha_1000 and full_2500 (Scaling Gate)
- Stricter thresholds on leakage, noise, and baseline divergence.
- Explicit checks for demographic and domain-wise balance.

---

## Difficulty Adjustment Rule

> [!WARNING]
> **Divergence Rule**: If simple baseline methods (such as `latest-only` or `retrieve-all`) achieve high success rates (e.g. >80% accuracy) on a candidate version, the benchmark is too trivial. Do **not** scale the dataset.
> **Remediation**:
> - Add longer-range dependencies (greater step separation between initial state and updates).
> - Introduce decoy facts and irrelevant updates to trigger over-update and over-forgetting.
> - Add complex policy boundary contexts.
> - Incorporate more subtle implicit conflicts and multi-step audit grounding requirements.
