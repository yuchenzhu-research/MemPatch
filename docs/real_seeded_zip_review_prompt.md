# ChatGPT Zip Review Prompt

Please review this MemPatch-Bench real-seeded GitHub mining package.

Focus on:

1. Whether the raw mining quality gate is correctly enforced before filtering, normalization, audit, or model evaluation.
2. Whether the scripts avoid token leakage and read GitHub credentials only from `GITHUB_TOKEN`.
3. Whether public benchmark rows can leak private labels or expected answers.
4. Whether accepted candidates would be paper-grade evidence, not just keyword matches.
5. Whether the audit checks cover forbidden fields, missing evidence IDs, PII/secrets, duplicate sources, and invalid state transitions.

Important context:

- The current local canonical raw file has 0 candidates because unauthenticated GitHub requests were rate-limited.
- Filtering and normalization should not be trusted until the mining quality report passes the raw gate.
- Do not run model evaluation as part of this review.
