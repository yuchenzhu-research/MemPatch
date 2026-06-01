# Contamination Policy

To ensure absolute clean methodology and avoid test-set leakage:

1. **Strict Data Segregation**:
   - `data/retrace_bench/` is strictly **evaluation-only** and must never be used for ReTrace-Learn training.
   - `data/retrace_learn/` contains internal method-training and SFT data.

2. **Loader Guardrails**:
   - Any data loaders running SFT, DPO, or reinforcement learning in this repository must call the contamination guard to immediately reject file paths containing `data/retrace_bench`.

3. **Leaderboard & Reporting**:
   - Benchmark reports and leaderboard submissions must declare if any evaluated policy was tuned on the public dev set.
   - Private test labels (`private_test_stub.jsonl` placeholders) remain hidden from general evaluation loops.
