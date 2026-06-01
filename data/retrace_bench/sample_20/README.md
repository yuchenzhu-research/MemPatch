# ReTrace-Bench Sample 20 Dataset

This directory contains a small canonical sample of ReTrace-Bench (20 scenarios, 80 queries) generated with seed 7.
It is intended for local integration testing, quick verification, and CI runs.

## Contents
- `scenarios.jsonl`: Raw scenario configurations.
- `queries.jsonl`: Generated queries.
- `public_dev.jsonl`: Public development split.
- `public_test.jsonl`: Public test split.
- `private_test_stub.jsonl`: Private test stub.
- `manifest.json`: Dataset metadata.

For the full evaluation suite, refer to the local-only `v1_smoke` dataset or regenerate using the builder script.
