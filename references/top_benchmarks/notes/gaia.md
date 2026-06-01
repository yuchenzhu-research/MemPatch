# GAIA Reference Note

* **Benchmark Name**: GAIA: A Benchmark for General AI Assistants
* **Venue / Publication Status**: Peer-reviewed, ICLR 2024
* **Official Repo**: https://github.com/huggingface/gaia
* **Dataset / HuggingFace Link**: https://huggingface.co/datasets/gaia-benchmark/GAIA

## Core Details

* **Task Type**: General multi-modal tasks (web search, data extraction, file conversion).
* **Scale**: 466 task instances.
* **Data Source**: Manually designed realistic assistant queries.
* **Splits**: validation, test.
* **Evaluation Harness**: Deterministic string verification of agent outputs.
* **Prediction Format**: Short string output.
* **Scoring Metrics**: Accuracy (exact matching).
* **Leaderboard/Private Test Policy**: Hidden test labels, public validation labels.
* **Repo Structure**: Validation check scripts, dataset folders.

## Imitate & Avoid

* **What ReTrace-Bench should imitate**: The strict exact matching design for evaluation queries to avoid flaky LLM judges.
* **What ReTrace-Bench should not imitate**: High percentage of manual data generation. ReTrace-Bench should scale automatically via programmatic seed expansion.
