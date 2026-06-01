# OSWorld Reference Note

* **Benchmark Name**: OSWorld: Benchmarking Multimodal Agents on OS Tasks
* **Venue / Publication Status**: Peer-reviewed, NeurIPS 2024
* **Official Repo**: https://github.com/OS-World/OSWorld
* **Dataset / HuggingFace Link**: https://huggingface.co/datasets/OS-World/OSWorld

## Core Details

* **Task Type**: Operating system tasks (Linux, desktop, file manipulation).
* **Scale**: 369 desktop tasks.
* **Data Source**: Manually created scenario scripts simulating OS user goals.
* **Splits**: test.
* **Evaluation Harness**: Running agents inside VM screenshots/VNC loops, checking final state files.
* **Prediction Format**: VNC interface action trajectory (bash command, clicks).
* **Scoring Metrics**: Success rate based on deterministic checker scripts.
* **Leaderboard/Private Test Policy**: Community-driven submission.
* **Repo Structure**: VM drivers, environment setup scripts, baseline models, checkers.

## Imitate & Avoid

* **What ReTrace-Bench should imitate**: Writing robust, localized deterministic state checkers.
* **What ReTrace-Bench should not imitate**: The VM framework overhead. ReTrace-Bench should test memory state and graph structures rather than UI pixels.
