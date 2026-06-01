# AgentBench Reference Note

* **Benchmark Name**: AgentBench: Evaluating Language Models as Agents
* **Venue / Publication Status**: Peer-reviewed, ICLR 2024
* **Official Repo**: https://github.com/THUDM/AgentBench
* **Dataset / HuggingFace Link**: https://huggingface.co/datasets/THUDM/AgentBench

## Core Details

* **Task Type**: Multi-environment agent actions (OS, Web, DB, games).
* **Scale**: 8 sub-environments, hundreds of tasks.
* **Data Source**: Recreated logic games, web datasets, bash traces.
* **Splits**: dev, test.
* **Evaluation Harness**: Sandbox interactive environments accessed via API loops.
* **Prediction Format**: Text actions or SQL/Bash calls.
* **Scoring Metrics**: Success rate, average progress, code matching.
* **Leaderboard/Private Test Policy**: Public data, local evaluation runs.
* **Repo Structure**: Environment clients, unified orchestration interface, agent APIs.

## Imitate & Avoid

* **What ReTrace-Bench should imitate**: A unified command runner that executes different baseline types seamlessly.
* **What ReTrace-Bench should not imitate**: High dependency sprawl. Maintaining connections to multiple environment databases can cause stability issues.
