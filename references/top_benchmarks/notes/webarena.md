# WebArena Reference Note

* **Benchmark Name**: WebArena: A Realistic Web Environment for Web Agents
* **Venue / Publication Status**: Peer-reviewed, NeurIPS 2023
* **Official Repo**: https://github.com/web-arena-x/webarena
* **Dataset / HuggingFace Link**: Hosted locally / locally deployed environment setup.

## Core Details

* **Task Type**: Web-based instruction following and web navigation.
* **Scale**: 812 task instances.
* **Data Source**: Synthesized and curated user queries over 4 mock web platforms.
* **Splits**: dev, test.
* **Evaluation Harness**: Interactive browser simulator tracking database states and network actions.
* **Prediction Format**: Sequential trajectory of keyboard/mouse clicks and page queries.
* **Scoring Metrics**: Success Rate (SR) based on goal state verification.
* **Leaderboard/Private Test Policy**: Public leaderboard using submitted execution traces.
* **Repo Structure**: Environment configs, playwright browser drivers, task parser, evaluation checkers.

## Imitate & Avoid

* **What ReTrace-Bench should imitate**: End-to-end task flow testing and verifying state changes directly.
* **What ReTrace-Bench should not imitate**: High environment orchestration complexity. WebArena is hard to install locally due to hosting separate web servers.
