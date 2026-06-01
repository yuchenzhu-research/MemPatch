# BFCL Reference Note

* **Benchmark Name**: Berkeley Function Calling Leaderboard (BFCL)
* **Venue / Publication Status**: Active Live Leaderboard, Gorilla Project (UC Berkeley)
* **Official Repo**: https://github.com/ShishirPatil/gorilla
* **Dataset / HuggingFace Link**: Hosted internally on Gorilla project page.

## Core Details

* **Task Type**: Tool use and function calling accuracy.
* **Scale**: Over 2,000 function-calling scenarios.
* **Data Source**: Real-world REST APIs, SDK definitions, and SQL queries.
* **Splits**: test.
* **Evaluation Harness**: Abstract syntax tree (AST) code checkers comparing function parameters and naming.
* **Prediction Format**: Structured tool call payload (JSON or Python format).
* **Scoring Metrics**: AST Matching Accuracy, Execution correctness.
* **Leaderboard/Private Test Policy**: Live rolling leaderboard, new tests added periodically.
* **Repo Structure**: Gorilla client libraries, eval suites, ast check source.

## Imitate & Avoid

* **What ReTrace-Bench should imitate**: Structuring API response comparison deterministically using AST or JSON payload mapping.
* **What ReTrace-Bench should not imitate**: High dependency on specific API endpoints or configurations. ReTrace-Bench runs locally with mock LLM providers.
