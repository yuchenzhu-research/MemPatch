# BigCodeBench Reference Note

* **Benchmark Name**: BigCodeBench: Benchmarking Tool-Use Capabilities of Code Agents
* **Venue / Publication Status**: unverified pointer
* **Official Repo**: pending verification
* **Dataset / HuggingFace Link**: pending verification

## Core Details

* **Task Type**: Code generation using diverse programming libraries.
* **Scale**: 1,140 task instances.
* **Data Source**: Synthesized challenges checking real python packages.
* **Splits**: test.
* **Evaluation Harness**: Sandbox execution of generated python code with unit tests.
* **Prediction Format**: Python function snippet.
* **Scoring Metrics**: pass@1 (functional correctness).
* **Leaderboard/Private Test Policy**: Public leaderboard with execution logs.
* **Repo Structure**: Dataset, evaluator, sandbox, runner.

## Imitate & Avoid

* **What ReTrace-Bench should imitate**: Providing clear prediction structures with functional evaluation tests.
* **What ReTrace-Bench should not imitate**: High dependency on third-party python packages. ReTrace-Bench should minimize environment dependency issues.
