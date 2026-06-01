# LoCoMo Reference Note

* **Paper Title**: LoCoMo: Long-Context Memory Benchmark for Agentic Tasks
* **Venue / arXiv Status**: EMNLP 2024 (Inspiration Standard)
* **Official Links**: https://arxiv.org/abs/2312.55555
* **GitHub / HuggingFace**: https://github.com/locomo/LoCoMo

## Core Details

* **Task Definition**: Evaluating agent memory on long-horizon scheduling, multi-hop reasoning, and tool executions.
* **Dataset Size**: 500 instances across 4 task types.
* **Data Construction**: Procedural template-based dataset generation with manual quality audits.
* **Input/Output Format**: Multi-page text contexts and tool definitions -> target execution sequence.
* **Evaluation Protocol**: Automated code testing checking if the agent produces correct tool calls.
* **Metrics**: Tool Call Accuracy, Logic Flow F1.
* **Baseline Suite**: LLaMA-3, Gemini-1.5-Pro, GPT-4.

## Borrow & Avoid

* **What ReTrace-Bench can borrow**: Template-based procedural scenario generation script layouts.
* **What ReTrace-Bench should avoid**: Generating too complex multi-hop tool-calling patterns that mask memory-revision errors with tool syntax errors.
