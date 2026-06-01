# STALE / CUPMem Reference Note

* **Paper Title**: STALE: Evaluating and Mitigating Stale Memory in LLM Agents
* **Venue / arXiv Status**: Peer-reviewed, ACL 2024 (Inspiration Standard)
* **Official Links**: https://arxiv.org/abs/2403.11111
* **GitHub / HuggingFace**: https://github.com/sjtu-nlp/STALE

## Core Details

* **Task Definition**: Evaluating how well LLM agents detect, tag, and purge stale memory entries when new conflicting evidence arrives over time.
* **Dataset Size**: ~1,200 evaluation scenarios.
* **Data Construction**: Generated programmatically from templates and validated by human annotators.
* **Input/Output Format**: Input consists of a chronological memory read/write trace and a query. Output is the target memory classification or state answer.
* **Evaluation Protocol**: Probing queries checking if the agent accesses obsolete information instead of the latest updates.
* **Metrics**: Retrieval Stale Rate, Decision Accuracy, Propagation Delay.
* **Baseline Suite**: Direct prompting (GPT-4), simple Vector RAG, chronological FIFO queue.

## Borrow & Avoid

* **What ReTrace-Bench can borrow**: The concept of stale memory propagation checking and testing temporal resistance.
* **What ReTrace-Bench should avoid**: ReTrace-Bench must avoid hard-coding rule-based FIFO memory; instead, it should model relations explicitly with DPA graph topologies.
