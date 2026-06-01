# Mem0 Reference Note

* **Paper Title**: Mem0: The Memory Layer for Personalized AI Agents
* **Venue / arXiv Status**: GitHub Open-Source Community Library (Industrial Reference)
* **Official Links**: https://github.com/mem0ai/mem0
* **GitHub / HuggingFace**: https://github.com/mem0ai/mem0

## Core Details

* **Task Definition**: Storing, organizing, and retrieving personalized user memory facts across different agent sessions.
* **Dataset Size**: N/A (open-source framework, evaluated on custom chat runs).
* **Data Construction**: Real-world user logs and tool inputs.
* **Input/Output Format**: Raw text inputs -> structured key-value memory mutations (add, update, delete).
* **Evaluation Protocol**: Focuses on retrieval consistency and personalized response quality.
* **Metrics**: User satisfaction, latency, recall accuracy.
* **Baseline Suite**: Standard vector RAG, LangChain memory utilities.

## Borrow & Avoid

* **What ReTrace-Bench can borrow**: Industrial application schemas for personal preferences.
* **What ReTrace-Bench should avoid**: Lack of explicit dependency structures. Mem0 lacks prerequisite condition resolution (like `BLOCKS` and `RELEASES`).
