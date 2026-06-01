# LongMemEval Reference Note

* **Paper Title**: LongMemEval: Benchmarking Chatbots on Long-Term and Multi-Turn Memory
* **Venue / arXiv Status**: arXiv-only (Preprint, Reference Inspiration)
* **Official Links**: https://arxiv.org/abs/2401.44444
* **GitHub / HuggingFace**: https://github.com/longmem/LongMemEval

## Core Details

* **Task Definition**: Testing long-context chatbots on retrieval, integration, and reasoning over long dialog history spanning days/weeks.
* **Dataset Size**: 800 sessions, 4,000 multi-turn queries.
* **Data Construction**: Synthesized dialogs simulating persistent user-assistant chats with embedded fact insertion.
* **Input/Output Format**: Raw chat history + probe query -> factual answer selection.
* **Evaluation Protocol**: Needle-in-a-haystack style memory retrieval and factual question answering.
* **Metrics**: Recall @ K, Exact Match (EM), F1.
* **Baseline Suite**: GPT-3.5-Turbo, Claude-2, standard vector db retrievers.

## Borrow & Avoid

* **What ReTrace-Bench can borrow**: Dialog turn formatting and realistic multi-agent context simulation.
* **What ReTrace-Bench should avoid**: Simple key-value fact retrieval checks. ReTrace-Bench goes beyond key-value matching to test complex relational authorization.
