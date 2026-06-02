# MemConflict Reference Note

* **Paper Title**: MemConflict: Resolving Belief and Memory Conflict in Long-Term LLM Interactions
* **Venue / arXiv Status**: unverified pointer
* **Official Links**: pending verification
* **GitHub / HuggingFace**: pending verification

## Core Details

* **Task Definition**: Evaluating how models resolve direct contradictions between user preferences or facts stated across different turns.
* **Dataset Size**: ~1,500 test cases.
* **Data Construction**: Programmatic synthesis of dialogues with explicit preference conflicts (e.g. "prefers tea" vs "prefers coffee").
* **Input/Output Format**: Multi-session dialogue transcript -> resolved preference decision.
* **Evaluation Protocol**: Probing model answers on resolved state queries.
* **Metrics**: Consistency Rate, Conflict Overlap Score.
* **Baseline Suite**: Direct prompting, agent debate, voting.

## Borrow & Avoid

* **What ReTrace-Bench can borrow**: Categorization of belief conflicts (explicit contradiction, obsolete information).
* **What ReTrace-Bench should avoid**: Simple voting or debate as the resolution mechanism. ReTrace-Bench mandates deterministic DPA rules.
