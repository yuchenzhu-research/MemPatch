# ReTrace Source Layout

```text
src/
├── retrace_learn/  # learned ReTrace-Learn components
└── retracemem/     # deterministic Authorization Court / ReTrace-Engine
```

- **`retrace_learn`**: Contains learned components such as Graph Extractor, Typed Revision Proposer, training/SFT pipelines, and baseline APIs.
- **`retracemem`**: Contains the deterministic Authorization Court (ReTrace-Engine) executing the DPA algorithm. It is not separate from the method paper; it represents the third, deterministic stage of the ReTrace-Learn pipeline.
