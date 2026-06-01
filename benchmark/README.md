# ReTrace-Bench Source Package

This directory contains the source code for the ReTrace-Bench paper experiments and benchmarks.

## Import Convention

All internal packages under this folder should be imported as:
```python
from benchmark.retrace_bench.schemas import ...
from benchmark.retrace_bench.taxonomy import ...
```
This is because ReTrace-Bench is structured under the top-level `benchmark/` folder to remain independent from the ReTrace-Learn SFT/RL training codebase.

## Layout

- `retrace_bench/`: Core python library containing taxonomies, schemas, generation rules, baselines, and scoring metrics.
