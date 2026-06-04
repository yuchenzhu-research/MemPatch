# Benchmark Package

`benchmark/retrace_bench/` contains the ReTrace-Bench evaluator/API code.

Surviving code should be limited to:

- public scoring API
- prediction/schema normalization
- taxonomy
- scorers and aggregate metrics
- public-view sanitization
- dataset/prediction validation helpers
- small utility modules required by those surfaces

Generation, audit, report, pilot, live-runner, and paper-writing code does not belong in this package.
