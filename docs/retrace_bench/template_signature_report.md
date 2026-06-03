# ReTrace-Bench Template Signature Report

> **Legacy (pre-v1.0) document.** Describes a legacy pre-v1.0 split/pilot, recoverable from the Git tag `legacy-retrace-bench-pre-v1.0`. Retained for provenance only; it does **not** describe the ReTrace-Bench v1.0 splits (`main`/`hard`/`realistic`/`calibration`).


This diagnostic de-identifies scenario text before comparing templates. It normalizes scenario IDs, case/project/person/workspace IDs, memory/event IDs, timestamps, numeric counters, and split prefixes.

The existing `test_800_en` is treated as prototype/diagnostic. The new `test_800_templateheldout_en` is the candidate paper-facing held-out split.

## Split Summary

| split | scenarios | event-text templates | workflow-context templates | scenario signatures |
| --- | ---: | ---: | ---: | ---: |
| train | 3000 | 178 | 40 | 2280 |
| dev | 400 | 174 | 40 | 391 |
| test | 800 | 1910 | 32 | 800 |
| prototype_test | 800 | 178 | 40 | 776 |

## Signature Overlap

| comparison | overlap count | percent of test signatures |
| --- | ---: | ---: |
| train∩test | 0 | 0.00% |
| dev∩test | 0 | 0.00% |
| train∩prototype_test | 0 | 0.00% |
| dev∩prototype_test | 0 | 0.00% |

## Examples: train∩test

No overlapping scenario signatures found.

## Examples: dev∩test

No overlapping scenario signatures found.

## Examples: train∩prototype_test

No overlapping scenario signatures found.

## Examples: dev∩prototype_test

No overlapping scenario signatures found.
