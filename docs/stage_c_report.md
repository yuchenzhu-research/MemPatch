# Stage C Go/No-Go Report: Deferred Learned Local Verifier

**Date**: 2026-05-28
**Status**: **NO-GO** for the current task
**Scope**: Stage C remains deferred pending real Stage A/B evidence and
leakage-safe training-data design.

---

## 1. Decision Summary

Do not begin Stage C in the current feasibility packet. Paper 1 must first
establish whether Stage A (`ReTrace-LLM`: local typed-edge proposal + DPA)
shows a real advantage over Stage B (`DirectJudge-LLM`) under the same fixed
`SharedCandidateView` inputs.

The current repository contains useful scaffolding, but it does not yet contain
verified live Stage A/B performance evidence, official STALE results, official
Memora results, or approved Stage C training labels.

---

## 2. Rationale for Deferral

1. **No verified Stage A/B advantage yet**:
   Stage C is only justified if the structured decomposition has empirical
   value over direct adjudication.

2. **Stage A predictions are not gold labels**:
   Stage A live traces, admitted edges, and DPA outcomes are model-produced
   artifacts. They may be useful for error analysis, but they are not
   ground-truth typed-edge labels.

3. **Official benchmark leakage risk**:
   Official STALE/Memora scored examples, evaluator outputs, judged results,
   and scored-run traces must not be converted into Stage C labels,
   prompt-development material, or training data.

4. **Training is out of scope**:
   This task may repair safety boundaries and run an internal Stage A/B
   feasibility diagnostic. It must not train, distill, fine-tune, or start
   local verifier development.

---

## 3. Prerequisites for Stage C Commencement

Before any future Stage C task, the following conditions must be met:

1. Real Stage A/B evidence shows that structured typed-edge authorization is
   promising.
2. A separately approved data plan uses development-safe sources.
3. Human-audited typed-edge annotations with provenance are available.
4. Contamination checks separate training/development data from official
   evaluation material.
5. Official STALE/Memora final evaluation instances, evaluator outputs, and
   scored-run traces remain excluded from training labels and prompt
   development.
6. Fallback and uncertainty handling rules are specified before any local
   verifier is trusted.

## 4. Prohibited Data Sources for Stage C Labels

- Stage A live outputs treated as gold.
- Official STALE or Memora scored examples.
- Official evaluator feedback or judged outputs.
- Traces from frozen official evaluation runs.
- Synthetic labels derived from official test failures.

## 5. Current Next Step

Complete the internal Ambiguity-and-Scope Stage A/B feasibility diagnostic.
Interpret the result honestly before considering any external pilot, official
evaluation, or Stage C planning.
