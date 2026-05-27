"""Gate-unit tests for the Wave 1A typed graph DPA core.

All tests in this package must run with verifiers disabled: typed nodes and
typed edges are injected directly, then `DefeatPathAuthorizationAlgorithm`
or `RevisionGate` is exercised against the resulting graph. No prompt-based
verifier, no heuristic verifier, no external API call may appear here.
"""
