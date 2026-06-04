from __future__ import annotations

from typing import Sequence

from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.schemas import AuthorizationStatus, AuthorizationTrace
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm


class RollbackDiagnostics:
    """Research diagnostics over the typed-graph DPA runtime.

    Diagnostics are computed from
    ``AuthorizationTrace.status`` produced by
    ``DefeatPathAuthorizationAlgorithm``.

    Three diagnostics are exposed:

    - ``calculate_her(total_inputs)`` -- Historical Evidence Retention.
      The ratio of ledger size to total ingested inputs. Should be 1.0
      because the ``EpisodeLedger`` is append-only.
    - ``reauthorization_history(belief_id, cutoffs)`` -- per-belief
      sequence of ``AuthorizationTrace`` snapshots across a list of
      cutoff evidence ids. Useful for plotting belief eligibility over
      time without coupling to a particular metric.
    - ``calculate_rar_and_urr(ground_truth)`` -- Reauthorization
      Recovery (RAR) and Unsafe Reactivation (URR_react), defined in
      terms of eligibility (``AuthorizationStatus.AUTHORIZED``). Ground
      truth is keyed by cutoff evidence id and then belief id, the same
      shape used by the development boundary-audit test data.

    All three metrics use only typed-graph state; no legacy flat-relation
    logic is consulted, and no mock ledger is ever fabricated.
    """

    def __init__(self, store: BeliefStore, ledger: EpisodeLedger) -> None:
        self.store = store
        self.ledger = ledger
        self.algorithm = DefeatPathAuthorizationAlgorithm(store, ledger)

    # ------------------------------------------------------------------
    # HER: Historical Evidence Retention
    # ------------------------------------------------------------------

    def calculate_her(self, total_inputs: int) -> float:
        """Returns ``len(ledger) / total_inputs``; ``0.0`` if ``total_inputs <= 0``."""
        if total_inputs <= 0:
            return 0.0
        return len(self.ledger) / total_inputs

    # ------------------------------------------------------------------
    # Per-belief authorization history across cutoffs
    # ------------------------------------------------------------------

    def reauthorization_history(
        self,
        belief_id: str,
        cutoff_evidence_ids: Sequence[str],
    ) -> list[tuple[str, AuthorizationTrace]]:
        """For each cutoff, return the ``AuthorizationTrace`` of ``belief_id``.

        The cutoffs are processed in the order given; no sorting is
        performed because callers may legitimately want to ask about
        non-monotonic time points (for example, "before and after a
        rollback experiment").
        """
        history: list[tuple[str, AuthorizationTrace]] = []
        for cutoff in cutoff_evidence_ids:
            trace = self.algorithm.authorize(belief_id, as_of_evidence_id=cutoff)
            history.append((cutoff, trace))
        return history

    # ------------------------------------------------------------------
    # RAR + URR_react
    # ------------------------------------------------------------------

    def calculate_rar_and_urr(
        self,
        ground_truth: dict[str, dict[str, bool]],
    ) -> dict[str, float]:
        """Compute RAR and URR_react over all (cutoff, belief) transitions.

        ``ground_truth`` maps ``cutoff_evidence_id -> {belief_id: expected_eligible}``.
        Only beliefs that are present in the store are considered.

        - **RAR** (Reauthorization Recovery): fraction of (False -> True
          expected) transitions where the algorithm did re-authorize.
        - **URR_react** (Unsafe Reactivation): fraction of (False ->
          False expected) transitions where the algorithm nevertheless
          re-authorized.

        ``True`` here means ``AuthorizationStatus.AUTHORIZED``; any
        other status (BLOCKED, SUPERSEDED, UNRESOLVED) is treated as
        not-eligible.
        """
        reauthorized_success = 0
        reauthorized_total = 0
        unsafe_reactivations = 0
        unsafe_total = 0

        beliefs = self.store.all_beliefs()

        # Deterministic processing order: ledger insertion order for
        # evidence atoms.
        sorted_ev_ids = [ev.evidence_id for ev in self.ledger.all()]

        for belief in beliefs:
            states: list[tuple[str, bool]] = []
            for ev_id in sorted_ev_ids:
                trace = self.algorithm.authorize(belief.belief_id, as_of_evidence_id=ev_id)
                states.append((ev_id, trace.status == AuthorizationStatus.AUTHORIZED))

            for i in range(1, len(states)):
                _, prev_auth = states[i - 1]
                curr_ev, curr_auth = states[i]

                expected_table = ground_truth.get(curr_ev)
                if expected_table is None:
                    continue
                if belief.belief_id not in expected_table:
                    continue
                expected_auth = expected_table[belief.belief_id]

                if not prev_auth and expected_auth:
                    reauthorized_total += 1
                    if curr_auth:
                        reauthorized_success += 1
                elif not prev_auth and not expected_auth:
                    unsafe_total += 1
                    if curr_auth:
                        unsafe_reactivations += 1

        rar = reauthorized_success / reauthorized_total if reauthorized_total > 0 else 1.0
        urr_react = unsafe_reactivations / unsafe_total if unsafe_total > 0 else 0.0
        return {"RAR": rar, "URR_react": urr_react}
