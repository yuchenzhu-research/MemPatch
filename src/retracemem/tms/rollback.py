from __future__ import annotations

from typing import Any
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.tms.authorization import AuthorizationEngine


class RollbackDiagnostics:
    """Computes diagnostic metrics for ReTrace belief authorization and rollback."""

    def __init__(self, store: BeliefStore, ledger: EpisodeLedger) -> None:
        self.store = store
        self.ledger = ledger
        self.engine = AuthorizationEngine(store, ledger)

    def calculate_her(self, total_inputs: int) -> float:
        """Historical Evidence Retention (HER).

        Should be 1.0 since ReTrace ledger is immutable.
        """
        if total_inputs <= 0:
            return 0.0
        return len(self.ledger) / total_inputs

    def calculate_rar_and_urr(
        self,
        ground_truth: dict[str, dict[str, bool]],  # {evidence_id: {belief_id: expected_authorized}}
    ) -> dict[str, float]:
        """Calculates Reauthorization Recovery (RAR) and Unsafe Reactivation (URR_react).

        ground_truth maps each cutoff evidence_id to a dict of expected belief authorization statuses.
        """
        reauthorized_success = 0
        reauthorized_total = 0

        unsafe_reactivations = 0
        unsafe_total = 0

        beliefs = self.store.all_beliefs()
        evidences = self.ledger.all()
        evidence_order = {ev.id: idx for idx, ev in enumerate(evidences)}
        sorted_ev_ids = [ev.id for ev in sorted(evidences, key=lambda e: (e.timestamp or "", evidence_order[e.id]))]

        history: dict[str, list[tuple[str, bool]]] = {b.id: [] for b in beliefs}

        for ev_id in sorted_ev_ids:
            for b in beliefs:
                dec = self.engine.decide(b, at_evidence_id=ev_id)
                history[b.id].append((ev_id, dec.authorized))

        for b_id, states in history.items():
            for i in range(1, len(states)):
                prev_ev, prev_auth = states[i - 1]
                curr_ev, curr_auth = states[i]

                if curr_ev in ground_truth and b_id in ground_truth[curr_ev]:
                    expected_auth = ground_truth[curr_ev][b_id]

                    # Transition from False to True expected
                    if not prev_auth and expected_auth:
                        reauthorized_total += 1
                        if curr_auth:
                            reauthorized_success += 1

                    # Expected to stay False but reactivated
                    elif not prev_auth and not expected_auth:
                        unsafe_total += 1
                        if curr_auth:
                            unsafe_reactivations += 1

        rar = reauthorized_success / reauthorized_total if reauthorized_total > 0 else 1.0
        urr_react = unsafe_reactivations / unsafe_total if unsafe_total > 0 else 0.0

        return {
            "RAR": rar,
            "URR_react": urr_react,
        }
