"""C36 pure core: correct the transfer/all-four contradiction that stopped C35.

C35 stopped at its fifth side control because two frozen clarifications are mutually
unsatisfiable: clarifications35_01 places the whole `transfer` phase in the all-four
support regime, requiring strictly positive measured load on every wheel at every native
substep, while clarifications35_03 ramps the intended swing pair's allocator weight to zero
inside that same 0.06 s transfer. The zero crossing of the swing-pair load therefore always
falls inside transfer. See docs/ad_lateral_pair_feasibility_20261004.md.

The correction here is a semantics fix, NOT a relaxation of a physical limit. Transfer is
split at the point where the design has already committed to unloading the pair:

    all-four regime      while the commanded swing-pair weight is at least 0.5
    stance-pair regime   below that, identical in strictness to the frozen air regime
                         (each stance wheel positive load, stance-pair sum >= 0.5 m g,
                          no flight, every pose/velocity/torque/collision gate unchanged)

Nothing else changes. The frozen native guard code is reused byte for byte; it switches
regime purely from the `support_mode` the reference declares, so this file plus adapter36
are the entire behavioural diff. No limit is widened, no phase is shortened, no new
authority is taken.
"""
from __future__ import annotations

from core35 import PAIRS35, Core35, Sensed35, smooth35

# Transfer ramps the swing-pair allocator weight as 1 - smooth35(elapsed/0.06). The
# commanded weight passes 0.5 at exactly half the ramp, so that is the declared boundary.
PREUNLOAD_WEIGHT36 = .5
PREUNLOAD_ELAPSED36 = .03
TRANSFER_RAMP36 = .06


def swing_weight36(elapsed_s: float) -> float:
    """Commanded swing-pair allocator weight during transfer, from the frozen ramp."""
    return 1.-smooth35(elapsed_s/TRANSFER_RAMP36)


class Core36(Core35):
    """Core35 with the transfer phase split into all-four and stance-pair regimes."""

    schema36 = 'd1-c36-preunload-transfer-core-v1'

    def preunload36(self) -> bool:
        """True when the consumed phase is already committing to unload the pair."""
        if self.phase == 'transfer':
            return self.elapsed >= PREUNLOAD_ELAPSED36
        return self.phase == 'transfer_restore'

    def _all_loaded(self, sensed: Sensed35) -> bool:
        """Phase-aware support requirement; see the module docstring.

        Outside the pre-unload window this is the inherited all-four condition, evaluated
        by the frozen staticmethod so the original arithmetic is preserved exactly.
        """
        if not self.preunload36():
            return Core35._all_loaded(sensed)
        # _stance_loaded takes the SWING pair and checks its complement, exactly as the
        # frozen transfer branch already calls it.
        return self._stance_loaded(sensed, PAIRS35[self.pair_index])
