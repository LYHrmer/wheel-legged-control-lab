"""C38 controller adapter: declare the pre-unload regime without overloading swing_legs.

The C37 independent reader rejected the C36/C37 record semantics, correctly. Its per-control
cross-check is

    scope['calls']['fullM'] == int(bool(reference['swing_legs']))

and the C36 override filled `swing_legs` with the pre-unload pair so the guard would see the
stance pair. But no coupled inertia is computed during transfer, so the count was 0 while the
declaration implied 1.

The guard never needed `swing_legs`: it selects its regime from `support_mode` and
`stance_legs`, and reads `swing_legs` only under `horizontal_active`, which is False through
transfer. So the fix is to declare the regime and leave `swing_legs` at the frozen value.

This is a record-semantics change only. The controller's own `self.swing_legs`, which gates
the coupled-inertia computation and the swing PD, is set by the frozen `PairController35.
compute` from `reference.phase in AIR_PHASES35` and is untouched here. The C38 trajectory is
therefore expected to reproduce C37's bit for bit, which doubles as a determinism check.
"""
from __future__ import annotations

from adapter35 import AIR_PHASES35, PairController35
from adapter36 import PairController36
from core36 import swing_weight36


class PairController38(PairController36):
    """PairController36 with the swing_legs overload removed from the declared record."""

    schema38 = 'd1-c38-preunload-record-semantics-v1'

    def _reference_record35(self, reference):
        # Deliberately skip PairController36's version and start from the frozen record, so
        # the only difference from frozen C35 is the regime declaration below.
        row = PairController35._reference_record35(self, reference)
        if reference.pair is None or reference.phase in AIR_PHASES35:
            return row
        if not self.preunload36(reference):
            return row
        swing = list(reference.pair)
        stance = [j for j in range(4) if j not in swing]
        # support_mode and stance_legs are what the frozen guard reads; swing_legs stays at
        # the frozen value so the reader's fullM cross-check holds.
        row.update(stance_legs=stance, support_mode='pair', support_gate_required=True,
                   preunload36=True, preunload_pair36=swing,
                   commanded_swing_weight36=swing_weight36(reference.phase_elapsed_s))
        return row
