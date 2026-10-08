"""C36 controller adapter: declare the pre-unload support regime to the frozen guard.

The frozen native guard in runtime35.NativeWindow35.sink selects its regime purely from the
`support_mode` and `stance_legs` the consumed reference declares:

    expected_count = 4 if proof['support_mode'] == 'all4' else 2

so declaring 'pair' during the pre-unload window makes the UNMODIFIED guard enforce exactly
the stance-pair gate - each stance wheel with a true active terrain contact and an
independently summed positive normal load, and a stance-pair normal sum of at least 0.5 m g.
That is the same strictness the frozen air regime already uses. `horizontal_active` stays
False through transfer, so no clearance requirement is added or removed.

This file and core36.py are the entire behavioural diff from C35. Everything else - the
paired IK, the coupled inertia, the force allocation, the wheel brake, the torque clipping,
the native guard, the archive and the per-case accounting - is the frozen C35 code reused
byte for byte.
"""
from __future__ import annotations

from adapter35 import AIR_PHASES35, PairController35
from core36 import PREUNLOAD_ELAPSED36, Core36, swing_weight36


class PairController36(PairController35):
    """PairController35 with a Core36 state machine and the pre-unload regime declared."""

    schema36 = 'd1-c36-preunload-side-control-v1'

    def __init__(self, plant, side_access=None):
        super().__init__(plant, side_access=side_access)
        self.core = Core36()

    def reset(self):
        # PairController35.reset and .start each install a fresh Core35; both are replaced
        # immediately afterwards, before any step consumes the core, so the swap is exact.
        result = super().reset()
        self.core = Core36()
        return result

    def start(self, command, sensed):
        accepted = super().start(command, sensed)
        if accepted:
            self.core = Core36()
        return accepted

    @staticmethod
    def preunload36(reference) -> bool:
        """Mirror of Core36.preunload36 evaluated on the consumed reference."""
        if reference.phase == 'transfer':
            return reference.phase_elapsed_s >= PREUNLOAD_ELAPSED36
        return reference.phase == 'transfer_restore'

    def _reference_record35(self, reference):
        row = super()._reference_record35(reference)
        if reference.pair is None or reference.phase in AIR_PHASES35:
            return row
        if not self.preunload36(reference):
            return row
        swing = list(reference.pair)
        stance = [j for j in range(4) if j not in swing]
        row.update(stance_legs=stance, swing_legs=swing, support_mode='pair',
                   support_gate_required=True,
                   preunload36=True,
                   commanded_swing_weight36=swing_weight36(reference.phase_elapsed_s))
        return row
