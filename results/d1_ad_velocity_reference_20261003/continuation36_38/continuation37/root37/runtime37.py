"""C37: repair the latent fullM static-query check that stopped the C36 attempt.

C36 attempt 1 got past the pre-unload transfer gate and entered the lift phase, then
crashed inside the frozen C35 static-query checker:

    runtime35.py:61
    packed = isinstance(args[2], np.ndarray) and np.shares_memory(args[2], self.scratch.qM)
    AttributeError: 'mujoco._structs.MjData' object has no attribute 'qM'

MuJoCo 3.12.0 exposes the inertia as `MjData.M`, not `MjData.qM`. The repository's
`d1_fast_side_step._dense_inertia` already handles both bindings and correctly takes the
modern branch, `mj_fullM(model, data, matrix)`, so `modern` is already True - but the
checker evaluates the legacy `packed` branch unconditionally and dereferences the attribute
that no longer exists. C35 never reached a coupled-inertia computation (its ledger records
`fullM: 0`), so the defect sat unexercised for a whole contract.

The repair is a short circuit and nothing else. Acceptance stays exactly `modern or packed`,
the registered-scratch requirement stays, the explicit-swing-pair requirement stays, the
counting stays, and the 6x6 cache extraction stays. The predicate is lifted into a pure
function so it can be tested without constructing a model.
"""
from __future__ import annotations

import numpy as np
from runtime35 import SideAccess35


def registered_destination37(modern: bool, third, scratch) -> bool:
    """Frozen acceptance `modern or packed`, with the legacy branch short-circuited.

    `packed` is only evaluated when the modern binding was not used, and only when the
    scratch actually exposes the legacy packed-inertia attribute. Nothing is accepted that
    the frozen predicate would have rejected.
    """
    if modern:
        return True
    if not isinstance(third, np.ndarray):
        return False
    legacy = getattr(scratch, 'qM', None)
    if legacy is None:
        return False
    return bool(np.shares_memory(third, legacy))


class SideAccess37(SideAccess35):
    """SideAccess35 with the fullM destination check short-circuited."""

    schema37 = 'd1-c37-fullm-destination-check-v1'

    def _checked_query(self, kind, original, *args, **kwargs):
        if kind != 'fullM' or self.active is None:
            return super()._checked_query(kind, original, *args, **kwargs)
        self.owner()
        if kwargs or len(args) != 3 or args[0] is not self.plant.model:
            self.reject('C37 fullM signature/model differs')
        modern = args[1] is self.scratch
        if not registered_destination37(modern, args[2], self.scratch):
            self.reject('C37 fullM must use registered scratch')
        pair = tuple(self.side.swing_legs)
        if pair not in ((0, 3), (1, 2)):
            self.reject('C37 inertia has no explicit swing pair')
        self.count(kind)
        result = original(*args, **kwargs)
        dense = args[2] if modern else args[1]
        dofs = self.side.vadr[list(pair), :3].ravel()
        self.last_pair_inertia_6x6 = np.asarray(dense)[np.ix_(dofs, dofs)].copy()
        return result
