"""Pure target assembly shared by the actual C35 paired scratch IK and fixture."""
from __future__ import annotations

import numpy as np


def pair_ik_targets35(feet_world_m, anchors_world_m, extent_m, swing_legs):
    """Keep both swing offsets while correcting every wheel-body target Z."""
    feet = np.asarray(feet_world_m,dtype=float)
    anchors = np.asarray(anchors_world_m,dtype=float)
    extent = np.asarray(extent_m,dtype=float)
    pair = tuple(swing_legs)
    if (feet.shape != (4,3) or anchors.shape != (4,3)
            or extent.shape != (4,) or pair not in ((),(0,3),(1,2))
            or not all(np.isfinite(x).all() for x in (feet,anchors,extent))):
        raise ValueError('C35 paired IK targets or mask differ')
    targets = feet.copy()
    targets[:,2] = extent-.001
    for leg in pair:
        targets[leg,2] += feet[leg,2]-anchors[leg,2]
    return targets
