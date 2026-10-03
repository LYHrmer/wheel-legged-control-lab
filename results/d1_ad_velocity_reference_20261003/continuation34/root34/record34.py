"""Bind the inherited C31 recorder to a distance-specific C34 task classifier."""
from __future__ import annotations

from functools import partial
from types import FunctionType

import record31
from macro34 import build_macros34
from task34 import gates34, score_task34


def recorder34(distance_m):
    if distance_m not in (.03, .04):
        raise ValueError('C34 recorder requires frozen distance')
    gates = gates34(distance_m)
    # Clone only the function's globals: the sealed record31 module itself is
    # untouched, and all its archive, native, control and terminal logic stays.
    namespace = dict(record31.record_case31.__globals__)
    namespace['GATES31'] = gates
    namespace['score_task31'] = partial(score_task34, distance_m=distance_m)
    namespace['build_macros31'] = partial(build_macros34, distance_m=distance_m)
    original = record31.record_case31
    clone = FunctionType(original.__code__, namespace, 'record_case34',
                         original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    return clone
