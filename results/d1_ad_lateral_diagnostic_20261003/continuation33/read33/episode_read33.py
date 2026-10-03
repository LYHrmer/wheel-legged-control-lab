"""Reuse the exact C31 physical episode proof with two narrow C33 verifiers.

Only global lookups for latch timing and body reference are replaced. C31 keeps
its archive translation, full native/controller/state gates, task, reward,
retention, and online-result reconciliation. Old source files stay untouched.
"""
from __future__ import annotations

from types import FunctionType

import episode_read31 as inherited
from latches_read33 import verify_latches33
from reference_read33 import verify_reference33


def physical_episode33(run, session, construction, binding, geometry, kin,
                       original_spec, cycle_index, offset, *, alpha33):
    def latches(*args, **kwargs):
        return verify_latches33(*args, **kwargs, alpha33=alpha33)

    def reference(*args, **kwargs):
        return verify_reference33(*args, **kwargs, alpha33=alpha33)

    namespace = dict(inherited.physical_episode31.__globals__,
                     verify_latches31=latches, verify_reference30=reference)
    proof = FunctionType(inherited.physical_episode31.__code__, namespace,
                         'inherited_C31_physical_episode_with_C33_body_verifiers')
    report, cycle, episode = proof(run, session, construction, binding, geometry,
                                   kin, original_spec, cycle_index, offset, None)
    report['body_alpha33'] = float(alpha33)
    report['inherited_physical_verifier'] = 'episode_read31.physical_episode31'
    report['replaced_global_lookups'] = [
        'verify_latches31 -> latches_read33.verify_latches33',
        'verify_reference30 -> reference_read33.verify_reference33 (calls original)']
    report['new_lateral_actor_or_optimizer_calls'] = 0
    return report, cycle, episode
