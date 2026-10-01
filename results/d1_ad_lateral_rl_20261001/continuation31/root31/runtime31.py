"""C31 episode-local permissions over the unchanged cumulative physics ledger."""
from __future__ import annotations

from contextlib import contextmanager
import math

from runtime30 import NativeWindow30, SideAccess30, fence_copy_data30


class SideAccess31(SideAccess30):
    def __init__(self, mj, runtime, plant, scratch, *, cycles_limit):
        if type(cycles_limit) is not int or not 1 <= cycles_limit <= 64:
            raise ValueError('finite C31 cycle limit required')
        super().__init__(mj, runtime, plant, scratch,
                         compute_limit=1500*cycles_limit,
                         prepare_limit=4*cycles_limit, start_limit=cycles_limit)
        self.cycles_limit31 = cycles_limit
        self.case31 = None
        self.case_receipts31 = []

    def begin_case31(self, cycle_index):
        self.owner()
        if (self.active is not None or self.case31 is not None
                or type(cycle_index) is not int
                or cycle_index != len(self.case_receipts31)
                or cycle_index >= self.cycles_limit31):
            self.reject('C31 begin case is not a new closed-cycle boundary')
        self.case31 = dict(cycle_index=cycle_index, starts=self.starts,
                           computes=self.computes, prepares=self.prepares,
                           counts=dict(self.counts), scopes=len(self.scopes))

    def _allow31(self, kind):
        if self.case31 is None:
            self.reject('C31 side work outside a declared cycle')
        limits = dict(starts=1, computes=1500, prepares=4)
        if getattr(self, kind)-self.case31[kind] >= limits[kind]:
            self.reject('C31 per-cycle '+kind+' exhausted')

    def note_prepare_leg(self):
        self._allow31('prepares')
        return super().note_prepare_leg()

    @contextmanager
    def scope(self, kind):
        self._allow31('starts' if kind == 'start' else 'computes')
        with super().scope(kind):
            yield

    def finish_case31(self):
        self.owner()
        if self.case31 is None or self.active is not None:
            self.reject('C31 cannot close inactive/nested side case')
        before = self.case31
        actual = {key: getattr(self, key)-before[key]
                  for key in ('starts', 'computes', 'prepares')}
        count_delta = {key: value-before['counts'][key]
                       for key, value in self.counts.items()}
        t, p, s = actual['computes'], actual['prepares'], actual['starts']
        bounds = dict(copy=t+4*p, forward=112*t+236*p,
                      jacBody=96*t+192*p, jac=4*t, fullM=t,
                      objectVelocity=2*t+s)
        receipt = dict(cycle_index=before['cycle_index'], before=before,
                       actual=actual, static_counts=count_delta,
                       actual_static_bounds=bounds,
                       scope_range=[before['scopes'], len(self.scopes)],
                       after=dict(starts=self.starts, computes=self.computes,
                                  prepares=self.prepares, counts=dict(self.counts)),
                       passed=(s <= 1 and t <= 1500 and p <= 4 and
                               all(0 <= count_delta[k] <= bounds[k] for k in bounds)))
        self.case_receipts31.append(receipt)
        self.case31 = None
        if not receipt['passed']:
            self.reject('C31 per-cycle static accounting does not close')
        return receipt

    def report(self):
        result = super().report()
        result.update(schema='d1-c31-cumulative-side-static-access-v1',
                      cycles_limit=self.cycles_limit31,
                      closed_cycles=list(self.case_receipts31),
                      open_cycle=self.case31)
        return result


class NativeWindow31:
    """Actual guard indices remain global even though simulation time resets."""
    def __init__(self, binding):
        self.binding31 = binding
        self.window31 = None
        self.cycle_index31 = None
        self.global_control_offset31 = None
        self.local_control31 = None
        self.pure_fk_calls = 0
        self.native_rows31 = []
        self.next_local31 = 0
        self.interval_open31 = False
        self.previous_global_end31 = 0

    def begin_case31(self, cycle_index, global_control_offset):
        if any(type(v) is not int or v < 0
               for v in (cycle_index, global_control_offset)):
            raise ValueError('C31 finite actual cycle/global offsets required')
        if (self.interval_open31 or global_control_offset != self.previous_global_end31
                or cycle_index != (0 if self.cycle_index31 is None else self.cycle_index31+1)):
            raise RuntimeError('C31 cannot overwrite an unclosed/global-discontinuous native case')
        self.cycle_index31 = cycle_index
        self.global_control_offset31 = global_control_offset
        self.local_control31 = None
        self.window31 = NativeWindow30(self.binding31)
        self.native_rows31 = []
        self.next_local31 = 0

    def begin(self, local_control_index):
        if (self.window31 is None or type(local_control_index) is not int
                or local_control_index != self.next_local31 or self.interval_open31):
            raise RuntimeError('C31 native interval has no actual cycle')
        self.local_control31 = local_control_index
        self.interval_open31 = True
        self.window31.begin(self.global_control_offset31+local_control_index)

    def sink(self, row):
        if self.local_control31 is None or not self.interval_open31:
            raise RuntimeError('C31 native sink outside an opened control')
        self.window31.sink(row)
        self.pure_fk_calls += 1
        proof = row['skill30_geometry']
        proof.update(cycle_index31=self.cycle_index31,
                     local_control_index31=self.local_control31,
                     global_control_index31=self.global_control_offset31+self.local_control31,
                     local_native_index31=row['native_index']-5*self.global_control_offset31,
                     global_native_index31=row['native_index'])
        self.native_rows31.append(row)

    def complete(self):
        if not self.interval_open31:
            raise RuntimeError('C31 native interval was already closed')
        result = self.window31.complete()
        result.update(cycle_index31=self.cycle_index31,
                      local_control_index31=self.local_control31,
                      global_control_offset31=self.global_control_offset31)
        self.interval_open31 = False
        self.next_local31 += 1
        self.previous_global_end31 = self.global_control_offset31+self.next_local31
        return result


def install_reset_yaw31(env):
    """Pass yaw into the existing plant/provider/controller reset, before prepare."""
    original = env.loop.reset
    state = dict(initial_yaw_rad=0.0, calls=0, receipts=[])

    def reset(*, seed=None, base_position=None, base_quaternion=None):
        yaw = float(state['initial_yaw_rad'])
        if yaw not in (0.0, .04, -.04) or base_quaternion is not None:
            raise ValueError('C31 reset yaw must be one of the frozen conditions')
        if yaw == 0.0:
            # Preserve the nominal original invocation and floating-point path.
            result = original(seed=seed, base_position=base_position)
            quaternion = None
        else:
            quaternion = (math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2))
            result = original(seed=seed, base_position=base_position,
                              base_quaternion=quaternion)
        state['calls'] += 1
        state['receipts'].append(dict(call_index=state['calls']-1,
                                      requested_initial_yaw_rad=yaw,
                                      forwarded_base_quaternion_wxyz=quaternion,
                                      original_reset_called_once=True,
                                      phase='plant_provider_controller_reset_before_prepare'))
        return result

    env.loop.reset = reset
    return state


class EpisodeBudget31:
    """Reservation before reset; actual cumulative engine counts close each case."""
    def __init__(self, *, cycles, controls, macros):
        if (type(cycles) is not int or not 1 <= cycles <= 64
                or type(controls) is not int or not 1 <= controls <= 140800
                or type(macros) is not int or not 1 <= macros <= 256):
            raise ValueError('invalid C31 cycle/control/macro caps')
        self.limits = dict(cycles=cycles, controls=controls, macros=macros)
        self.closed = []
        self.open = None
        self.macros = 0

    def begin(self, *, controls, native):
        previous = self.closed[-1]['global_control_end'] if self.closed else 0
        if (self.open is not None or len(self.closed) >= self.limits['cycles']
                or self.closed and not self.closed[-1]['complete']
                or controls != previous or native != 5*controls
                or controls >= self.limits['controls']
                or self.macros >= self.limits['macros']):
            raise RuntimeError('C31 cannot reserve an additional cycle')
        self.open = dict(cycle_index=len(self.closed), global_control_begin=controls,
                         global_native_begin=native,
                         control_limit=min(2200, self.limits['controls']-controls))
        return dict(self.open)

    def finish(self, *, controls, native, macros, complete):
        if self.open is None:
            raise RuntimeError('C31 actual result lacks a cycle reservation')
        start = self.open
        delta = controls-start['global_control_begin']
        if (type(macros) is not int or not 0 <= macros <= 4
                or not 0 <= delta <= start['control_limit'] or native != 5*controls
                or self.macros+macros > self.limits['macros']):
            raise RuntimeError('C31 cumulative budget or native accounting differs')
        row = dict(start, global_control_end=controls, global_native_end=native,
                   actual_controls=delta, actual_normal_native=5*delta,
                   actual_macros=macros, complete=bool(complete))
        self.closed.append(row)
        self.macros += macros
        self.open = None
        return row


__all__ = ['SideAccess31', 'NativeWindow31', 'EpisodeBudget31',
           'install_reset_yaw31', 'fence_copy_data30']
