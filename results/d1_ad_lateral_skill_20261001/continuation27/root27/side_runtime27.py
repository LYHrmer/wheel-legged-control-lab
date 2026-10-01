"""C27: one owner-only scratch, counted static queries, unchanged integration."""
from __future__ import annotations

from contextlib import contextmanager
import threading


STATE_FIELDS = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart')


class SideAccess27:
    def __init__(self, mj, runtime, plant, scratch, *, compute_limit=1500,
                 prepare_limit=4, start_limit=1):
        if any(type(v) is not int or v < 1 for v in
               (compute_limit, prepare_limit, start_limit)):
            raise ValueError('finite side limits must be positive integers')
        if scratch is plant.data or scratch is plant.measurement_data:
            raise ValueError('side scratch aliases physical data')
        self.mj, self.runtime, self.plant, self.scratch = mj, runtime, plant, scratch
        self.owner_ident = threading.get_ident()
        self.compute_limit, self.prepare_limit, self.start_limit = (
            compute_limit, prepare_limit, start_limit)
        self.active = None
        self.scope_counts = None
        self.starts = self.computes = self.prepares = self.rejected = 0
        self.counts = {k: 0 for k in ('copy', 'forward', 'jacBody', 'jac', 'fullM', 'objectVelocity')}
        self.limits = {'copy': compute_limit+3*prepare_limit,
                       'forward': 112*compute_limit+177*prepare_limit,
                       'jacBody': 96*compute_limit+144*prepare_limit,
                       'jac': 4*compute_limit, 'fullM': compute_limit,
                       'objectVelocity': 2*compute_limit+start_limit}
        self.scopes = []
        self.originals = {}
        self.side = None
        self.last_scope_receipt = None
        self.last_jacobians_by_leg = None
        self.last_swing_inertia_3x3 = None
        self.last_origin_velocity_world = None
        self.last_com_velocity_world = None
        self.last_angular_velocity_world = None

    def reject(self, reason):
        self.rejected += 1
        self.runtime._stop_event.set()
        raise RuntimeError('C27 scratch access rejected: '+reason)

    def owner(self):
        if threading.get_ident() != self.owner_ident:
            self.reject('foreign thread')

    def count(self, kind):
        self.owner()
        if self.active not in ('start', 'compute'):
            self.reject('static query outside side scope')
        if self.counts[kind] >= self.limits[kind]:
            self.reject(kind+' exceeds static upper bound')
        if (kind == 'objectVelocity' and self.scope_counts is not None
                and self.counts[kind]-self.scope_counts[kind] >=
                    (1 if self.active == 'start' else 2)):
            self.reject('objectVelocity exceeds per-scope upper bound')
        self.counts[kind] += 1

    def note_prepare_leg(self):
        self.owner()
        if self.active is None or self.prepares >= self.prepare_limit:
            self.reject('prepare_leg outside scope or over budget')
        self.prepares += 1

    def _physical_snapshot(self):
        import numpy as np
        return tuple((float(d.time), tuple(np.asarray(getattr(d, key)).copy()
                     for key in STATE_FIELDS))
                     for d in (self.plant.data, self.plant.measurement_data))

    def _native_snapshot(self):
        reader = getattr(self.runtime, 'side_native_counters', self.runtime.state)
        state = reader()
        return {key: state[key] for key in ('control_attempts', 'control_returns',
                 'construction_attempts', 'construction_returns', 'ccd_attempts',
                 'ccd_returns', 'violations')}

    @contextmanager
    def scope(self, kind):
        self.owner()
        if kind not in ('start', 'compute') or self.active is not None:
            self.reject('unknown or nested scope')
        if kind == 'start':
            if self.starts >= self.start_limit:
                self.reject('start attempt limit')
            self.starts += 1
        else:
            if self.computes >= self.compute_limit:
                self.reject('compute limit')
            self.computes += 1
        before = self._physical_snapshot()
        boundary = self._native_snapshot()
        counts = dict(self.counts)
        self.scope_counts = counts
        self.last_jacobians_by_leg = [None]*4
        self.last_swing_inertia_3x3 = None
        self.last_origin_velocity_world = None
        self.last_com_velocity_world = None
        self.last_angular_velocity_world = None
        self.active = kind
        failure = None
        try:
            yield
        except BaseException as error:
            failure = type(error).__name__+': '+str(error)
            raise
        finally:
            self.active = None
            after = self._physical_snapshot()
            unchanged = all(bt == at and all(b.dtype == a.dtype and b.shape == a.shape
                and b.tobytes() == a.tobytes()
                for b, a in zip(ba, aa)) for (bt, ba), (at, aa) in zip(before, after))
            after_boundary = self._native_snapshot()
            integration_keys = ('control_attempts', 'control_returns',
                                'construction_attempts', 'construction_returns', 'violations')
            no_native = all(after_boundary[k] == boundary[k] for k in integration_keys)
            ccd = {k: after_boundary[k]-boundary[k] for k in ('ccd_attempts', 'ccd_returns')}
            ccd_valid = ccd['ccd_attempts'] == ccd['ccd_returns'] and ccd['ccd_returns'] >= 0
            self.scopes.append({'kind': kind, 'compute_count': self.computes,
                'prepare_count': self.prepares, 'start_count': self.starts,
                'calls': {k: self.counts[k]-counts[k] for k in counts},
                'physical_arrays_unchanged': unchanged, 'native_boundary_unchanged': no_native,
                'static_ccd_delta': ccd, 'static_ccd_returned': ccd_valid,
                'failure': failure})
            self.last_scope_receipt = self.scopes[-1]
            if not unchanged or not no_native or not ccd_valid:
                self.reject('side static work changed physical state or integration counters')

    def _checked_query(self, kind, original, *args, **kwargs):
        # Existing rolling queries retain their original semantics. No scratch
        # query is allowed outside the explicitly counted side owner scope.
        if self.active is None:
            import numpy as np
            values = (*args, *kwargs.values())
            if any(value is self.scratch or
                   isinstance(value, np.ndarray) and hasattr(self.scratch, 'qM')
                   and np.shares_memory(value, self.scratch.qM) for value in values):
                self.reject('scratch query outside scope')
            return original(*args, **kwargs)
        self.owner()
        if kwargs or not args or args[0] is not self.plant.model:
            self.reject(kind+' has foreign model or signature')
        if kind in ('jacBody', 'jac', 'objectVelocity'):
            target = self.scratch if kind == 'jacBody' else self.plant.measurement_data
            if len(args) < 2 or args[1] is not target:
                self.reject(kind+' has wrong data')
            if kind == 'objectVelocity' and (len(args) != 6
                    or int(args[2]) != int(self.mj.mjtObj.mjOBJ_BODY)
                    or args[3] != self.plant.base_body_id or args[5] != 0):
                self.reject('objectVelocity is not the original world base query')
        elif kind == 'fullM':
            import numpy as np
            modern = len(args) == 3 and args[1] is self.scratch
            packed = (len(args) == 3 and hasattr(self.scratch, 'qM')
                      and isinstance(args[2], np.ndarray)
                      and np.shares_memory(args[2], self.scratch.qM)
                      and args[2].shape == self.scratch.qM.shape)
            if not (modern or packed):
                self.reject('fullM does not reference the registered scratch')
        self.count(kind)
        result = original(*args, **kwargs)
        import numpy as np
        if kind == 'jac':
            if self.side is None or int(args[5]) not in self.side.bodies:
                self.reject('Jacobian is not an actual side wheel')
            leg = list(self.side.bodies).index(int(args[5]))
            self.last_jacobians_by_leg[leg] = np.asarray(args[2])[:, self.plant.dof_addresses].copy()
        elif kind == 'fullM':
            if self.side is None or self.side.leg is None:
                self.reject('inertia requested without an actual swing leg')
            dense = args[2] if args[1] is self.scratch else args[1]
            dofs = self.side.vadr[self.side.leg, :3]
            self.last_swing_inertia_3x3 = np.asarray(dense)[np.ix_(dofs, dofs)].copy()
        elif kind == 'objectVelocity':
            angular, linear = np.asarray(args[4])[:3], np.asarray(args[4])[3:]
            rotation = self.plant.measurement_data.xmat[self.plant.base_body_id].reshape(3, 3)
            offset = rotation @ self.plant.model.body_ipos[self.plant.base_body_id]
            self.last_com_velocity_world = linear.copy()
            self.last_angular_velocity_world = angular.copy()
            self.last_origin_velocity_world = linear - np.cross(angular, offset)
        return result

    def install(self):
        self.owner()
        if self.originals:
            self.reject('query wrappers already installed')
        for attr, kind in (('mj_jacBody', 'jacBody'), ('mj_jac', 'jac'),
                           ('mj_fullM', 'fullM'), ('mj_objectVelocity', 'objectVelocity')):
            original = getattr(self.mj, attr)
            self.originals[attr] = original
            def wrapped(*args, _kind=kind, _original=original, **kwargs):
                return self._checked_query(_kind, _original, *args, **kwargs)
            setattr(self.mj, attr, wrapped)

    def restore(self):
        self.owner()
        if self.active is not None:
            self.reject('restore inside active scope')
        for attr, original in self.originals.items():
            setattr(self.mj, attr, original)

    def report(self):
        actual_bounds = {'copy': self.computes+3*self.prepares,
                         'forward': 112*self.computes+177*self.prepares,
                         'jacBody': 96*self.computes+144*self.prepares,
                         'jac': 4*self.computes, 'fullM': self.computes,
                         'objectVelocity': 2*self.computes+self.starts}
        return {'schema': 'd1-c27-side-static-access-v1', 'owner_ident': self.owner_ident,
                'starts': self.starts, 'computes': self.computes, 'prepares': self.prepares,
                'counts': dict(self.counts), 'limits': dict(self.limits),
                'actual_T_P_bounds': actual_bounds,
                'within_actual_bounds': all(self.counts[k] <= actual_bounds[k] for k in self.counts),
                'rejected': self.rejected, 'scopes': list(self.scopes),
                'scratch_address': int(self.scratch._address),
                'static_contact_is_not_native_load': True}


def side_owner_runtime_type(stop_event):
    from gui13_bridge import owner_runtime_type
    from scripts.d1_single_step_records import PhysicsCallLedger
    base = owner_runtime_type(stop_event)

    class SideOwnerRuntime27(base):
        side_access = None

        def side_native_counters(self):
            """Read the frozen C ABI counters without repeating library hashes.

            Full state/provenance verification remains at worker boundaries.
            mjc_ccd counts static collision kernels, not mj_step integration.
            """
            import ctypes
            from engine_binding import _CState
            self._owner('side_native_counters')
            state = _CState()
            if self.guard._lib.epa01_get_state(ctypes.byref(state), ctypes.sizeof(state)) != 0:
                raise RuntimeError('failed to read side static-work engine counters')
            return {name: int(getattr(state, name)) for name, _ in _CState._fields_}

        def register_side_access(self, access):
            self._owner('register_side_access')
            if (self.side_access is not None or access.runtime is not self
                    or access.plant is not self._actual_plant
                    or access.owner_ident != self.owner_ident):
                raise RuntimeError('C27 scratch can only be registered once by actual owner')
            self.side_access = access

        def _forward(self, *args, **kwargs):
            access = self.side_access
            if access is not None and len(args) >= 2 and args[1] is access.scratch:
                self._owner('side_forward')
                if kwargs or len(args) != 2 or args[0] is not self._actual_plant.model:
                    return self._forbidden()
                access.count('forward')
                # This is an explicitly registered extension of the old
                # whitelist, still using its original counted ledger entry.
                return PhysicsCallLedger._forward(self, *args)
            return super()._forward(*args, **kwargs)

    return SideOwnerRuntime27


def fence_copy_data27(mj, *, plant, slots, display, mailbox, owner_ident,
                      main_ident, stop_event, side_access):
    original = mj.mj_copyData
    objects = (*slots, display, plant.data, plant.measurement_data, side_access.scratch)
    if len(objects) != 7 or len({id(v) for v in objects}) != 7:
        raise RuntimeError('C27 requires exactly seven independent MjData objects')
    edges = {'live_to_measurement': 0, 'measurement_to_writing': 0,
             'reading_to_display': 0, 'measurement_to_side_scratch': 0, 'rejected': 0}

    def checked(destination, model, source):
        ident = threading.get_ident()
        if model is not plant.model:
            edge = None
        elif ident == owner_ident and destination is plant.measurement_data and source is plant.data:
            edge = 'live_to_measurement'
        elif ident == owner_ident and source is plant.measurement_data and destination is side_access.scratch:
            side_access.count('copy')
            edge = 'measurement_to_side_scratch'
        elif ident == owner_ident and source is plant.measurement_data and mailbox.is_writing_buffer(destination):
            edge = 'measurement_to_writing'
        elif ident == main_ident and destination is display and mailbox.is_reading_buffer(source):
            edge = 'reading_to_display'
        else:
            edge = None
        if edge is None:
            edges['rejected'] += 1
            stop_event.set()
            raise RuntimeError('C27 copy crossed exclusive data fence')
        edges[edge] += 1
        return original(destination, model, source)

    mj.mj_copyData = checked
    return original, edges
