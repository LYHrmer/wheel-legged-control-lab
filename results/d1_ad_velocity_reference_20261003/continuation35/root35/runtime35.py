"""C35 owner-only static fence and actual native support evidence."""
from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np

from side_runtime27 import SideAccess27
from runtime31 import NativeWindow31


def static_bounds35(t, s):
    return dict(copy=2*t+4*s, forward=256*t+64*s,
                jacBody=192*t+48*s, jac=4*t, fullM=t,
                objectVelocity=2*t+s)


class SideAccess35(SideAccess27):
    """Same actual owner/data fence; explicit paired inertia, no leg alias."""
    def __init__(self, mj, runtime, plant, scratch):
        super().__init__(mj, runtime, plant, scratch, compute_limit=6000,
                         prepare_limit=1, start_limit=5)
        self.limits = static_bounds35(6000, 5)
        self.case = None
        self.closed_cases = []
        self.last_pair_inertia_6x6 = None

    def begin_case35(self, index):
        self.owner()
        if self.case is not None or self.active or index != len(self.closed_cases) or index >= 5:
            self.reject('C35 invalid case boundary')
        self.case = dict(index=index, starts=self.starts, computes=self.computes,
                         counts=dict(self.counts), scope_begin=len(self.scopes))

    def note_prepare_leg(self):
        self.reject('C35 has no old single-leg plan/prepare authority')

    @contextmanager
    def scope(self, kind):
        if self.case is None:
            self.reject('C35 side scope without case')
        key, limit = ('starts', 1) if kind == 'start' else ('computes', 1200)
        if getattr(self, key)-self.case[key] >= limit:
            self.reject('C35 per-case '+key+' exhausted')
        self.last_pair_inertia_6x6 = None
        with super().scope(kind):
            yield
        delta = self.last_scope_receipt['calls']
        bounds = static_bounds35(int(kind == 'compute'), int(kind == 'start'))
        if any(delta[k] > bounds[k] for k in bounds):
            self.reject('C35 per-scope static query cap exceeded')

    def _checked_query(self, kind, original, *args, **kwargs):
        if kind != 'fullM' or self.active is None:
            return super()._checked_query(kind, original, *args, **kwargs)
        self.owner()
        if kwargs or len(args) != 3 or args[0] is not self.plant.model:
            self.reject('C35 fullM signature/model differs')
        modern = args[1] is self.scratch
        packed = isinstance(args[2], np.ndarray) and np.shares_memory(args[2], self.scratch.qM)
        if not (modern or packed):
            self.reject('C35 fullM must use registered scratch')
        pair = tuple(self.side.swing_legs)
        if pair not in ((0, 3), (1, 2)):
            self.reject('C35 inertia has no explicit swing pair')
        self.count(kind)
        result = original(*args, **kwargs)
        dense = args[2] if modern else args[1]
        dofs = self.side.vadr[list(pair), :3].ravel()
        self.last_pair_inertia_6x6 = np.asarray(dense)[np.ix_(dofs, dofs)].copy()
        return result

    def finish_case35(self):
        self.owner()
        if self.case is None or self.active:
            self.reject('C35 closing undeclared/active case')
        before = self.case
        t, s = self.computes-before['computes'], self.starts-before['starts']
        delta = {k: v-before['counts'][k] for k, v in self.counts.items()}
        bounds = static_bounds35(t, s)
        row = dict(case_index=before['index'], before=before,
                   actual=dict(computes=t, starts=s), static_counts=delta,
                   actual_static_bounds=bounds, scope_end=len(self.scopes),
                   passed=s <= 1 and t <= 1200 and all(delta[k] <= bounds[k] for k in bounds))
        self.closed_cases.append(row)
        self.case = None
        if not row['passed']:
            self.reject('C35 case static ledger differs')
        return row

    def report(self):
        row = super().report()
        bounds = static_bounds35(self.computes, self.starts)
        row.update(schema='d1-c35-side-static-access-v1', actual_T_P_bounds=bounds,
                   within_actual_bounds=all(self.counts[k] <= bounds[k] for k in bounds),
                   closed_cases=self.closed_cases, open_case=self.case)
        return row


def native_evidence35(row, binding, reference):
    """Online derivation. The cold reader separately derives every quantity."""
    from kinematics24 import reconstruct24
    state = reconstruct24(binding['kinematics24'], row['after']['qpos'], row['after']['qvel'])
    com = np.asarray(state['whole_com_position'])
    force, moment, loads = np.zeros(3), np.zeros(3), np.zeros(4)
    weighted_contact = np.zeros((4, 3))
    active = [False]*4
    for c in row['contacts']:
        if c['terrain_family'] is None or c['efc_address'] < 0:
            continue
        sign = 1. if c['terrain_geom_id'] == c['geom1'] else -1.
        frame = np.asarray(c['frame_world'])
        local = np.asarray(c['contact_force_local'])
        f = sign * (frame.T @ local[:3])
        force += f
        moment += np.cross(np.asarray(c['position_world_m'])-com, f) + sign*(frame.T@local[3:])
        leg = c['robot_wheel_index']
        if leg is not None:
            active[leg] = True
            normal = max(0., float(c['normal_load_on_robot_n']))
            loads[leg] += normal
            weighted_contact[leg] += normal*np.asarray(c['position_world_m'])
    stance = [] if reference is None else list(reference['stance_legs'])
    swing = [] if reference is None else list(reference['swing_legs'])
    return dict(schema='d1-c35-native-contact-evidence-v1',
                native_index=row['native_index'],
                global_control_index=row['native_index']//5,
                whole_com_world_m=com.tolist(), world_force_sum_n=force.tolist(),
                world_moment_about_com_sum_nm=moment.tolist(),
                active_wheel_terrain_contact=active,
                positive_wheel_normal_load_sum_n=loads.tolist(),
                positive_load_weighted_wheel_contact_world_m=[
                    (weighted_contact[j]/loads[j]).tolist() if loads[j] > 0 else None
                    for j in range(4)],
                stance_legs=stance, swing_legs=swing,
                consumed_phase=None if reference is None else reference['phase'],
                support_mode=None if reference is None else reference['support_mode'],
                stance_pair_normal_sum_n=None if len(stance) != 2 else float(loads[stance].sum()),
                support_normal_sum_n=None if not stance else float(loads[stance].sum()),
                whole_wheel_min_z_m=list(row['skill30_geometry']['whole_wheel_min_z_m']),
                horizontal_active=False if reference is None else bool(reference['horizontal_active']),
                support_gate_required=False if reference is None else bool(reference['support_gate_required']))


class NativeWindow35(NativeWindow31):
    def __init__(self, binding, reference_provider, joint_ranges, joint_limited):
        super().__init__(binding)
        self.reference_provider = reference_provider
        self.joint_ranges = np.asarray(joint_ranges)
        self.joint_limited = np.asarray(joint_limited, dtype=bool)
        self.evidence_fk_calls35 = 0

    def sink(self, row):
        super().sink(row)
        reference = self.reference_provider()
        proof = native_evidence35(row, self.binding31, reference)
        row['contact_evidence35'] = proof
        self.evidence_fk_calls35 += 1
        if row['nonwheel_contact_count']:
            raise RuntimeError('C35 native nonwheel terrain contact')
        q, v = np.asarray(row['after']['qpos']), np.asarray(row['after']['qvel'])
        kin = self.binding31['kinematics24']
        excursions = []
        excursion_rows = []
        for j, limited in enumerate(self.joint_limited):
            if limited:
                x = q[int(kin['jnt_qposadr'][j])]
                lo, hi = self.joint_ranges[j]
                excursions.append(max(0., lo-x, x-hi))
                excursion_rows.append(dict(joint_index=j, lower_excursion_rad=max(0.,lo-x),
                                           upper_excursion_rad=max(0.,x-hi)))
        proof['max_actual_soft_joint_range_excursion_rad'] = max(excursions, default=0.)
        proof['actual_soft_joint_range_excursions'] = excursion_rows
        if reference is None:
            if (max(abs(row['roll_deg']), abs(row['pitch_deg'])) > 10.
                    or q[2] < .28 or abs(q[0]) > 11 or abs(q[1]) > 6):
                raise RuntimeError('C35 original B22 rolling native bound')
            return
        leg_dofs = np.asarray(self.binding31['joint_dof_addresses']).reshape(4, 4)[:, :3].ravel()
        if (max(abs(row['roll_deg']), abs(row['pitch_deg'])) > math.degrees(.32)
                or q[2] < .32 or np.max(np.abs(v[leg_dofs])) > 18.):
            raise RuntimeError('C35 original side native pose/velocity bound')
        if proof['support_gate_required']:
            stance = proof['stance_legs']
            expected_count = 4 if proof['support_mode'] == 'all4' else 2
            if (len(stance) != expected_count or not all(proof['active_wheel_terrain_contact'][j]
                    and proof['positive_wheel_normal_load_sum_n'][j] > 1e-8 for j in stance)
                    or proof['support_normal_sum_n'] <
                    .5*self.binding31['nominal_total_mass_kg']*9.81):
                raise RuntimeError('C35 new '+proof['support_mode']+' actual support/load bound')
        if proof['horizontal_active']:
            if not all(proof['whole_wheel_min_z_m'][j] > .012
                       and not proof['active_wheel_terrain_contact'][j]
                       for j in proof['swing_legs']):
                raise RuntimeError('C35 actual whole-wheel horizontal clearance/contact bound')

    def complete(self):
        row = super().complete()
        row['contact_evidence35'] = [r['contact_evidence35'] for r in self.native_rows31[-5:]]
        return row
