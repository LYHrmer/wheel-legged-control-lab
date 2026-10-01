"""C30 headless extensions: one owner, three data objects, no extra integration."""
from __future__ import annotations

import threading

from side_runtime27 import SideAccess27


class SideAccess30(SideAccess27):
    """Original static API fence plus at most one plan IK per actual leg."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limits.update(
            copy=self.compute_limit + 4*self.prepare_limit,
            forward=112*self.compute_limit + 236*self.prepare_limit,
            jacBody=96*self.compute_limit + 192*self.prepare_limit,
        )

    def report(self):
        report = super().report()
        bounds = dict(report['actual_T_P_bounds'])
        bounds.update(copy=self.computes + 4*self.prepares,
                      forward=112*self.computes + 236*self.prepares,
                      jacBody=96*self.computes + 192*self.prepares)
        report.update(schema='d1-c30-side-static-access-v1',
                      actual_T_P_bounds=bounds,
                      within_actual_bounds=all(self.counts[k] <= bounds[k]
                                               for k in self.counts),
                      extra_plan_ik_limit_per_prepare=1)
        return report


def fence_copy_data30(mj, *, plant, side_access, stop_event):
    """Permit live→measurement and measurement→registered scratch only."""
    original = mj.mj_copyData
    objects = (plant.data, plant.measurement_data, side_access.scratch)
    if len({id(v) for v in objects}) != 3:
        raise RuntimeError('C30 requires three distinct headless data objects')
    owner = threading.get_ident()
    edges = dict(live_to_measurement=0, measurement_to_side_scratch=0, rejected=0)

    def checked(destination, model, source):
        edge = None
        if threading.get_ident() == owner and model is plant.model:
            if destination is plant.measurement_data and source is plant.data:
                edge = 'live_to_measurement'
            elif destination is side_access.scratch and source is plant.measurement_data:
                side_access.count('copy')
                edge = 'measurement_to_side_scratch'
        if edge is None:
            edges['rejected'] += 1
            stop_event.set()
            raise RuntimeError('C30 headless copy crossed the owner/data fence')
        edges[edge] += 1
        return original(destination, model, source)

    mj.mj_copyData = checked
    return original, edges


class NativeWindow30:
    """Read saved post-native states; never query or advance the engine."""

    def __init__(self, binding):
        self.binding = binding['kinematics24']
        self.wheel_map = {int(k): v for k, v in binding['wheel_index_by_body_id'].items()}
        self.rows = []
        self.control_index = 0
        self.pure_fk_calls = 0

    def begin(self, control_index):
        if type(control_index) is not int or control_index < 0:
            raise ValueError('invalid control index')
        self.control_index = control_index
        self.rows = []

    def sink(self, row):
        from kinematics24 import reconstruct24, collision_bounds24
        if row['native_index'] != 5*self.control_index + len(self.rows) or len(self.rows) >= 5:
            raise RuntimeError('C30 native window index differs')
        state = reconstruct24(self.binding, row['after']['qpos'], row['after']['qvel'])
        bounds = collision_bounds24(self.binding, state, self.wheel_map)
        minimum = [min(g['minimum_world_m'][2] for g in bounds
                       if g['wheel_index'] == leg) for leg in range(4)]
        active = [any(c['robot_wheel_index'] == leg and c['terrain_family'] is not None
                      and c['efc_address'] >= 0 for c in row['contacts']) for leg in range(4)]
        proof = dict(native_index=row['native_index'], start_time_s=row['start_time_s'],
                     end_time_s=row['end_time_s'], whole_wheel_min_z_m=minimum,
                     active_wheel_terrain_contact=active,
                     source='compiled_FK_of_actual_post_native_qpos_and_actual_solver_contacts')
        row['skill30_geometry'] = proof
        self.rows.append(proof)
        self.pure_fk_calls += 1

    def complete(self):
        if len(self.rows) != 5:
            raise RuntimeError('C30 last completed interval must contain five actual returns')
        return dict(control_index=self.control_index, native_rows=[dict(row) for row in self.rows])
