"""Pure fixed true-box geometry and command contract; no engine imports."""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import asdict, dataclass

CONTRACT_ID = 'C24_true_15mm_B22_v1'
BOX_NAME = 'terrain_single_15mm_box'
BOX_CENTER = (-3.1, 0., .0075)
BOX_HALF = (.18, .62, .0075)
FRONT, FAR = -3.28, -2.92
SPAWN = (-3.8, 0., .455)
CONTROLS = 1200
SEED = 241001
HELDOUT_SCHEMA = 'd1-true15mm-heldout-24-v1'
CASE_ID = 'true_15mm_0p2'
RECORD_SCHEMA = 'd1-true15mm-heldout-record-24-v1'
PAIR_FIELDS = ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'observation')


@dataclass(frozen=True, slots=True)
class StepSchedule24:
    raw_commands: tuple
    case_id: str = CASE_ID
    terrain: str = 'single_step'
    seed: int = SEED
    spawn_position_m: tuple = SPAWN
    schema: str = HELDOUT_SCHEMA

    def __post_init__(self):
        expected = raw_rows24()
        if (self.case_id != CASE_ID or self.terrain != 'single_step'
                or self.seed != SEED or self.spawn_position_m != SPAWN
                or self.schema != HELDOUT_SCHEMA
                or [asdict(row) for row in self.raw_commands] != expected):
            raise ValueError('C24 fixed schedule changed')

    @property
    def control_cap(self):
        return CONTROLS

    @property
    def command_sha256(self):
        return hashlib.sha256(json.dumps(raw_rows24(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def command_source(self, tick, time_s):
        if (type(tick) is not int or not 0 <= tick < CONTROLS
                or not math.isclose(time_s, tick*.01, rel_tol=0., abs_tol=1e-10)):
            raise ValueError('C24 command clock outside fixed sequence')
        return self.raw_commands[tick]


def raw_rows24():
    return [dict(forward_velocity_mps=.2 if 200 <= tick < 1000 else 0.,
        lateral_velocity_mps=0., yaw_rate_rps=0., clearance_m=.455,
        jump_requested=False) for tick in range(CONTROLS)]


def make_schedule():
    from full_drive_command_08 import FullDriveCommand
    return StepSchedule24(tuple(FullDriveCommand(**row) for row in raw_rows24()))


def nominal_path24():
    """100 Hz velocity slew only: feasibility, never claimed physical travel."""
    x, vx = SPAWN[0], 0.
    path = [x]
    for row in raw_rows24():
        vx += max(-.005, min(.005, row['forward_velocity_mps']-vx))
        x += .01*vx
        path.append(x)
    return path


def validate_geometry24(manifest):
    rows = manifest['world_collision_geoms']
    if (len(rows) != 2 or {r['name'] for r in rows} != {'floor', BOX_NAME}
            or manifest['control_dt_s'] != .01 or manifest['native_dt_s'] != .002):
        raise ValueError('C24 requires exactly native plane plus one real 15mm box')
    for row in rows:
        box = row['name'] == BOX_NAME
        if (row['body_id'] != 0 or row['collision'] is not True
                or row['type'] != ('box' if box else 'plane')
                or row['quaternion_wxyz'] != [1., 0., 0., 0.]
                or any(abs(a-b) > 1e-14 for a,b in zip(row['position_m'],
                    BOX_CENTER if box else (0., 0., 0.)))):
            raise ValueError('C24 compiled terrain identity/transform differs')
        if box and row['size_m'] != list(BOX_HALF):
            raise ValueError('C24 actual box height or footprint differs')
    return {row['geom_id']: row for row in rows}


def reset_geometry_gate24(endpoint, manifest, ground_samples):
    """Actual reset admission plus fixed-pose nominal room, before any action."""
    validate_geometry24(manifest)
    bounds = endpoint['collision_bounds']
    if (endpoint['tick'] != 0 or endpoint['time_s'] != 0
            or endpoint['qpos'][:3] != list(SPAWN)
            or endpoint['qpos'][3:7] != [1.,0.,0.,0.]
            or any(endpoint['qvel']) or not bounds):
        raise ValueError('C24 reset does not match fixed initial physical pose')
    front_gap=min(FRONT-b['maximum_world_m'][0]-b['margin_m'] for b in bounds)
    rear_offset=min(b['minimum_world_m'][0]-SPAWN[0]-b['margin_m'] for b in bounds)
    path=nominal_path24()
    release_gap=path[1000]+rear_offset-FAR
    stop_gap=path[1100]+rear_offset-FAR
    lateral_extent=max(max(abs(b['minimum_world_m'][1]),abs(b['maximum_world_m'][1]))
                       for b in bounds)
    expected=[(SPAWN[0],0.,0.),(BOX_CENTER[0],0.,.015),
              (FRONT-.01,0.,0.),(FAR+.01,0.,0.)]
    if (front_gap <= 0 or release_gap <= 0 or stop_gap <= 0
            or lateral_extent >= BOX_HALF[1] or len(ground_samples)!=len(expected)):
        raise ValueError('C24 fixed initial/nominal space not sufficient')
    for row,(x,y,height) in zip(ground_samples,expected):
        if (row['x_m']!=x or row['y_m']!=y
                or abs(row['height_m']-height)>1e-14
                or row['geom_name'] != (BOX_NAME if height else 'floor')):
            raise ValueError('C24 oracle is not bound to actual native plane/box')
    return dict(schema='d1-true15mm-reset-geometry-gate-24-v1',passed=True,
        initial_all_robot_front_gap_with_margin_m=front_gap,
        nominal_release_all_robot_far_edge_gap_with_margin_m=release_gap,
        nominal_final100_start_all_robot_far_edge_gap_with_margin_m=stop_gap,
        nominal_endpoint_x_m=path[-1],initial_collision_lateral_extent_m=lateral_extent,
        ground_samples=ground_samples,
        geometry_claim_scope='actual reset, fixed-pose nominal translation only; actual crossing and stop require full saved physics')
