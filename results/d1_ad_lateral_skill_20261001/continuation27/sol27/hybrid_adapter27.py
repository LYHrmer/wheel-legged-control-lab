"""C27 traditional side torque through the existing C18 prepared control loop.

Importing this module constructs no model or scratch data. The old fast gait
remains unchanged; its torque is consumed by the same env.step/plant.step path.
"""
from __future__ import annotations

import numpy as np

from controller18 import FullDriveController18
from world_upright_course_11 import WorldUprightCourseEnv, WorldUprightLoop
from full_drive_controller_08 import FullDriveControllerAdapter
from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry


SIDE_SCHEMA = 'd1-c27-hybrid-traditional-side-v1'
HYBRID_SCHEMA = 'd1-c27-c18-fastside-hybrid-v1'
SIDE_ARMS = frozenset(('teacher', 'damping_plus20'))


def _copy(value):
    return np.asarray(value, dtype=np.float64).copy()


def _side_snapshot(side):
    names = ('pose', 'body_from', 'body_to', 'body_accel', 'initial_pose', 'delta',
             'feet', 'foot_offset',
             'previous_target', 'previous_velocity_target', 'wheel_angles')
    record = {name: _copy(getattr(side, name)) for name in names}
    record.update({'phase': str(side.phase), 'phase_time_s': float(side.phase_time),
                   'shift_time_s': float(side.shift_time),
                   'swing_time_s': float(side.swing_time),
                   'leg': None if side.leg is None else int(side.leg),
                   'loading_leg': None if side.loading_leg is None else int(side.loading_leg),
                   'load_weight': float(side.load_weight),
                   'static_margin_m': float(side.margin),
                   'dynamic_margin_m': float(side.dynamic_margin),
                   'start_yaw_rad': float(side.yaw),
                   'direction': int(side.direction),
                   'failure': side.failure, 'done': bool(side.done),
                   'wheel_contacts': np.asarray(side._contacts(), dtype=bool).copy(),
                   'status': dict(side.status)})
    phase = record['phase']
    record['nominal_duration_s'] = (
        float(side.shift_time) if phase in ('shift', 'recenter') else
        float(side.swing_time) if phase == 'swing' else
        float(getattr(side.cfg, phase + '_time_s')) if hasattr(side.cfg, phase + '_time_s')
        else None)
    return record


def _ready_predicates(side, before, after):
    """Describe the frozen phase gates using its already captured speed query."""
    source = getattr(side.access, 'last_origin_velocity_world', None)
    speed = None if source is None else float(np.linalg.norm(np.asarray(source)))
    phase = before['phase']
    elapsed = float(before['phase_time_s'] + side.dt)
    contacts = np.asarray(after['wheel_contacts'], dtype=bool)
    result = {'source_phase': phase, 'elapsed_s': elapsed,
              'origin_speed_mps': speed, 'all_wheels_contact': bool(contacts.all()),
              'dynamic_margin_m': float(after['dynamic_margin_m']),
              'static_margin_m': float(after['static_margin_m'])}
    if phase == 'shift':
        result.update({'time_ready': elapsed >= before['shift_time_s'],
                       'margin_ready': after['dynamic_margin_m'] > side.cfg.margin_gate_m,
                       'speed_ready': None if speed is None else speed < side.cfg.shift_settle_speed,
                       'speed_limit_mps': float(side.cfg.shift_settle_speed)})
    elif phase == 'load':
        result.update({'time_ready': elapsed >= side.cfg.load_time_s,
                       'speed_ready': None if speed is None else speed < side.cfg.load_settle_speed,
                       'speed_limit_mps': float(side.cfg.load_settle_speed)})
    elif phase == 'recenter':
        target = np.asarray(side.initial_pose + side.delta)
        position_error = float(np.linalg.norm(
            np.asarray(side.plant.base_position[:2]) - target[:2]))
        yaw_error = float(np.arctan2(
            np.sin(float(after['status']['current_yaw_rad'])-side.yaw),
            np.cos(float(after['status']['current_yaw_rad'])-side.yaw)))
        result.update({'time_ready': elapsed >= before['shift_time_s'],
                       'speed_ready': None if speed is None else speed < side.cfg.recenter_settle_speed,
                       'speed_limit_mps': float(side.cfg.recenter_settle_speed),
                       'position_error_m': position_error,
                       'position_ready': position_error < side.cfg.final_position_tolerance_m,
                       'yaw_error_rad': yaw_error,
                       'yaw_ready': abs(yaw_error) < side.cfg.final_yaw_tolerance_rad})
    return result


def make_fast_side27(plant, side_arm: str):
    """Construct one old Fast gait plus read-only hooks for its actual intermediates."""
    if side_arm not in SIDE_ARMS:
        raise ValueError('unknown fixed C27 side arm')
    from d1_fast_side_step import FAST_PROFILE, FastSideStepController

    class AuditedFast(FastSideStepController):
        def __init__(self, target_plant):
            self.access = None  # root registers the exact seventh scratch data
            self.side_arm = side_arm
            super().__init__(target_plant, profile=FAST_PROFILE)

        def reset(self):
            super().reset()
            self.prepare_leg_calls = 0
            self.fixed_cancelled = False
            self.last_target_position_rad = None
            self.last_allocation = None

        def _prepare_leg(self):
            if self.access is None:
                raise RuntimeError('side scratch authority must be attached before start')
            self.access.note_prepare_leg()
            self.prepare_leg_calls += 1
            return super()._prepare_leg()

        def cancel(self):
            self.fixed_cancelled = True
            return super().cancel()

        def _targets(self):
            target = super()._targets()
            self.last_target_position_rad = _copy(target)
            return target

        def _allocate(self, points, com, wrench, weights):
            original = _copy(wrench)
            delta = np.zeros(2, dtype=np.float64)
            enabled = (self.side_arm == 'damping_plus20'
                       and self.phase in ('shift', 'recenter')
                       and self.failure is None and not self.fixed_cancelled)
            if enabled:
                # The frozen Fast.compute formed `wrench` from the original
                # scalar body_kd. Recover only that existing XY damping term;
                # no second velocity query, Jacobian or native call is made.
                no_damping_xy = self.mass * (
                    self.body_accel[:2] + self.cfg.body_kp *
                    (self.pose[:2] - self.plant.base_position[:2]))
                delta = .2 * (original[:2] - no_damping_xy)
                modified = original.copy()
                modified[:2] += delta
            else:
                modified = wrench  # teacher/cancel follows the old exact path
            forces = super()._allocate(points, com, modified, weights)
            self.last_allocation = {
                'contact_points_m': _copy(points), 'whole_com_m': _copy(com),
                'original_wrench': original, 'desired_wrench': _copy(modified),
                'weights': _copy(weights), 'allocated_forces': _copy(forces),
                'xy_damping_delta_n': delta.copy(),
                'xy_damping_multiplier': 1.2 if enabled else 1.0,
            }
            return forces

    return AuditedFast(plant)


class HybridController27(FullDriveController18):
    """One torque owner; side uses zero policy action and the old Fast gait."""

    control_schema = HYBRID_SCHEMA

    def __init__(self, plant, *, mode: str, side_arm: str,
                 side_factory=None) -> None:
        if side_arm not in SIDE_ARMS:
            raise ValueError('side arm must be teacher or damping_plus20')
        self.side_arm = side_arm
        super().__init__(plant, mode=mode, variant='combined')
        self.side = (make_fast_side27(plant, side_arm) if side_factory is None
                     else side_factory(plant))
        if self.side.plant is not plant or self.side.model is not plant.model:
            raise RuntimeError('side controller must share the one live plant/model')

    def reset(self):
        super().reset()
        if hasattr(self, 'side'):
            self.side.reset()
        self._side_active = False
        self._side_intent = None
        self._handoff_next_preview = False
        self.accepted_starts = 0
        self.side_compute_count = 0
        self.last_handoff_record = None
        self.last_start_record = None

    def attach_side_access(self, access) -> None:
        if self.side.access is not None or access is None:
            raise RuntimeError('side scratch authority may be attached once')
        self.side.access = access

    @property
    def side_active(self) -> bool:
        return bool(self._side_active)

    def set_side_intent(self, direction: int, *, cancel: bool = False) -> None:
        if (type(direction) is not int or direction not in (-1, 0, 1)
                or type(cancel) is not bool):
            raise ValueError('side intent must be an explicit direction and cancel bool')
        if self._preview is None or self._side_intent is not None:
            raise RuntimeError('side intent must bind exactly one current preview')
        self._side_intent = (direction, cancel)

    def _wheel_memory(self):
        return {'wheel_integral_nm': self.wheel_integral_nm,
                'wheel_common_reference_z_rad_s': self.wheel_common_reference_z_rad_s,
                'previous_servo_forward_mps': self._previous_servo_forward_mps,
                'stop_latched': self.stop_latched}

    def preview(self, command, state, ground):
        if self._handoff_next_preview:
            if self._preview is not None or self._side_active:
                raise RuntimeError('side handoff requires the consumed final side tick')
            before = self._wheel_memory()
            # Clear controller memory only. The loop receipt is already the
            # side tick's real zero action; plant/provider/servo stay continuous.
            FullDriveController18.reset(self)
            after = self._wheel_memory()
            self.last_handoff_record = {
                'schema': SIDE_SCHEMA, 'before': before, 'after': after,
                'provider_sequence': int(state.sequence),
                'control_time_s': float(state.control_time_s),
                'source': 'next_existing_prepare_after_final_side_torque'}
            self._handoff_next_preview = False
        return super().preview(command, state, ground)

    def compute(self, command, state, ground, action):
        preview = self._preview
        if (preview is None or command is not preview.command
                or state is not preview.state or ground is not preview.ground
                or self._side_intent is None):
            raise RuntimeError('hybrid compute needs its exact prepared decision and intent')
        direction, cancel = self._side_intent
        self._side_intent = None
        accepted = False
        if not self._side_active and direction and not cancel:
            if self.accepted_starts:
                raise RuntimeError('C27 case permits at most one accepted side start')
            angular_speed = float(np.linalg.norm(state.base_angular_velocity_world))
            stopped_command = all(float(value) == 0.0 for value in (
                command.forward_velocity_mps, command.lateral_velocity_mps,
                command.yaw_rate_rps, command.raw_forward_velocity_mps,
                command.raw_lateral_velocity_mps, command.raw_yaw_rate_rps))
            rejected_reason = None
            if not stopped_command:
                rejected_reason = 'prepared_motion_not_zero'
            elif angular_speed >= .08:
                rejected_reason = 'angular_velocity_gate'
            else:
                if self.side.access is None:
                    raise RuntimeError('side scratch authority is not registered')
                with self.side.access.scope('start'):
                    accepted = bool(self.side.start(direction, distance_m=.03))
                if not accepted:
                    rejected_reason = 'fast_side_start_guard'
            self.last_start_record = {'provider_sequence': int(state.sequence),
                                      'direction': direction, 'accepted': accepted,
                                      'angular_speed_rps': angular_speed,
                                      'prepared_motion_zero': stopped_command,
                                      'rejected_reason': rejected_reason}
            if accepted:
                self._side_active = True
                self.accepted_starts += 1
        if not self._side_active:
            return super().compute(command, state, ground, action)
        if direction not in (0, self.side.direction) and not cancel:
            raise RuntimeError('reverse intent cannot change a running side cycle')
        if not np.array_equal(np.asarray(action), np.zeros(16)):
            raise RuntimeError('side tick must carry exact zero16 policy action')
        if cancel:
            self.side.cancel()
        before = _side_snapshot(self.side)
        memory_before = self._wheel_memory()
        prior_velocity_target = _copy(self.side.previous_velocity_target)
        bias = _copy(self.plant.measurement_data.qfrc_bias[self.plant.dof_addresses])
        if self.side.access is None:
            raise RuntimeError('side scratch authority disappeared')
        with self.side.access.scope('compute'):
            torque = _copy(self.side.compute())
        self.side_compute_count += 1
        after = _side_snapshot(self.side)
        memory_after = self._wheel_memory()
        if (self.side_compute_count > 1500
                or not np.array_equal(memory_before['wheel_integral_nm'],
                                      memory_after['wheel_integral_nm'])
                or memory_before['wheel_common_reference_z_rad_s'] !=
                   memory_after['wheel_common_reference_z_rad_s']
                or memory_before['previous_servo_forward_mps'] !=
                   memory_after['previous_servo_forward_mps']
                or memory_before['stop_latched'] != memory_after['stop_latched']):
            raise RuntimeError('side compute exceeded budget or mutated C18 wheel memory')
        if torque.shape != (16,) or not np.isfinite(torque).all():
            raise RuntimeError('old side controller returned invalid torque')
        if self.side.last_target_position_rad is None or self.side.last_allocation is None:
            raise RuntimeError('old side controller did not publish its actual intermediates')
        velocity_target = _copy(self.side.previous_velocity_target)
        acceleration_target = np.clip(
            (velocity_target-prior_velocity_target)/self.side.dt,
            -self.side.cfg.target_accel_clip, self.side.cfg.target_accel_clip)
        self._preview = None  # consume the same preview; never prepare twice
        side_tick = self.side_compute_count
        self._side_active = not bool(self.side.done)
        self._handoff_next_preview = bool(self.side.done)
        self.last_record = {
            'schema': SIDE_SCHEMA, 'torque_source': 'traditional_side',
            'actor': 'traditional_side', 'policy_predict_called': False,
            'action16': np.zeros(16, dtype=np.float64),
            'preview_consumed': {'provider_sequence': int(state.sequence),
                'control_time_s': float(state.control_time_s),
                'proposal_tick': int(preview.proposal.tick),
                'proposal_time_s': float(preview.proposal.control_time_s)},
            'side_compute_index': side_tick, 'intent_direction': direction,
            'cancel_requested': cancel, 'accepted_start': accepted,
            'diagnostic_before': before, 'diagnostic_after': after,
            'ready_predicates': _ready_predicates(self.side, before, after),
            'side_calculation': {
                'target_position_rad': _copy(self.side.last_target_position_rad),
                'velocity_target_rad_s': velocity_target,
                'accel_target_rad_s2': acceleration_target,
                'bias_torque_nm': bias,
                **self.side.last_allocation,
                'damping_multiplier_xy': [
                    self.side.last_allocation['xy_damping_multiplier']]*2,
                'jacobians_by_leg': getattr(self.side.access,
                    'last_jacobians_by_leg', None),
                'swing_inertia_3x3': getattr(self.side.access,
                    'last_swing_inertia_3x3', None)},
            'returned_side_torque_nm': torque.copy(),
            'wheel_memory_before': memory_before,
            'wheel_memory_after': memory_after,
            'handoff_next_preview': self._handoff_next_preview,
            'access_scope_receipt': getattr(self.side.access, 'last_scope_receipt', None),
        }
        return torque


def install_hybrid27(env: WorldUprightCourseEnv, *, side_arm: str,
                     side_factory=None) -> HybridController27:
    """Replace the virgin C16 adapter once; root then registers side scratch."""
    if side_arm not in SIDE_ARMS:
        raise ValueError('unknown fixed side arm')
    if type(env) is not WorldUprightCourseEnv or env.pure_test_components:
        raise TypeError('hybrid installer needs the exact production course env')
    old_loop, old_controller = env.loop, env.controller
    if (type(old_loop) is not WorldUprightLoop
            or type(old_controller) is not FullDriveControllerAdapter
            or env._steps is not None or env._episode_index != -1 or env._active
            or env.decision is not None or env.last_transition is not None
            or old_loop._initialized or old_loop._decision is not None
            or old_loop._receipt is not None or old_controller._preview is not None
            or old_loop.plant is not env.plant or old_loop.controller is not old_controller
            or old_controller.plant is not env.plant):
        raise RuntimeError('hybrid controller requires the never-reset source chain')
    cache_before = _nominal_geometry.cache_info()
    if cache_before.currsize != 1:
        raise RuntimeError('nominal geometry must already be warm')
    controller = HybridController27(env.plant, mode=env.mode, side_arm=side_arm,
                                    side_factory=side_factory)
    cache_after = _nominal_geometry.cache_info()
    if cache_after.misses != cache_before.misses or cache_after.hits != cache_before.hits+1:
        raise RuntimeError('hybrid nominal construction was not a cache hit')
    loop = WorldUprightLoop(env.plant, old_loop.provider, controller, caps=env.caps)
    if (loop.plant is not env.plant or loop.provider is not old_loop.provider
            or loop.controller is not controller or loop.caps is not env.caps):
        raise RuntimeError('hybrid replacement did not preserve plant/provider identity')
    env.controller, env.loop = controller, loop
    return controller
