"""One C18 prepared decision, one torque owner, continuous side command."""
from __future__ import annotations

from dataclasses import asdict
import copy
import numpy as np

from controller18 import FullDriveController18
from hybrid_adapter27 import HybridController27
from world_upright_course_11 import WorldUprightCourseEnv, WorldUprightLoop
from full_drive_controller_08 import FullDriveControllerAdapter
from wheel_legged_control.d1.wheel_leg_controller import _nominal_geometry


class ContinuousOwner35(HybridController27):
    control_schema = 'd1-c35-continuous-pair-owner-v1'

    def __init__(self, plant, *, mode, side_factory):
        super().__init__(plant, mode=mode, side_arm='teacher', side_factory=side_factory)

    def reset(self):
        super().reset()
        self._continuous_intent35 = None
        self.consumed_reference35 = None

    def _wheel_memory(self):
        return copy.deepcopy(super()._wheel_memory())

    def set_command35(self, command, sensed, *, begin=False):
        if self._preview is None or self._continuous_intent35 is not None:
            raise RuntimeError('C35 command must bind one current prepared decision')
        self._continuous_intent35 = (command, sensed, bool(begin))

    def compute(self, command, state, ground, action):
        preview = self._preview
        if (preview is None or command is not preview.command or state is not preview.state
                or ground is not preview.ground or self._continuous_intent35 is None):
            raise RuntimeError('C35 exact prepared decision/intent required')
        side_command, sensed, begin = self._continuous_intent35
        self._continuous_intent35 = None
        self.consumed_reference35 = None
        accepted = False
        if begin:
            if self._side_active or self.accepted_starts or side_command is None or sensed is None:
                raise RuntimeError('C35 only one explicit continuous start per case')
            stopped = all(float(value) == 0. for value in (
                command.forward_velocity_mps, command.lateral_velocity_mps,
                command.yaw_rate_rps, command.raw_forward_velocity_mps,
                command.raw_lateral_velocity_mps, command.raw_yaw_rate_rps))
            if not stopped or np.linalg.norm(state.base_angular_velocity_world) >= .08:
                raise RuntimeError('C35 original side entry prepared-command/angular gate')
            with self.side.access.scope('start'):
                accepted = bool(self.side.start(side_command, sensed))
            self.last_start_record = dict(accepted=accepted, prepared_motion_zero=stopped,
                provider_sequence=int(state.sequence), command=asdict(side_command))
            if not accepted:
                raise RuntimeError('C35 pair controller rejected its sole start')
            self._side_active = True
            self.accepted_starts += 1
        if not self._side_active:
            if side_command is not None or sensed is not None:
                raise RuntimeError('C35 side command outside actual side session')
            return FullDriveController18.compute(self, command, state, ground, action)
        if side_command is None or sensed is None or not np.array_equal(action, np.zeros(16)):
            raise RuntimeError('C35 side needs fresh explicit command and exact zero16 action')
        self.side.set_command(side_command)
        self.side.set_evidence(sensed)
        memory_before = self._wheel_memory()
        with self.side.access.scope('compute'):
            torque = np.asarray(self.side.compute(), dtype=float).copy()
        self.side_compute_count += 1
        memory_after = self._wheel_memory()
        for key in memory_before:
            if not np.array_equal(memory_before[key], memory_after[key]):
                raise RuntimeError('C35 side altered B22 wheel memory')
        if self.side_compute_count > 1200 or torque.shape != (16,) or not np.isfinite(torque).all():
            raise RuntimeError('C35 invalid side torque/count')
        record = self.side.last_record
        self.consumed_reference35 = dict(record['reference'])
        self._preview = None
        self._side_active = not self.side.done
        self._handoff_next_preview = bool(self.side.done)
        self.last_record = dict(schema='d1-c35-continuous-side-control-v1',
            torque_source='continuous_pair35', mode=self.mode,
            requested_policy_action=np.zeros(16), physical_policy_residual_nm=np.zeros(16),
            prepared_preview_consumed=True, prepared_sequence=int(state.sequence),
            prepared_control_time_s=float(state.control_time_s), accepted_start=accepted,
            side_compute_index=self.side_compute_count, start_record=self.last_start_record,
            command35=asdict(side_command), sensed35=asdict(sensed), controller_record35=record,
            returned_side_torque_nm=torque, wheel_memory_before=memory_before,
            wheel_memory_after=memory_after, handoff_next_preview=self._handoff_next_preview,
            access_scope_receipt=self.side.access.last_scope_receipt)
        return torque


def install_owner35(env, side_factory):
    if type(env) is not WorldUprightCourseEnv or env.pure_test_components:
        raise TypeError('C35 exact production course environment required')
    old, controller = env.loop, env.controller
    if (type(old) is not WorldUprightLoop or type(controller) is not FullDriveControllerAdapter
            or env._steps is not None or env._episode_index != -1 or env._active
            or env.decision is not None or env.last_transition is not None
            or old._initialized or old._decision is not None or old._receipt is not None
            or controller._preview is not None or old.plant is not env.plant
            or old.controller is not controller or controller.plant is not env.plant):
        raise RuntimeError('C35 replacement requires virgin source chain')
    before = _nominal_geometry.cache_info()
    if before.currsize != 1:
        raise RuntimeError('C35 nominal geometry cache should already be warm')
    new = ContinuousOwner35(env.plant, mode=env.mode, side_factory=side_factory)
    after = _nominal_geometry.cache_info()
    if after.misses != before.misses or after.hits != before.hits+1:
        raise RuntimeError('C35 owner construction missed nominal cache')
    loop = WorldUprightLoop(env.plant, old.provider, new, caps=env.caps)
    env.controller, env.loop = new, loop
    return new
