"""Explicit production constructor for the real-box C18 99D/16D task.

No injected factory or temporary course model. Reset/step/observation/servo/
reward remain inherited frozen world-upright implementations.
"""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from world_upright_course_11 import WorldUprightCourseEnv, WorldUprightLoop
from full_drive_command_08 import QualifiedCommandCaps
from full_drive_servo_08 import FullDriveCommandServo
from controller18 import FullDriveController18
from plant24 import SingleStepPlant24
from geometry24 import SPAWN, CONTROLS, CONTRACT_ID
from wheel_legged_control.d1.actuator_channel import ActuatorChannel, ActuatorChannelConfig
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT
from wheel_legged_control.d1.state_provider import D1StateProviderConfig, build_d1_state_provider


class SingleStepWorldEnv24(WorldUprightCourseEnv):
    def __init__(self, *, caps, command_source):
        gym.Env.__init__(self)
        if not isinstance(caps, QualifiedCommandCaps) or not callable(command_source):
            raise TypeError('C24 explicit caps and command source required')
        self.pure_test_components = False
        self.mode, self.caps = 'eval', caps
        self.command_source = command_source
        self.spawn_position_m, self.max_steps = SPAWN, CONTROLS
        self._servo = FullDriveCommandServo(caps)
        self.action_space = gym.spaces.Box(-1., 1., (16,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-5., 5., (99,), dtype=np.float32)
        channel = ActuatorChannel(ActuatorChannelConfig(
            torque_limit_nm=tuple(float(x) for x in JOINT_TORQUE_LIMIT)))
        plant = SingleStepPlant24(actuator_channel=channel)
        controller = FullDriveController18(plant, mode='eval', variant='combined')
        provider = build_d1_state_provider(plant, D1StateProviderConfig('oracle'),
            oracle_ground_query=plant.locomotion_ground_reference)
        self.plant, self.controller = plant, controller
        self.loop = WorldUprightLoop(plant, provider, controller, caps=caps)
        self.ground_map = plant.ground_map
        if type(plant) is not SingleStepPlant24:
            raise RuntimeError('C24 did not construct exact actual single-step plant')
        self._steps = None
        self._active = False
        self._episode_index = -1
        self.decision = self.last_transition = None
        self._last_observation = self._last_packet = self._current_servo_receipt = None
        self._episode_metadata = {}
        self._native_interval_reader = self._native_interval_beginner = None

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self._episode_metadata.update(execution_contract_id=CONTRACT_ID,
            geometry_ground_source='actual_native_plane_plus_single_15mm_box',
            command_height_semantics='clearance_above_actual_box_or_plane',
            prior_task_semantics_not_claimed_identical=True)
        info['episode_metadata'] = self.episode_metadata
        return observation, info
