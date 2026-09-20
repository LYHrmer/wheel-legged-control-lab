"""Nonintegrating checks of one fixed kinematic wheel-target correction."""
import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from scripts.d1_flat_plane_env import D1FlatPlaneHeadingEnv
from scripts.d1_heading_tracking_env import D1HeadingTrackingEnv, load_heading_policy
from scripts.d1_turn_center_compensation import TurnCenterCompensationController
from scripts.probe_d1_heading_plane_stop_damping import PlaneStopDampingEnv
from scripts.probe_d1_heading_turn_center import TurnCenterEnv
from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.controllers import D1Command
from wheel_legged_control.d1.locomotion_checkpoint import (
    _environment_contract,
    load_locomotion_policy,
)
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT, D1Plant
from wheel_legged_control.d1.state_estimation import D1MujocoTruthStateSource
from wheel_legged_control.d1.wheel_leg_controller import D1WheelLegController


@pytest.fixture(autouse=True, scope="module")
def no_integration():
    def forbidden(*args, **kwargs):
        raise AssertionError("turn preflight must not integrate")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            patch.setattr(mujoco, name, forbidden)
        yield


@pytest.fixture(scope="module")
def state(no_integration):
    plant = D1Plant(sampling_mode="synchronized")
    snapshot = D1MujocoTruthStateSource(plant).reset()
    assert plant.data.time == 0.
    return replace(snapshot, joint_velocity=np.tile([.2, -.4, .1, .3], 4))


def bind(controller, state, forward, yaw):
    controller.bind_raw_command(forward_velocity_mps=forward, yaw_rate_rps=yaw,
                                control_time_s=state.control_time_s)


@pytest.mark.parametrize("forward,raw_yaw,servo_yaw,active", [
    (0.,.6,.9,True), (-0.,-.6,-.9,True), (.25,.6,.9,False), (-.25,-.6,-.9,False),
    (0.,0.,1.,False), (-0.,-0.,-1.,False), (0.,0.,0.,False)])
def test_gate_preview_parent_once_and_compute_snapshot(state, monkeypatch, forward, raw_yaw, servo_yaw, active):
    c = TurnCenterCompensationController()
    bind(c,state,forward,raw_yaw)
    command = D1Command(forward_velocity_mps=forward,yaw_rate_rps=servo_yaw)
    before = c.control_memory.wheel_integral_nm.copy()
    nominal = c.nominal_targets(command,state)
    np.testing.assert_array_equal(nominal.nominal_wheel_speed_rad_s,c.nominal_targets(command,state).nominal_wheel_speed_rad_s)
    np.testing.assert_array_equal(c.control_memory.wheel_integral_nm,before)
    assert c.last_compensation is None and c.last_result is None
    calls = []
    original = D1WheelLegController.compute

    def parent(self,*args,**kwargs):
        value = original(self,*args,**kwargs)
        calls.append(value)
        return value

    monkeypatch.setattr(D1WheelLegController,"compute",parent)
    torque = c.compute(command,state,np.zeros(8))
    assert len(calls)==1 and torque is calls[0]
    record = c.last_compensation
    assert record.active == active
    assert record.raw_yaw_rate_rps == raw_yaw and record.servo_yaw_rate_rps == servo_yaw
    if not active:
        assert not np.any(record.correction_unclipped_rad_s)
        baseline = D1WheelLegController()
        baseline.compute(command,state,np.zeros(8))
        assert torque.tobytes() == baseline.last_result.torque_nm.tobytes()
        assert c.control_memory.wheel_integral_nm.tobytes() == baseline.control_memory.wheel_integral_nm.tobytes()
    bind(c,state,0.,0.)
    assert c.last_compensation is record and record.active == active
    c.reset()
    assert c.last_compensation is None and c.raw_binding is None
    assert not np.any(c.control_memory.wheel_integral_nm)


def test_horizontal_rotation_covariance_and_positive_correction(state):
    c = TurnCenterCompensationController()
    bind(c,state,0.,.6)
    command = D1Command(yaw_rate_rps=.6)
    c.compute(command,state,np.zeros(8))
    record = c.last_compensation
    expected = np.array([state.foot_jacobian[i,0,:3] @ state.joint_velocity[4*i:4*i+3] for i in range(4)])
    np.testing.assert_allclose(record.correction_unclipped_rad_s,expected/.087,rtol=0,atol=1e-12)
    angle = .7
    rotation = np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    turned = replace(state,base_rotation=rotation@state.base_rotation,
        base_linear_velocity_world=rotation@state.base_linear_velocity_world,
        base_angular_velocity_world=rotation@state.base_angular_velocity_world,
        foot_position=state.base_position+(state.foot_position-state.base_position)@rotation.T,
        foot_jacobian=np.einsum("ij,kjl->kil",rotation,state.foot_jacobian))
    candidate = TurnCenterCompensationController()
    bind(candidate,turned,0.,.6)
    candidate.compute(command,turned,np.zeros(8))
    np.testing.assert_allclose(candidate.last_compensation.corrected_target_rad_s,
                               record.corrected_target_rad_s,rtol=0,atol=1e-12)


def test_inactive_returns_exact_parent_nominal_object(state,monkeypatch):
    original = D1WheelLegController.nominal_targets
    returned = []

    def parent(self,*args,**kwargs):
        value = original(self,*args,**kwargs)
        returned.append(value)
        return value

    monkeypatch.setattr(D1WheelLegController,"nominal_targets",parent)
    c = TurnCenterCompensationController()
    bind(c,state,0.,0.)
    assert c.nominal_targets(D1Command(yaw_rate_rps=1.),state) is returned[-1]


def test_zero_center_motion_and_final_target_clip_original_antiwindup(state):
    c = TurnCenterCompensationController()
    bind(c,state,0.,.6)
    zero = replace(state,joint_velocity=np.zeros(16))
    c.compute(D1Command(yaw_rate_rps=.6),zero,np.zeros(8))
    assert not np.any(c.last_compensation.correction_unclipped_rad_s)
    np.testing.assert_array_equal(c.last_compensation.base_target_rad_s,c.last_compensation.corrected_target_rad_s)
    jac = np.zeros((4,3,4))
    jac[:,0,0] = 1.
    velocity = np.zeros(16)
    velocity[::4] = 100.
    changed = replace(state,foot_jacobian=jac,joint_velocity=velocity)
    c.reset()
    bind(c,changed,0.,.6)
    c.compute(D1Command(yaw_rate_rps=.6),changed,np.zeros(8))
    record = c.last_compensation
    np.testing.assert_array_equal(record.corrected_target_rad_s,np.full(4,30.))
    assert np.all(record.target_clipped)
    np.testing.assert_array_equal(c.control_memory.wheel_integral_nm,np.zeros(4))
    np.testing.assert_array_equal(c.last_result.torque_nm[3::4],JOINT_TORQUE_LIMIT[3::4])


@pytest.mark.parametrize("action", [np.ones(8)*1e-300,np.zeros(8,dtype=bool),np.zeros(8,dtype=complex),
                                    np.full(8,np.nan),np.zeros(7),["0"]*8])
def test_invalid_action_before_memory_mutation(state,action):
    c = TurnCenterCompensationController()
    bind(c,state,0.,.6)
    with pytest.raises((TypeError,ValueError)):
        c.compute(D1Command(yaw_rate_rps=.6),state,action)
    assert c.last_compensation is c.last_result is None
    assert not np.any(c.control_memory.wheel_integral_nm)


def test_missing_or_stale_binding_rejected(state):
    c = TurnCenterCompensationController()
    with pytest.raises((ValueError,RuntimeError)):
        c.nominal_targets(D1Command(),state)
    c.bind_raw_command(forward_velocity_mps=0.,yaw_rate_rps=.6,control_time_s=1.)
    with pytest.raises((ValueError,RuntimeError)):
        c.compute(D1Command(yaw_rate_rps=.6),state,np.zeros(8))
    assert c.last_result is None


def test_next_preview_preserves_execution_record_across_pulse_boundaries(state):
    c = TurnCenterCompensationController()
    for tick, raw_yaw in ((199,0.),(200,.6),(249,.6),(250,0.)):
        now = replace(state,control_time_s=tick*.01+1e-11,measurement_time_s=tick*.01)
        c.bind_raw_command(forward_velocity_mps=0.,yaw_rate_rps=raw_yaw,control_time_s=tick*.01)
        c.compute(D1Command(yaw_rate_rps=.8),now,np.zeros(8))
        record = c.last_compensation
        assert record.active == (raw_yaw != 0.) and record.control_time_s == tick*.01
        memory = c.control_memory.wheel_integral_nm.copy()
        next_time = (tick+1)*.01
        next_state = replace(state,control_time_s=next_time,measurement_time_s=next_time)
        c.bind_raw_command(forward_velocity_mps=0.,yaw_rate_rps=.6 if 200 <= tick+1 < 250 else 0.,
                            control_time_s=next_time)
        c.nominal_targets(D1Command(yaw_rate_rps=.8),next_state)
        assert c.last_compensation is record
        np.testing.assert_array_equal(c.control_memory.wheel_integral_nm,memory)


def test_disabled_unbound_controller_preserves_parent_schema_and_torque(state):
    candidate = TurnCenterCompensationController(enabled=False)
    baseline = D1WheelLegController()
    command = D1Command(yaw_rate_rps=.8)
    a = candidate.compute(command,state,np.zeros(8))
    b = baseline.compute(command,state,np.zeros(8))
    assert candidate.control_schema == baseline.control_schema
    assert a.tobytes() == b.tobytes()
    assert not candidate.last_compensation.active


def test_callback_prepare_cache_and_plane_binding(tmp_path):
    calls = []
    command = D1MotionCommand(yaw_rate_rps=.6)

    def source(time_s):
        calls.append(time_s)
        return command

    env = TurnCenterEnv(command_source=source,diagnostic_output=tmp_path,episode_seconds=.01)
    try:
        observation,info = env.reset(seed=55101)
        assert env.plant.data.time == 0. and observation.shape==(85,)
        assert env.loop.controller is env._controller and env.loop.plant is env.plant
        assert env.heading_decision.user_command is command
        assert calls==[0.] and env.raw_callback_count==1
        env._prepare()
        env._prepare()
        assert calls==[0.] and env.raw_callback_count==1
        assert env._controller.controller.last_compensation is None
        config = info['episode_metadata']['heading_task_config']
        assert config['task_schema']==env.task_schema
        assert config['turn_center_compensation']['coefficient']==1.
    finally:
        env.close()


@pytest.mark.parametrize("source_type", [D1HeadingTrackingEnv,D1FlatPlaneHeadingEnv,PlaneStopDampingEnv])
@pytest.mark.parametrize("loader,match", [(load_heading_policy,"heading_task_config"),
                                          (load_locomotion_policy,"task_schema")])
def test_prior_checkpoint_rejected_before_deserialization(tmp_path,monkeypatch,source_type,loader,match):
    from stable_baselines3 import PPO

    def forbidden(*args,**kwargs):
        pytest.fail("incompatible checkpoint deserialized")

    monkeypatch.setattr(PPO,"load",forbidden)
    kwargs = {'diagnostic_output':None} if source_type is PlaneStopDampingEnv else {}
    source = source_type(episode_seconds=.01,**kwargs)
    target = TurnCenterEnv(command_source=lambda t:D1MotionCommand(),diagnostic_output=tmp_path,episode_seconds=.01)
    try:
        source.reset(seed=55101)
        target.reset(seed=55101)
        sidecar = tmp_path/'old_checkpoint.json'
        sidecar.write_text(json.dumps({**_environment_contract(source),'recorded_episode':source.episode_metadata,'model_sha256':'not-read'}))
        with pytest.raises(ValueError,match=match):
            loader(tmp_path/'absent.zip',sidecar,target)
        assert source.plant.data.time==target.plant.data.time==0.
    finally:
        source.close()
        target.close()
