"""Independent analytical fixtures; no production modules or model execution."""
import math

import numpy as np
import pytest

from math_read31 import (PARAMETER_SHAPES, REWARD_WEIGHTS, action3, adam_step31,
                         advantages31, clip_groups31, gaussian_log_probability,
                         observation54, ppo_objective31, reward31)


def parameters():
    return {key: np.zeros(shape) for key, shape in PARAMETER_SHAPES.items()}


def test_duration_gae_does_not_cross_reset_and_uses_population_std():
    got = advantages31([1., 2., 10.], [.5, .25, 3.], [100, 400, 50], [7, 7, 8], [False, True, True])
    expected = np.array([.75+.95*1.75, 1.75, 7.])
    np.testing.assert_allclose(got['gae'], expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(got['value_target'], expected+[.5, .25, 3.])
    assert abs(got['normalized_advantage'].mean()) < 1e-14
    with pytest.raises(ValueError):
        advantages31([1., 2.], [0., 0.], [100, 100], [0, 1], [False, True])


def test_latent_probability_survives_action_projection_and_zero_matches_fixed():
    origin, target = [0., 0., .455], [.04, 0., .455]
    zero = action3(origin, target, 0., [0., 0., 0.])
    fixed = action3(origin, target, 0., None, mode='fixed')
    assert zero['proposed'].tobytes() == fixed['proposed'].tobytes()
    assert np.array_equal(action3(origin, target, 0., None, mode='zero')['projected'], np.zeros(3))
    logp = gaussian_log_probability([[.35, 0., 0.]], [[0., 0., 0.]])[0]
    assert abs(logp-(-.5-3*math.log(.35*math.sqrt(2*math.pi)))) < 1e-14
    far = action3(origin, target, 0., [-50., 50., 50.])['projected']
    assert np.all(np.abs(far[:2]) <= .012) and 0 <= far[2] <= 1


def test_failure_time_charge_and_unsafe_cycle_rejection():
    components = {key: 0. for key in REWARD_WEIGHTS}
    components['elapsed_s'] = 4.
    early = reward31(components, 'controlled_failure', 4., True)
    components['elapsed_s'] = 8.
    late = reward31(components, 'controlled_failure', 8., True)
    assert early['reward'] == late['reward'] == -72.
    with pytest.raises(ValueError):
        reward31(components, 'controlled_failure', 8., False)


def test_analytic_actor_and_value_gradients_against_loss_perturbations():
    p = parameters()
    obs = np.zeros((2, 54))
    obs[:, 0] = [1., -.5]
    latent = np.array([[.1, -.2, .15], [-.2, .05, .1]])
    old = gaussian_log_probability(latent, np.zeros((2, 3)))
    args = (obs, latent, old, np.array([1., -1.]), np.array([2., -1.]))
    result = ppo_objective31(p, *args)
    for name, index in (('actor.weight', (0, 0)), ('actor.bias', (1,)),
                        ('value.weight', (0, 0)), ('value.bias', (0,))):
        plus, minus = {k: v.copy() for k, v in p.items()}, {k: v.copy() for k, v in p.items()}
        plus[name][index], minus[name][index] = 1e-6, -1e-6
        derivative = (ppo_objective31(plus, *args)['total_loss']-ppo_objective31(minus, *args)['total_loss'])/2e-6
        assert abs(derivative-result['raw_gradients'][name][index]) < 1e-8


def test_separate_group_clipping_and_actual_first_adam_step():
    p = parameters()
    gradients = parameters()
    gradients['actor.bias'][0] = 1.
    gradients['value.bias'][0] = 1000.
    clip = clip_groups31(gradients)
    assert clip['gradients']['actor.bias'][0] > .49
    assert clip['groups']['actor']['clip_factor'] > 900*clip['groups']['value']['clip_factor']
    before = {key: dict(step=0, exp_avg=np.zeros_like(v), exp_avg_sq=np.zeros_like(v)) for key, v in p.items()}
    got = adam_step31(p, clip['gradients'], before)
    assert got['optimizer_calls'] == 1
    assert all(state['step'] == 1 for state in got['adam'].values())
    assert -.001 < got['parameters']['actor.bias'][0] < -.00099


def test_observation_uses_compiled_address_order_and_pre_control_elapsed():
    q, v = np.zeros(23), np.zeros(22)
    q[2], q[3] = .455, 1.
    q[7:23] = np.arange(16)*.1
    v[6:22] = np.arange(16)
    order = np.arange(16)[::-1]
    mapping = dict(joint_ids=order+1, joint_qpos_addresses=order+7,
                   joint_dof_addresses=order+6, actuator_ids=order,
                   actuator_joint_ids=order+1,
                   jnt_range=np.tile([-2., 2.], (16, 1)), jnt_limited=np.ones(16, dtype=bool),
                   leg_qpos_addresses=(order+7).reshape(4, 4)[:, :3].ravel(),
                   leg_range_mid=np.zeros(12), leg_range_half=np.full(12, 2.))
    source = dict(pre_qpos23=q, pre_qvel22=v, compiled_joint_map31=mapping,
                  initial_xy_m=[0., 0.], initial_yaw_rad=0., direction=-1, leg_index=2,
                  macro_index=0, teacher_body_from_m=[0., 0., .455], teacher_body_to_m=[.03, 0., .455],
                  teacher_shift_time_s=1.5, plan_com_height_m=.5, previous_consumed_action=[0., 0., 0.],
                  completed_side_controls=0, side_elapsed_s=0.)
    got = observation54(source)
    assert got['raw54'][18] == 1.5 and got['raw54'][30] == 15.
    assert got['normalized54'][18] == .75 and got['normalized54'][53] == 0.
    source['side_elapsed_s'] = .01
    with pytest.raises(ValueError):
        observation54(source)
