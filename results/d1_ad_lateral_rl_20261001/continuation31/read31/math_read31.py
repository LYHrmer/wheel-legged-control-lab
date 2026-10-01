"""Independent NumPy arithmetic for saved C31 events and learner updates.

No production observation, controller, learner, Torch or MuJoCo imports.
Record/physics eligibility and actual ledger joins belong to the saved reader.
"""
from __future__ import annotations

import math
import numpy as np

PARAMETER_SHAPES = {'actor.weight': (3, 54), 'actor.bias': (3,),
                    'value.weight': (1, 54), 'value.bias': (1,)}
REWARD_WEIGHTS = {'progress_potential_difference': 10., 'elapsed_s': -1., 'backtrack_normalized': -.05,
                  'longitudinal_normalized_square_integral_s': -.02,
                  'lateral_goal_error_normalized_square_integral_s': -.02,
                  'yaw_error_normalized_square_integral_s': -.02,
                  'torque_normalized_square_integral_s': -.5}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite_array(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    need(np.isfinite(result).all() and (shape is None or result.shape == shape),
         'nonfinite or incorrectly shaped numerical evidence')
    return result


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def observation54(source):
    """Reconstruct the pre-integration, pre-projection observation from raw evidence."""
    q = finite_array(source['pre_qpos23'], (23,))
    v = finite_array(source['pre_qvel22'], (22,))
    mapping = source['compiled_joint_map31']
    ids = np.asarray(mapping['joint_ids'], dtype=int)
    qa = np.asarray(mapping['joint_qpos_addresses'], dtype=int)
    va = np.asarray(mapping['joint_dof_addresses'], dtype=int)
    actuator_ids = np.asarray(mapping['actuator_ids'], dtype=int)
    ranges = finite_array(mapping['jnt_range'])
    limited = np.asarray(mapping['jnt_limited'], dtype=bool)
    actuator_joint_ids = np.asarray(mapping['actuator_joint_ids'], dtype=int)
    need(all(a.shape == (16,) and len(set(a.tolist())) == 16 for a in (ids, qa, va, actuator_ids))
         and np.all((qa >= 7) & (qa < 23)) and np.all((va >= 6) & (va < 22))
         and np.array_equal(actuator_joint_ids, ids)
         and ranges.shape == (16, 2) and limited.shape == (16,),
         'actuator/joint address binding differs')
    # The saved map stores ranges already selected in the 16 actuator order.
    # The full reader independently joins that map to actual compiled arrays.
    leg_slots = np.arange(16).reshape(4, 4)[:, :3].ravel()
    leg_qa = qa.reshape(4, 4)[:, :3].ravel()
    bounds = ranges[leg_slots]
    need(bounds.shape == (12, 2) and limited[leg_slots].all()
         and np.all(bounds[:, 1] > bounds[:, 0])
         and np.array_equal(mapping['leg_qpos_addresses'], leg_qa)
         and np.array_equal(mapping['leg_range_mid'], bounds.mean(axis=1))
         and np.array_equal(mapping['leg_range_half'], (bounds[:, 1]-bounds[:, 0])/2),
         'leg position normalization lacks finite limited ranges')
    direction, leg, macro = source['direction'], source['leg_index'], source['macro_index']
    completed = source['completed_side_controls']
    need(direction in (-1, 1) and type(leg) is int and 0 <= leg < 4
         and type(macro) is int and 0 <= macro < 4
         and type(completed) is int and completed >= 0
         and abs(source['side_elapsed_s']-.01*completed) <= 1e-10,
         'invalid actual leg/macro/direction/pre-control elapsed boundary')
    yaw0 = float(source['initial_yaw_rad'])
    cy, sy = math.cos(yaw0), math.sin(yaw0)
    initial_rotation = np.array([[cy, sy], [-sy, cy]])
    quaternion = q[3:7]
    norm = float(np.linalg.norm(quaternion))
    need(abs(norm-1.) <= 1e-3, 'nonunit base quaternion')
    w, x, y, z = quaternion
    rpy = np.array([math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                    math.asin(float(np.clip(2*(w*y-z*x), -1., 1.))),
                    wrap(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))-yaw0)])
    raw = np.zeros(54, dtype=np.float64)
    raw[0], raw[1+leg], raw[5] = direction, 1., macro
    raw[6:8] = initial_rotation@(q[:2]-finite_array(source['initial_xy_m'], (2,)))
    raw[8] = q[2]-.455
    raw[9:12] = rpy
    raw[12:14], raw[14] = initial_rotation@v[:2], v[2]
    raw[15:18] = v[3:6]  # Original free-joint coordinates, not COM velocity.
    raw[18:30] = q[leg_qa]
    raw[30:46] = v[va]
    span = finite_array(source['teacher_body_to_m'], (3,))-finite_array(source['teacher_body_from_m'], (3,))
    raw[46:48] = initial_rotation@span[:2]
    raw[48], raw[49] = source['teacher_shift_time_s'], source['plan_com_height_m']
    raw[50:53] = finite_array(source['previous_consumed_action'], (3,))
    raw[53] = .01*completed
    need(np.isfinite(raw).all(), 'nonfinite raw54')
    normalized = raw.copy()
    normalized[5] /= 3.
    normalized[6:8] /= .12
    normalized[8] /= .05
    normalized[9:12] /= .12
    normalized[12:15] /= [.2, .2, .1]
    normalized[15:18] /= .5
    normalized[18:30] = (raw[18:30]-bounds.mean(axis=1))/((bounds[:, 1]-bounds[:, 0])/2)
    normalized[30:46] /= [4., 4., 4., 20.]*4
    normalized[46:48] /= .12
    normalized[48] /= 1.5
    normalized[49] /= .5
    normalized[50:53] /= [.012, .012, 1.]
    normalized[53] /= 15.
    mask = np.zeros(54, dtype=bool)
    mask[5:] = (normalized[5:] < -5.) | (normalized[5:] > 5.)
    normalized[5:] = np.clip(normalized[5:], -5., 5.)
    return dict(raw54=raw, normalized54=normalized, clip_mask54=mask)


def action3(teacher_from, teacher_to, initial_yaw, latent, mode='learned'):
    need(mode in ('zero', 'fixed', 'learned', 'train'), 'unknown action source')
    if mode == 'zero':
        return dict(fixed_base=np.zeros(3), proposed=np.zeros(3), projected=np.zeros(3))
    origin, target = finite_array(teacher_from, (3,)), finite_array(teacher_to, (3,))
    toward = origin[:2]-target[:2]
    distance = float(np.linalg.norm(toward))
    world = np.zeros(2) if distance == 0 else min(.008, distance)*toward/distance
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    base = np.array([c*world[0]+s*world[1], -s*world[0]+c*world[1], .5])
    if mode == 'fixed':
        proposed = base.copy()
    else:
        z = finite_array(latent, (3,))
        proposed = base.copy() if np.array_equal(z, np.zeros(3)) else np.r_[
            base[:2]+.004*np.tanh(z[:2]), .5+.5*math.tanh(float(z[2]))]
    projected = proposed.copy()
    projected[:2] = np.clip(projected[:2], -.012, .012)
    return dict(fixed_base=base, proposed=proposed, projected=projected)


def gaussian_log_probability(latent, mean, std=.35):
    z, mu = finite_array(latent), finite_array(mean)
    need(z.shape == mu.shape and z.shape[-1] == 3 and std == .35, 'latent Gaussian evidence shape/std')
    return -.5*np.sum(((z-mu)/std)**2+2*math.log(std)+math.log(2*math.pi), axis=-1)


def gaussian_mean_kl(old_mean, new_mean, std=.35):
    old, new = finite_array(old_mean), finite_array(new_mean)
    need(old.shape == new.shape and old.ndim == 2 and old.shape[1] == 3 and len(old) > 0
         and std == .35, 'Gaussian KL rows/std')
    return float(np.mean(np.sum((old-new)**2, axis=1)/(2*std*std)))


def reward31(components, outcome, cycle_elapsed_s, cycle_safe_complete):
    """outcome is next_leg/success/controlled_failure; unsafe cycles never train."""
    need(cycle_safe_complete is True and outcome in ('next_leg', 'success', 'controlled_failure'),
         'incomplete/unsafe/budget/software cycle is not an eligible learning transition')
    need(math.isfinite(cycle_elapsed_s) and 0 <= cycle_elapsed_s <= 22., 'invalid actual cycle duration')
    values = {key: float(components[key]) for key in REWARD_WEIGHTS}
    need(all(math.isfinite(v) for v in values.values()) and values['elapsed_s'] > 0,
         'invalid actual macro components')
    adjustment = (5. if outcome == 'success' else
                  -50.-max(0., 22.-cycle_elapsed_s) if outcome == 'controlled_failure' else 0.)
    return dict(component_reward=sum(REWARD_WEIGHTS[k]*v for k, v in values.items()),
                terminal_adjustment=adjustment,
                reward=sum(REWARD_WEIGHTS[k]*v for k, v in values.items())+adjustment)


def advantages31(rewards, old_values, elapsed_controls, cycle_ids, terminal):
    rewards, values = finite_array(rewards), finite_array(old_values)
    counts, cycles, done = np.asarray(elapsed_controls), np.asarray(cycle_ids), np.asarray(terminal)
    n = len(rewards)
    need(0 < n <= 32 and all(a.shape == (n,) for a in (rewards, values, counts, cycles, done))
         and counts.dtype.kind in 'iu' and np.all(counts > 0) and done.dtype.kind == 'b',
         'GAE needs actual one-to-32 closed-cycle events with positive integer durations')
    need(done[-1] and all(bool(done[i]) == bool(cycles[i] != cycles[i+1]) for i in range(n-1)),
         'GAE termination must exactly separate complete cycles')
    need(len(set(cycles.tolist())) == 1+sum(cycles[i] != cycles[i+1] for i in range(n-1)),
         'cycle IDs recur across rollout boundaries')
    delta, gae = np.zeros(n), np.zeros(n)
    for i in reversed(range(n)):
        delta[i] = rewards[i]+(0. if done[i] else values[i+1])-values[i]
        gae[i] = delta[i]+(0. if done[i] else .95**(.01*int(counts[i]))*gae[i+1])
    centered = gae-float(gae.mean())
    population_std = math.sqrt(float(np.mean(centered**2)))
    return dict(delta=delta, gae=gae, value_target=gae+values,
                normalized_advantage=centered/(population_std+1e-8),
                population_std=population_std)


def checked_parameters(parameters):
    need(set(parameters) == set(PARAMETER_SHAPES), 'unexpected/missing learned parameters')
    return {key: finite_array(parameters[key], shape) for key, shape in PARAMETER_SHAPES.items()}


def linear_outputs(parameters, observations):
    p, x = checked_parameters(parameters), finite_array(observations)
    need(x.ndim == 2 and x.shape[1] == 54 and len(x) > 0, 'actual normalized observation rows')
    return x@p['actor.weight'].T+p['actor.bias'], (x@p['value.weight'].T+p['value.bias'])[:, 0]


def ppo_objective31(parameters, observations, latent, old_log_probability, advantages, targets):
    x = finite_array(observations)
    mean, value = linear_outputs(parameters, x)
    n = len(x)
    z = finite_array(latent, (n, 3))
    old_logp, adv, target = (finite_array(v, (n,)) for v in (old_log_probability, advantages, targets))
    logp = gaussian_log_probability(z, mean)
    ratio = np.exp(logp-old_logp)
    need(np.isfinite(ratio).all(), 'nonfinite PPO ratio')
    clipped = np.clip(ratio, .8, 1.2)
    actor_loss = -float(np.mean(np.minimum(ratio*adv, clipped*adv)))
    value_loss = float(np.mean((value-target)**2))
    active = ((adv >= 0) & (ratio <= 1.2)) | ((adv < 0) & (ratio >= .8))
    dmean = -(adv*ratio*active/n)[:, None]*(z-mean)/(.35**2)
    dvalue = (value-target)/n  # .5 * MSE has derivative (value-target)/n.
    gradients = {'actor.weight': dmean.T@x, 'actor.bias': dmean.sum(axis=0),
                 'value.weight': (dvalue@x)[None, :], 'value.bias': np.array([dvalue.sum()])}
    need(all(np.isfinite(v).all() for v in gradients.values()), 'nonfinite analytic gradient')
    return dict(actor_loss=actor_loss, value_mse=value_loss, total_loss=actor_loss+.5*value_loss,
                mean=mean, value=value, log_probability=logp, ratio=ratio, raw_gradients=gradients)


def clip_groups31(gradients):
    raw = checked_parameters(gradients)
    clipped, groups = {}, {}
    for group in ('actor', 'value'):
        names = (group+'.weight', group+'.bias')
        norm = math.sqrt(sum(float(np.sum(raw[key]**2)) for key in names))
        factor = min(1., .5/(norm+1e-6))
        groups[group] = dict(raw_norm=norm, clip_factor=factor, clipped_norm=norm*factor)
        for key in names:
            clipped[key] = raw[key]*factor
    return dict(gradients=clipped, groups=groups)


def adam_step31(parameters, gradients, before):
    """One Adam call with two groups, no weight decay or AMSGrad, float64."""
    p, g = checked_parameters(parameters), checked_parameters(gradients)
    need(set(before) == set(p), 'Adam state keys differ')
    result, after = {}, {}
    steps = []
    for key in p:
        state = before[key]
        need(type(state['step']) is int and 0 <= state['step'] < 64, 'Adam actual per-parameter step')
        m = finite_array(state['exp_avg'], p[key].shape)
        v = finite_array(state['exp_avg_sq'], p[key].shape)
        need(np.all(v >= 0), 'negative Adam variance')
        step = state['step']+1
        m = .9*m+.1*g[key]
        v = .999*v+.001*g[key]**2
        result[key] = p[key]-.001*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        after[key] = dict(step=step, exp_avg=m, exp_avg_sq=v)
        steps.append(step)
    need(len(set(steps)) == 1, 'actor/value Adam step counters diverged')
    return dict(parameters=result, adam=after, optimizer_calls=1)
