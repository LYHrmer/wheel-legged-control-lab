"""Independent saved C31 rollout/GAE/minibatch/gradient/Adam audit, NumPy only."""
from __future__ import annotations

import math
import numpy as np

from math_read31 import (PARAMETER_SHAPES, REWARD_WEIGHTS, need, finite_array,
    observation54, gaussian_log_probability, gaussian_mean_kl, linear_outputs,
    reward31, advantages31, ppo_objective31, clip_groups31, adam_step31)

SCHEMA = 'd1-c31-event-smdp-ppo-v1'


def same(actual, expected, message):
    a, b = finite_array(actual), finite_array(expected)
    need(a.shape == b.shape and np.allclose(a, b, rtol=2e-10, atol=2e-10), message)


def unpack_parameters(record):
    need(set(record) == {'actor', 'value'}, 'parameter groups differ')
    parameters, adam = {}, {}
    for group in ('actor', 'value'):
        need(set(record[group]) == {'weight', 'bias'}, 'unexpected layer parameter')
        for leaf in ('weight', 'bias'):
            name = group+'.'+leaf
            row = record[group][leaf]
            parameters[name] = finite_array(row['parameter'], PARAMETER_SHAPES[name])
            adam[name] = dict(step=row['adam_step'], exp_avg=row['exp_avg'], exp_avg_sq=row['exp_avg_sq'])
            need(type(row['adam_step']) is int and 0 <= row['adam_step'] <= 64, 'Adam step is not actual integer')
            finite_array(row['exp_avg'], PARAMETER_SHAPES[name])
            finite_array(row['exp_avg_sq'], PARAMETER_SHAPES[name])
    return parameters, adam


def zero_state31():
    parameters = {key: np.zeros(shape) for key, shape in PARAMETER_SHAPES.items()}
    adam = {key: dict(step=0, exp_avg=np.zeros(shape), exp_avg_sq=np.zeros(shape))
            for key, shape in PARAMETER_SHAPES.items()}
    return parameters, adam


def compare_state(record, expected_parameters, expected_adam):
    parameters, adam = unpack_parameters(record)
    for key in PARAMETER_SHAPES:
        same(parameters[key], expected_parameters[key], 'parameter chain differs: '+key)
        need(adam[key]['step'] == expected_adam[key]['step'], 'actual Adam step chain differs')
        for field in ('exp_avg', 'exp_avg_sq'):
            same(adam[key][field], expected_adam[key][field], 'Adam moment chain differs: '+key+'/'+field)
    return parameters, adam


def verify_rollout31(cycles, batch, rollout_parameters):
    """cycles must already carry independently established physical eligibility.

    Caller binds original archived macros/eligibility, including every failed
    cycle; this function never obtains eligibility from learner self-reports.
    """
    need(batch['schema'] == SCHEMA and len(cycles) == 8
         and batch['cycle_indices'] == [c['cycle_index'] for c in cycles], 'eight actual closed-cycle batch differs')
    need(len(set(batch['cycle_indices'])) == 8, 'reused cycle in rollout')
    expected = []
    for cycle in cycles:
        need(cycle['independent_learning_eligible'] is True
             and cycle['terminal_kind'] in ('success', 'controlled_failure')
             and 1 <= len(cycle['macros']) <= 4, 'unsafe/incomplete cycle entered PPO')
        previous_end = None
        for mi, macro in enumerate(cycle['macros']):
            latch = macro['skill31_latch']
            source = latch['observation31']
            need(latch['mode'] == 'train' and latch['cycle_index'] == cycle['cycle_index']
                 and source['macro_index'] == mi, 'actual training latch identity differs')
            obs = observation54(source)
            same(source['raw54'], obs['raw54'], 'raw54 differs from actual pre-state')
            same(source['normalized54'], obs['normalized54'], 'normalized54 arithmetic differs')
            need(np.array_equal(source['clip_mask54'], obs['clip_mask54']), 'observation clipping mask differs')
            mean, value = linear_outputs(rollout_parameters, obs['normalized54'][None, :])
            same(latch['old_mean_z3'], mean[0], 'latch did not use this rollout actor')
            same(latch['old_value'], value[0], 'latch did not use this rollout value')
            need(latch['latent_std_z3'] == [.35]*3, 'latent standard deviation changed')
            logp = float(gaussian_log_probability(latch['latent_z3'], mean[0]))
            same(latch['old_log_prob'], logp, 'old likelihood is not sampled Gaussian latent density')
            begin, end = macro['control_range']
            count = end-begin
            need(type(begin) is int and type(end) is int and count > 0
                 and macro['elapsed_controls'] == count and latch['local_control_index'] == begin
                 and (previous_end is None and begin == 200 or previous_end == begin)
                 and macro.get('truncated') is not True, 'actual macro range/elapsed/closure differs')
            previous_end = end
            terminal = mi == len(cycle['macros'])-1
            components = macro['reward_components']
            same(components['elapsed_s'], .01*count, 'reward duration is not actual controls')
            outcome = cycle['terminal_kind'] if terminal else 'next_leg'
            outcome = 'controlled_failure' if outcome == 'controlled_failure' else outcome
            reward = reward31(components, outcome, cycle['actual_seconds'], True)
            expected.append(dict(cycle_index=cycle['cycle_index'], macro_index=mi,
                control_start=begin, control_end=end, elapsed_controls=count, duration_s=.01*count,
                observation54=obs['normalized54'], latent_z3=latch['latent_z3'], old_mean_z3=mean[0],
                old_log_prob=logp, old_value=float(value[0]), reward_components=components,
                terminal_adjustment=reward['terminal_adjustment'], reward=reward['reward'], terminal=terminal))
        same(cycle['actual_seconds'], .01*previous_end, 'whole-cycle duration omits preparation or retention')
    rows = batch['rows']
    need(batch['macro_count'] == len(rows) == len(expected) <= 32, 'real macro batch count differs')
    gae = advantages31([r['reward'] for r in expected], [r['old_value'] for r in expected],
                       [r['elapsed_controls'] for r in expected], [r['cycle_index'] for r in expected],
                       [r['terminal'] for r in expected])
    for i, (saved, row) in enumerate(zip(rows, expected)):
        for key in ('cycle_index', 'macro_index', 'control_start', 'control_end', 'elapsed_controls', 'terminal'):
            need(saved[key] == row[key], 'prepared batch identity differs: '+key)
        for key in ('duration_s', 'observation54', 'latent_z3', 'old_mean_z3', 'old_log_prob',
                    'old_value', 'terminal_adjustment', 'reward'):
            same(saved[key], row[key], 'prepared numeric row differs: '+key)
        for key in REWARD_WEIGHTS:
            same(saved['reward_components'][key], row['reward_components'][key], 'physical cost component replaced')
        for key in ('delta', 'gae', 'value_target', 'normalized_advantage'):
            same(saved[key], gae[key][i], 'actual duration GAE/target differs: '+key)
        same(saved['gae_factor'], .95**row['duration_s'], 'GAE uses wrong time basis')
    same(batch['advantage_mean'], float(gae['gae'].mean()), 'advantage mean differs')
    same(batch['advantage_divisor'], gae['population_std']+1e-8, 'advantage normalization differs')
    return dict(passed=True, actual_cycles=8, actual_macros=len(rows), gae=gae,
                old_policy_checked_against_pre_batch_parameters=True)


def verify_update31(batch, update, parameters, adam, *, batch_index, permutation_seed):
    """Audit recorded order as permutations; do not claim a NumPy RNG equals Torch."""
    rows = batch['rows']
    n = len(rows)
    need(update['schema'] == SCHEMA and update['batch_index'] == batch_index
         and update['macro_count'] == n and update['permutation_seed'] == permutation_seed,
         'update identity/seed/real rows differ')
    observations = finite_array([r['observation54'] for r in rows], (n, 54))
    latent = finite_array([r['latent_z3'] for r in rows], (n, 3))
    old_mean = finite_array([r['old_mean_z3'] for r in rows], (n, 3))
    old_logp = finite_array([r['old_log_prob'] for r in rows], (n,))
    advantage = finite_array([r['normalized_advantage'] for r in rows], (n,))
    target = finite_array([r['value_target'] for r in rows], (n,))
    attempts = update['attempts']
    need(1 <= len(attempts) <= 2*math.ceil(n/8), 'extra/missing minibatch attempts')
    counts = dict(actual_evaluated_minibatches=len(attempts), actual_backward=0,
                  actual_optimizer_steps=0, actual_pre_actor_rows=0,
                  actual_pre_value_rows=0, actual_post_kl_actor_rows=0)
    epochs = {0: [], 1: []}
    stopped = False
    actor_changed = actor_gradient = False
    for ai, attempt in enumerate(attempts):
        epoch = ai//math.ceil(n/8)
        mini = ai%math.ceil(n/8)
        ids = attempt['indices']
        expected_size = min(8, n-8*mini)
        need(not stopped and attempt['batch_index'] == batch_index and attempt['epoch'] == epoch
             and attempt['minibatch'] == mini and len(ids) == expected_size
             and all(type(i) is int and 0 <= i < n for i in ids)
             and len(set(ids)) == len(ids) and not(set(ids) & set(epochs[epoch])),
             'minibatch ordering/size repeats or invents real events')
        epochs[epoch].extend(ids)
        p, state = compare_state(attempt['parameters_before'], parameters, adam)
        mean, _ = linear_outputs(p, observations[ids])
        kl = gaussian_mean_kl(old_mean[ids], mean)
        same(attempt['pre_kl'], kl, 'pre-update minibatch analytical KL differs')
        need(attempt['actor_forward_rows'] == len(ids), 'pre-KL actor rows double-counted/omitted')
        counts['actual_pre_actor_rows'] += len(ids)
        if kl > .03:
            need(ai == len(attempts)-1 and attempt['normal_kl_stop'] is True
                 and attempt['optimizer_step'] is False and attempt['backward'] is False
                 and attempt['value_forward_rows'] == 0
                 and 'parameters_after' not in attempt and 'post_kl_all_rows' not in attempt,
                 'KL-rejected attempt ran value/backward/update or did not stop')
            stopped = True
            break
        need(attempt['optimizer_step'] is True and attempt['backward'] is True
             and attempt['value_forward_rows'] == len(ids), 'accepted minibatch lacks actual update')
        calculated = ppo_objective31(p, observations[ids], latent[ids], old_logp[ids], advantage[ids], target[ids])
        for key, value in (('policy', calculated['actor_loss']), ('value', .5*calculated['value_mse']),
                           ('total', calculated['total_loss'])):
            same(attempt['losses'][key], value, 'PPO loss differs: '+key)
        clip = clip_groups31(calculated['raw_gradients'])
        for group in ('actor', 'value'):
            record = attempt['gradients'][group]
            names = (group+'.weight', group+'.bias')
            for field, key in (('raw_norm', 'raw_norm'), ('factor', 'clip_factor'), ('clipped_norm', 'clipped_norm')):
                same(record[field], clip['groups'][group][key], 'independent gradient-group clipping differs')
            need(len(record['raw']) == len(record['clipped']) == len(record['actual_clipped']) == 2,
                 'gradient weight/bias evidence incomplete')
            for index, name in enumerate(names):
                same(record['raw'][index], calculated['raw_gradients'][name], 'analytical raw gradient differs')
                same(record['clipped'][index], clip['gradients'][name], 'claimed clipped gradient differs')
                same(record['actual_clipped'][index], clip['gradients'][name], 'actual clipped gradient differs')
        next_state = adam_step31(p, clip['gradients'], state)
        parameters, adam = compare_state(attempt['parameters_after'], next_state['parameters'], next_state['adam'])
        post_mean, _ = linear_outputs(parameters, observations)
        post_kl = gaussian_mean_kl(old_mean, post_mean)
        same(attempt['post_kl_all_rows'], post_kl, 'post-step full-batch KL differs')
        need(attempt['post_kl_actor_forward_rows'] == n, 'post-KL actor row count differs')
        counts['actual_backward'] += 1
        counts['actual_optimizer_steps'] += 1
        counts['actual_pre_value_rows'] += len(ids)
        counts['actual_post_kl_actor_rows'] += n
        actor_gradient |= clip['groups']['actor']['raw_norm'] > 0
        actor_changed |= any(not np.array_equal(parameters[k], p[k]) for k in ('actor.weight', 'actor.bias'))
        if post_kl > .10:
            need(ai == len(attempts)-1 and update['hard_stop'] == 'nonfinite_or_excessive_post_update_kl',
                 'hard post-KL breach did not stop this update')
            stopped = True
    if not stopped:
        need(all(sorted(ids) == list(range(n)) for ids in epochs.values()), 'non-stopped update omitted epoch rows')
    need(update['normal_kl_stop'] is bool(attempts[-1].get('normal_kl_stop', False)), 'normal KL stop summary differs')
    for key, value in counts.items():
        need(update[key] == value, 'actual learning call count differs: '+key)
    valid = counts['actual_optimizer_steps'] > 0 and update['hard_stop'] is None
    need(update['valid_train_call'] is valid, 'valid update summary differs')
    return dict(passed=True, valid_train_call=valid, counts=counts, parameters=parameters, adam=adam,
                actor_nonzero_gradient=actor_gradient, actor_parameter_changed=actor_changed,
                permutation_partition_verified=True, torch_random_permutation_replayed=False)
