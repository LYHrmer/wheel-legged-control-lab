"""One consumed, zero-physics final-critic diagnostic; never optimize or retry."""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


class NoPhysics(importlib.abc.MetaPathFinder):
    blocked = ('mujoco', 'pybullet', 'engine_binding', 'full_drive_env_08',
               'course_native_guard_08', 'world_upright_course_11')

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.blocked:
            raise RuntimeError('diagnostic forbids physics import: ' + fullname)
        return None


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if plan.get('status') != 'GO' or plan.get('schema') != 'd1-value14-plan-v1':
        raise ValueError('a reviewed frozen GO is required')
    if output.resolve() != Path(plan['output']).resolve():
        raise ValueError('output differs from the consumed plan')
    output.mkdir(exist_ok=False)
    write_json(output / 'reservation.json', {
        'plan': str(plan_path), 'plan_identity': identity(plan_path),
        'pid': os.getpid(), 'retry_permitted': False,
        'limits': plan['limits'], 'started_wall_ns': time.time_ns()})
    started = time.monotonic()
    counts = {key: 0 for key in (
        'ppo_load_attempted', 'ppo_load_returned', 'torch_load_attempted',
        'torch_load_returned', 'value_batches_attempted', 'value_batches_returned',
        'value_states_attempted', 'value_states_returned', 'head_batches_attempted',
        'head_batches_returned', 'head_states_attempted', 'head_states_returned',
        'backward_attempted', 'backward_returned', 'actor_forward_attempted',
        'optimizer_step_attempted', 'learn_attempted', 'train_attempted',
        'save_attempted', 'autograd_grad_attempted', 'clip_attempted')}
    journal = (output / 'journal.jsonl').open('x')
    report = {'schema': 'd1-value14-result-v1', 'status': 'failed',
              'plan_identity': identity(plan_path), 'counts': counts,
              'physics_controls': 0, 'native_steps': 0,
              'historical_GAE_reconstructed': False, 'on_policy_claim': False}

    def record(event):
        journal.write(json.dumps({'event': event, 'elapsed_s': time.monotonic()-started,
                                  'counts': counts}, sort_keys=True) + '\n')
        journal.flush()
        os.fsync(journal.fileno())

    def bounded(key, amount, cap):
        if counts[key] + amount > cap:
            raise RuntimeError('budget would be exceeded: ' + key)
        counts[key] += amount

    def forbidden(key):
        def deny(*args, **kwargs):
            counts[key] += 1
            record('forbidden:' + key)
            raise RuntimeError('forbidden diagnostic operation: ' + key)
        return deny

    def stop(signum, frame):
        raise TimeoutError('bounded diagnostic stop signal ' + str(signum))

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(plan['limits']['worker_wall_s'])
    def mapped_compute_libraries():
        frozen = {str(Path(name).resolve()) for name in plan['inputs']}
        mapped = set()
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6:
                continue
            name = fields[5]
            if '/torch/' not in name and '/nvidia/' not in name:
                continue
            if name.endswith(' (deleted)'):
                raise RuntimeError('deleted compute library mapping')
            actual = str(Path(name).resolve())
            if actual not in frozen:
                raise RuntimeError('unfrozen compute library mapping: ' + actual)
            mapped.add(actual)
        return sorted(mapped)

    model = None
    primary_error = None
    try:
        for name, expected in plan['inputs'].items():
            if identity(name) != expected:
                raise RuntimeError('frozen input differs: ' + name)
        if os.environ.get('LD_PRELOAD') or os.environ.get('LD_LIBRARY_PATH'):
            raise RuntimeError('diagnostic requires a clean loader environment')
        sys.meta_path.insert(0, NoPhysics())
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.base_class import BaseAlgorithm
        from stable_baselines3.common.policies import ActorCriticPolicy
        from rl16_learning_11 import hash_state
        from value14_math import diagnostic_stats

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError('thread contract differs')
        report['compute_library_maps_before_load'] = mapped_compute_libraries()
        record('preflight_complete')
        original_load = torch.load
        original_backward = torch.autograd.backward

        def checked_load(*args, **kwargs):
            bounded('torch_load_attempted', 1, 3)
            record('torch_load_entry')
            value = original_load(*args, **kwargs)
            counts['torch_load_returned'] += 1
            return value

        def checked_backward(*args, **kwargs):
            bounded('backward_attempted', 1, 4)
            record('backward_entry')
            value = original_backward(*args, **kwargs)
            counts['backward_returned'] += 1
            record('backward_return')
            return value

        torch.load = checked_load
        torch.autograd.backward = checked_backward
        torch.autograd.grad = forbidden('autograd_grad_attempted')
        torch.save = forbidden('save_attempted')
        BaseAlgorithm.save = forbidden('save_attempted')
        PPO.learn = forbidden('learn_attempted')
        PPO.train = forbidden('train_attempted')
        torch.optim.Adam.step = forbidden('optimizer_step_attempted')
        torch.optim.Optimizer.step = forbidden('optimizer_step_attempted')
        torch.nn.utils.clip_grad_norm_ = forbidden('clip_attempted')
        ActorCriticPolicy.forward = forbidden('actor_forward_attempted')
        ActorCriticPolicy._predict = forbidden('actor_forward_attempted')
        ActorCriticPolicy.evaluate_actions = forbidden('actor_forward_attempted')
        bounded('ppo_load_attempted', 1, 1)
        record('ppo_load_entry')
        model = PPO.load(plan['model'], env=None, device='cpu')
        counts['ppo_load_returned'] += 1
        policy = model.policy
        policy.set_training_mode(False)
        policy.optimizer.step = forbidden('optimizer_step_attempted')
        metadata = json.loads(Path(plan['metadata']).read_text())
        expected_state = metadata['training_receipt']['final_hashes']
        initial_policy = hash_state('policy', policy.state_dict())
        initial_optimizer = hash_state('optimizer', policy.optimizer.state_dict())
        if initial_policy != expected_state['policy_state']:
            raise RuntimeError('loaded policy is not the recorded training final')
        if initial_optimizer != expected_state['optimizer_state']:
            raise RuntimeError('loaded optimizer differs from the recorded final')
        if (model.gamma != .99 or model.gae_lambda != .95
                or model.vf_coef != .5 or model.max_grad_norm != .5
                or model.clip_range_vf is not None or model.num_timesteps != 65536
                or model.n_steps != 1024 or model.batch_size != 256
                or model.n_epochs != 4 or model.get_env() is not None
                or model.get_vec_normalize_env() is not None
                or model.seed != metadata['ppo_seed']):
            raise RuntimeError('loaded PPO contract differs')
        if (type(policy) is not ActorCriticPolicy or policy.use_sde or policy.squash_output
                or policy.observation_space.shape != (99,)
                or policy.action_space.shape != (16,)):
            raise RuntimeError('loaded policy type/spaces differ')
        for space, bound in ((policy.observation_space, 5.), (policy.action_space, 1.)):
            if (space.dtype != np.float32 or not np.all(space.low == -bound)
                    or not np.all(space.high == bound)):
                raise RuntimeError('space dtype/bounds differ')

        def describe(stack):
            return [(type(layer).__name__, getattr(layer, 'in_features', None),
                     getattr(layer, 'out_features', None)) for layer in stack]

        architecture = [('Linear', 99, 128), ('Tanh', None, None),
                        ('Linear', 128, 128), ('Tanh', None, None)]
        if (describe(policy.mlp_extractor.value_net) != architecture
                or describe(policy.mlp_extractor.policy_net) != architecture
                or type(policy.value_net) is not torch.nn.Linear
                or policy.value_net.in_features != 128 or policy.value_net.out_features != 1
                or policy.action_net.in_features != 128 or policy.action_net.out_features != 16
                or type(policy.features_extractor).__name__ != 'FlattenExtractor'
                or list(policy.features_extractor.parameters())):
            raise RuntimeError('network architecture differs')
        named = dict(policy.named_parameters())
        critic = {k: v for k, v in named.items()
                  if k.startswith(('mlp_extractor.value_net.', 'value_net.'))}
        actor = {k: v for k, v in named.items() if k not in critic}
        if len(critic) != 6 or len(actor) != 7:
            raise RuntimeError('unexpected actor/critic parameter groups')
        if {id(v) for v in actor.values()} & {id(v) for v in critic.values()}:
            raise RuntimeError('actor/critic parameter sharing detected')
        if any(p.device.type != 'cpu' for p in named.values()):
            raise RuntimeError('non-CPU parameter')
        report['loaded_state'] = {'policy': initial_policy, 'optimizer': initial_optimizer}
        report['parameters'] = {
            key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                  'sha256': hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
                  'group': 'critic' if key in critic else 'actor', 'numel': value.numel(),
                  'l2_norm': float(torch.linalg.vector_norm(value.detach().double()))}
            for key, value in named.items()}
        report['buffers'] = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                                  'sha256': hashlib.sha256(
                                      value.detach().cpu().numpy().tobytes()).hexdigest()}
                             for key, value in policy.named_buffers()}
        report['shared_trainable_parameters'] = 0
        policy.action_net.register_forward_pre_hook(forbidden('actor_forward_attempted'))
        policy.mlp_extractor.policy_net.register_forward_pre_hook(
            forbidden('actor_forward_attempted'))

        def head_entry(module, args):
            bounded('head_batches_attempted', 1, 5)
            bounded('head_states_attempted', int(args[0].shape[0]), 1024)

        def head_return(module, args, result):
            counts['head_batches_returned'] += 1
            counts['head_states_returned'] += int(result.shape[0])

        policy.value_net.register_forward_pre_hook(head_entry)
        policy.value_net.register_forward_hook(head_return)

        def values(observations):
            if (observations.dtype != np.float32 or observations.shape[1:] != (99,)
                    or not np.isfinite(observations).all()
                    or np.max(np.abs(observations)) > 5):
                raise RuntimeError('invalid diagnostic observations')
            n = len(observations)
            bounded('value_batches_attempted', 1, 5)
            bounded('value_states_attempted', n, 1024)
            record('value_entry')
            value = policy.predict_values(torch.from_numpy(observations)).flatten()
            if value.shape != (n,) or not bool(torch.isfinite(value).all()):
                raise RuntimeError('invalid predicted values')
            counts['value_batches_returned'] += 1
            counts['value_states_returned'] += n
            record('value_return')
            return value

        with np.load(plan['samples'], allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        if (arrays['start_obs'].shape != (512, 99)
                or arrays['endpoint_obs'].shape != (512, 99)
                or arrays['rewards'].shape != (512, 64)
                or not np.array_equal(arrays['quartile'], np.repeat(np.arange(4), 128))):
            raise RuntimeError('sample shape/order differs')
        gamma = .99
        gamma64 = gamma**64
        rewards = arrays['rewards']
        g64 = rewards @ (gamma**np.arange(64))
        if not np.allclose(g64, arrays['discounted_rewards'], rtol=0, atol=1e-10):
            raise RuntimeError('independent discounted reward check differs')
        d64 = (1-gamma64)/(1-gamma)
        scale = max(1., float(np.median(np.abs(g64/d64)))/(1-gamma))
        with torch.no_grad():
            end_values = values(arrays['endpoint_obs']).cpu().numpy().copy()
        with (output / 'endpoint_values.npz').open('xb') as stream:
            np.savez_compressed(stream, endpoint_values=end_values,
                                endpoint_indices=arrays['endpoint_indices'])
        targets = g64 + gamma64*end_values.astype(np.float64)
        train_targets = targets.astype(np.float32)
        if not np.isfinite(train_targets).all():
            raise RuntimeError('target cannot be represented in float32')
        starts = np.empty(512, dtype=np.float32)
        batches = []
        report['completed_batches'] = batches
        for q in range(4):
            sl = slice(q*128, (q+1)*128)
            policy.zero_grad(set_to_none=True)
            predicted = values(arrays['start_obs'][sl])
            starts[sl] = predicted.detach().cpu().numpy()
            target = torch.from_numpy(train_targets[sl])
            loss = .5 * torch.mean((predicted-target)**2)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('nonfinite critic proxy loss')
            loss.backward()
            if any(p.grad is not None for p in actor.values()):
                raise RuntimeError('critic-only loss reached actor parameters')
            norms = {}
            for name, parameter in critic.items():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError('missing/nonfinite critic gradient: ' + name)
                norms[name] = float(torch.linalg.vector_norm(parameter.grad.double()))
            norm = math.sqrt(sum(x*x for x in norms.values()))
            actual_bias_grad = float(policy.value_net.bias.grad.item())
            expected_bias_grad = float(np.mean(starts[sl].astype(np.float64)
                                               - train_targets[sl].astype(np.float64)))
            if abs(actual_bias_grad-expected_bias_grad) > 1e-5+1e-5*abs(expected_bias_grad):
                raise RuntimeError('analytic output-bias gradient check failed')
            delta = targets[sl]-starts[sl]
            batch_scale = max(1., float(np.median(np.abs(g64[sl]/d64)))/(1-gamma))
            batches.append({'quartile': q, 'rows': 128,
                            'weighted_proxy_loss_float32': float(loss.detach()),
                            'critic_gradient_l2': norm, 'per_parameter_l2': norms,
                            'per_parameter_grad_over_parameter_norm': {
                                name: value/max(float(torch.linalg.vector_norm(
                                    critic[name].detach().double())), 1e-12)
                                for name, value in norms.items()},
                            'critic_only_clip_factor': min(1., .5/(norm+1e-6)),
                            'analytic_bias_gradient_expected': expected_bias_grad,
                            'actual_bias_gradient': actual_bias_grad,
                            'actor_gradients_all_none': True,
                            'actor_gradient_counts': {'none': len(actor), 'zero': 0, 'nonzero': 0},
                            'reward_scale_S': batch_scale,
                            'C_bias': float(np.median(delta)/((1-gamma64)*batch_scale)),
                            'C_rmse': float(np.sqrt(np.mean(delta**2))/((1-gamma64)*batch_scale)),
                            'stats': diagnostic_stats(starts[sl], targets[sl])})
            write_json(output / f'batch_{q}.json', batches[-1])
            with (output / f'batch_{q}_gradients.npz').open('xb') as stream:
                np.savez_compressed(stream, **{
                    name: parameter.grad.detach().cpu().numpy()
                    for name, parameter in critic.items()})
            with (output / f'batch_{q}_values.npz').open('xb') as stream:
                np.savez_compressed(stream, start_indices=arrays['start_indices'][sl],
                                    start_values=starts[sl], endpoint_values=end_values[sl],
                                    targets_float64=targets[sl], targets_used_float32=train_targets[sl])
            record('batch_completed')
        delta = targets-starts
        normalized_bias = float(np.median(delta)/((1-gamma64)*scale))
        normalized_rmse = float(np.sqrt(np.mean(delta**2))/((1-gamma64)*scale))
        positive_fraction = float(np.mean(delta > 0))
        negative_fraction = float(np.mean(delta < 0))
        under = (normalized_bias >= .10 and positive_fraction >= .75
                 and sum(b['C_bias'] >= .10 for b in batches) >= 3
                 and batches[-1]['C_bias'] >= .10)
        over = (normalized_bias <= -.10 and negative_fraction >= .75
                and sum(b['C_bias'] <= -.10 for b in batches) >= 3
                and batches[-1]['C_bias'] <= -.10)
        pressure = (sum(b['critic_gradient_l2'] >= 5 for b in batches) >= 3
                    and batches[-1]['critic_gradient_l2'] >= 5)
        policy.zero_grad(set_to_none=True)
        final_policy = hash_state('policy', policy.state_dict())
        final_optimizer = hash_state('optimizer', policy.optimizer.state_dict())
        if initial_policy != final_policy or initial_optimizer != final_optimizer:
            raise RuntimeError('diagnostic changed model or optimizer state')
        if any(name.split('.')[0] in NoPhysics.blocked for name in sys.modules):
            raise RuntimeError('forbidden module appeared')
        maps = Path('/proc/self/maps').read_text()
        if 'libmujoco' in maps or 'libepa01_engine' in maps:
            raise RuntimeError('physics library mapped')
        if not (counts['ppo_load_attempted'] == counts['ppo_load_returned'] == 1
                and counts['torch_load_attempted'] == counts['torch_load_returned']
                and counts['value_states_attempted'] == counts['value_states_returned'] == 1024
                and counts['head_states_attempted'] == counts['head_states_returned'] == 1024
                and counts['value_batches_attempted'] == counts['value_batches_returned'] == 5
                and counts['head_batches_attempted'] == counts['head_batches_returned'] == 5
                and counts['backward_attempted'] == counts['backward_returned'] == 4):
            raise RuntimeError('call closure differs')
        if any(counts[key] for key in (
                'actor_forward_attempted', 'optimizer_step_attempted', 'learn_attempted',
                'train_attempted', 'save_attempted', 'autograd_grad_attempted', 'clip_attempted')):
            raise RuntimeError('forbidden operation attempted')
        with (output / 'values.npz').open('xb') as stream:
            np.savez_compressed(stream, start_indices=arrays['start_indices'],
                                quartile=arrays['quartile'], start_values=starts,
                                endpoint_values=end_values, discounted_rewards=g64,
                                targets_float64=targets, targets_used_float32=train_targets,
                                residual=delta)
        masks = {'all': np.ones(512, dtype=bool)}
        for q in range(4):
            masks['quartile:' + str(q)] = arrays['quartile'] == q
        for terrain in sorted(set(arrays['terrain_label'].tolist())):
            masks['terrain:' + terrain] = arrays['terrain_label'] == terrain
        fraction = arrays['window_effective_action_nonzero_fraction']
        masks.update({'effective:zero': fraction == 0, 'effective:partial': (fraction > 0) & (fraction < 1),
                      'effective:all': fraction == 1})
        speed = arrays['window_mean_abs_servo_vx']
        masks.update({'speed:below_0p05': speed < .05,
                      'speed:0p05_to_0p8': (speed >= .05) & (speed < .8),
                      'speed:above_0p8': speed >= .8,
                      'yaw:present': arrays['window_has_yaw'],
                      'yaw:absent': ~arrays['window_has_yaw']})
        groups = {}

        def stats(vector):
            return {'mean': float(np.mean(vector)), 'median': float(np.median(vector)),
                    'std': float(np.std(vector)), 'min': float(np.min(vector)),
                    'max': float(np.max(vector)), 'p05': float(np.quantile(vector, .05)),
                    'p95': float(np.quantile(vector, .95))}

        for label, mask in masks.items():
            if not np.any(mask):
                groups[label] = {'n': 0, 'enough_for_group_decision': False}
                continue
            group_scale = max(1., float(np.median(np.abs(g64[mask]/d64)))/(1-gamma))
            groups[label] = {'n': int(mask.sum()),
                             'episodes': len(set(arrays['episode_index'][mask].tolist())),
                             'enough_for_group_decision': bool(mask.sum() >= 32),
                             'reward_scale_S': group_scale,
                             'C_bias': float(np.median(delta[mask])/((1-gamma64)*group_scale)),
                             'C_rmse': float(np.sqrt(np.mean(delta[mask]**2))/((1-gamma64)*group_scale)),
                             'metrics': diagnostic_stats(starts[mask], targets[mask]),
                             'mae': float(np.mean(np.abs(delta[mask]))),
                             'positive_residual_fraction': float(np.mean(delta[mask] > 0)),
                             'distributions': {key: stats(value[mask]) for key, value in {
                                 'discounted_rewards': g64, 'start_values': starts,
                                 'endpoint_values': end_values, 'targets': targets,
                                 'residual': delta}.items()}}
        concentration = {}
        for family in ('terrain:', 'effective:', 'speed:', 'yaw:'):
            eligible = {k: v for k, v in groups.items()
                        if k.startswith(family) and v['enough_for_group_decision']}
            concentration[family] = bool(len(eligible) >= 2 and any(
                group['C_rmse'] >= .2 and all(
                    other['C_rmse'] < .1 for name, other in eligible.items() if name != label)
                for label, group in eligible.items()))
        report.update(status='passed', batches=batches,
                      stats=diagnostic_stats(starts, targets), groups=groups,
                      normalization={'gamma': gamma, 'horizon': 64, 'D64': d64, 'S': scale,
                                     'C_bias': normalized_bias, 'C_rmse': normalized_rmse,
                                     'positive_residual_fraction': positive_fraction,
                                     'negative_residual_fraction': negative_fraction},
                      gates={'broad_underestimate': bool(under),
                             'broad_overestimate': bool(over),
                             'state_or_distribution_mismatch': bool(normalized_rmse >= .20
                                                                    and not under and not over),
                             'critic_only_clip_pressure': bool(pressure),
                             'all_batches_critic_norm_below_cap': bool(all(
                                 b['critic_gradient_l2'] <= .5 for b in batches)),
                             'early_time_concentration': bool(
                                 abs(batches[3]['C_bias']) < .1 and batches[3]['C_rmse'] < .2
                                 and any(b['C_rmse'] >= .2 for b in batches[:2])),
                             'state_group_concentration': concentration},
                      unchanged_state={'policy': final_policy, 'optimizer': final_optimizer},
                      arrays=identity(output / 'values.npz'),
                      physics_modules_and_libraries_absent=True,
                      actor_forward_zero=True, final_gradients_cleared=True,
                      model_state_matches_recorded_final=True,
                      imports={name: sys.modules[name].__file__
                               for name in ('numpy', 'torch', 'stable_baselines3',
                                            'rl16_learning_11', 'value14_math')},
                      boundary='64-step saved behavior with current final bootstrap; not historical GAE or on-policy return')
    except BaseException as error:
        primary_error = error
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        signal.alarm(0)
        cleanup_errors = []
        try:
            report['compute_library_maps_after'] = mapped_compute_libraries()
        except BaseException as error:
            cleanup_errors.append('compute mapping closure: ' + repr(error))
        if model is not None:
            try:
                model.policy.zero_grad(set_to_none=True)
                report['final_gradients_cleared'] = all(
                    p.grad is None for p in model.policy.parameters())
                current = {'policy': hash_state('policy', model.policy.state_dict()),
                           'optimizer': hash_state('optimizer', model.policy.optimizer.state_dict())}
                report['cleanup_state'] = current
                if report.get('loaded_state') != current:
                    cleanup_errors.append('loaded and cleanup states differ or initial hash unavailable')
            except BaseException as error:
                cleanup_errors.append(type(error).__name__ + ': ' + str(error))
        try:
            changed = [name for name, expected in plan['inputs'].items()
                       if identity(name) != expected]
            report['source_postcheck_passed'] = not changed
            if changed:
                cleanup_errors.append('input postcheck differs: ' + repr(changed))
            if identity(plan_path) != report['plan_identity']:
                cleanup_errors.append('plan identity changed during execution')
        except BaseException as error:
            cleanup_errors.append(type(error).__name__ + ': ' + str(error))
        report['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            report['status'] = 'failed'
        report['elapsed_s'] = time.monotonic()-started
        try:
            record('worker_exit:' + report['status'])
        except BaseException as error:
            cleanup_errors.append('journal finalization: ' + repr(error))
            report['status'] = 'failed'
        try:
            journal.close()
        except BaseException as error:
            cleanup_errors.append('journal close: ' + repr(error))
            report['status'] = 'failed'
        try:
            write_json(output / 'receipt.json', report)
        except BaseException as error:
            cleanup_errors.append('receipt write: ' + repr(error))
            report['status'] = 'failed'
            print(json.dumps({'primary_error': report.get('error'),
                              'cleanup_errors': cleanup_errors, 'counts': counts}), file=sys.stderr)
    if primary_error is not None:
        raise primary_error
    if report['status'] != 'passed':
        raise RuntimeError('diagnostic finalization failed')
    print(json.dumps({'status': report['status'], 'counts': counts, 'gates': report['gates']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.plan.resolve(), arguments.output.resolve())
