"""Bounded C31 event policy and SMDP PPO; no robot construction on import.

The owner supplies only completed, safe, archived cycles. Every attempted
minibatch and actual Adam step is returned for independent saved-data review.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch


SCHEMA31 = 'd1-c31-event-smdp-ppo-v1'
STD31 = .35
LOG_NORMALIZER31 = 3 * math.log(STD31 * math.sqrt(2 * math.pi))


class LinearEventPolicy31(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(54, 3, dtype=torch.float64)
        self.value = torch.nn.Linear(54, 1, dtype=torch.float64)
        with torch.no_grad():
            for layer in (self.actor, self.value):
                layer.weight.zero_()
                layer.bias.zero_()

    def optimizer31(self):
        return torch.optim.Adam([
            {'params': list(self.actor.parameters()), 'lr': .001},
            {'params': list(self.value.parameters()), 'lr': .001},
        ], lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0,
            amsgrad=False, foreach=False, fused=False)


def _matrix54(rows) -> torch.Tensor:
    array = np.asarray(rows, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != 54 or len(array) < 1
            or not np.isfinite(array).all() or np.max(np.abs(array)) > 5.):
        raise ValueError('C31 normalized observation rows must be finite 54-D')
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float64)


def _log_prob(z: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return -.5 * (((z-mean)/STD31)**2).sum(dim=-1) - LOG_NORMALIZER31


def _mean_kl(old_mean: torch.Tensor, new_mean: torch.Tensor) -> torch.Tensor:
    return (((old_mean-new_mean)**2).sum(dim=-1)/(2*STD31**2)).mean()


class LatchSampler31:
    """One actual actor row per true latch; train samples, eval uses mean."""
    def __init__(self, policy: LinearEventPolicy31, *, mode: str, seed=310031):
        if mode not in ('train', 'learned') or not isinstance(policy, LinearEventPolicy31):
            raise ValueError('C31 lateral sampler mode/policy differs')
        self.policy = policy
        self.mode = mode
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.actor_rows = 0
        self.value_rows = 0
        self.rng_samples = 0

    def __call__(self, observation, teacher_plan, cycle_index, local_control_index):
        del teacher_plan
        x = _matrix54([observation['normalized54']])
        with torch.no_grad():
            mean = self.policy.actor(x)[0]
            self.actor_rows += 1
            if self.mode == 'train':
                z = mean + STD31 * torch.randn(3, generator=self.generator,
                                               dtype=torch.float64)
                self.rng_samples += 1
                value = float(self.policy.value(x)[0, 0])
                self.value_rows += 1
            else:
                z = mean.clone()
                value = None
            logp = float(_log_prob(z[None, :], mean[None, :])[0])
        if not all(math.isfinite(float(v)) for v in (mean.tolist()+z.tolist()+[logp])):
            raise RuntimeError('C31 lateral actor produced nonfinite latch')
        return {
            'latent_z3': z.tolist(), 'old_mean_z3': mean.tolist(),
            'latent_std_z3': [STD31]*3, 'old_log_prob': logp,
            'old_value': value,
            'policy_receipt': {'schema': SCHEMA31, 'mode': self.mode,
                'cycle_index': int(cycle_index),
                'local_control_index': int(local_control_index),
                'actor_rows_cumulative': self.actor_rows,
                'value_rows_cumulative': self.value_rows,
                'rng_samples_cumulative': self.rng_samples},
        }


def reward31(components: dict, *, terminal_adjustment=0.) -> float:
    # Use the independently reconstructed C30 names, without an unreviewed
    # worker-side alias layer between physical components and learning.
    keys = ('progress_potential_difference', 'elapsed_s',
            'backtrack_normalized',
            'longitudinal_normalized_square_integral_s',
            'lateral_goal_error_normalized_square_integral_s',
            'yaw_error_normalized_square_integral_s',
            'torque_normalized_square_integral_s')
    if not all(key in components and math.isfinite(float(components[key])) for key in keys):
        raise ValueError('C31 reward needs all actual C30 macro components')
    if not math.isfinite(float(terminal_adjustment)):
        raise ValueError('C31 terminal reward is nonfinite')
    return float(10*components['progress_potential_difference'] - components['elapsed_s']
        - .05*components['backtrack_normalized']
        - .02*components['longitudinal_normalized_square_integral_s']
        - .02*components['lateral_goal_error_normalized_square_integral_s']
        - .02*components['yaw_error_normalized_square_integral_s']
        - .5*components['torque_normalized_square_integral_s']
        + terminal_adjustment)


def prepare_batch31(closed_cycles: Sequence[dict]) -> dict:
    """Eight *closed* cycles, 1..4 real latches each; never bootstrap a reset."""
    if len(closed_cycles) != 8:
        raise ValueError('C31 batch requires exactly eight closed cycles')
    first_index = closed_cycles[0]['cycle_index']
    if (type(first_index) is not int or first_index not in range(0,64,8)
            or [c['cycle_index'] for c in closed_cycles]
               != list(range(first_index,first_index+8))):
        raise ValueError('C31 batch must contain eight consecutive reserved cycles')
    rows = []
    seen_cycles = set()
    for cycle in closed_cycles:
        index = cycle['cycle_index']
        events = cycle['macros']
        status = cycle['terminal_kind']
        seconds = float(cycle['actual_seconds'])
        if (type(index) is not int or index in seen_cycles or
                status not in ('success', 'controlled_failure') or
                not (1 <= len(events) <= 4) or not math.isfinite(seconds)
                or seconds <= 0 or seconds > 22 or
                abs(seconds*100-round(seconds*100)) > 1e-9 or
                events[0]['control_range'][0] != 200 or
                events[-1]['control_range'][1] != round(seconds*100) or
                cycle.get('archive_valid') is not True or
                cycle.get('native_valid') is not True or
                cycle.get('safe_handoff') is not True or
                cycle.get('retention_complete') is not True):
            raise ValueError('C31 incomplete/unsafe cycle cannot enter PPO')
        seen_cycles.add(index)
        adjustment = (5. if status == 'success' else
                      -50. - max(0., 22. - seconds))
        cycle_rows = []
        previous_end = None
        for j, event in enumerate(events):
            latch = event['skill31_latch']
            obs = latch['observation31']
            duration = float(event['elapsed_controls']) * .01
            control_range = event['control_range']
            components = event.get('reward_components', event)
            if (latch['mode'] != 'train' or obs['macro_index'] != j
                    or latch['cycle_index'] != index
                    or duration <= 0 or not math.isfinite(duration)
                    or event.get('truncated') is True
                    or len(control_range) != 2
                    or control_range[1]-control_range[0] != event['elapsed_controls']
                    or latch['local_control_index'] != control_range[0]
                    or previous_end is not None and control_range[0] != previous_end
                    or abs(float(components['elapsed_s'])-duration) > 1e-12):
                raise ValueError('C31 training event is not a complete true latch')
            previous_end = control_range[1]
            reward = reward31(components,
                terminal_adjustment=adjustment if j == len(events)-1 else 0.)
            row = {
                'cycle_index': index, 'macro_index': j,
                'control_start': int(control_range[0]),
                'control_end': int(control_range[1]),
                'elapsed_controls': int(event['elapsed_controls']),
                'duration_s': duration,
                'observation54': obs['normalized54'],
                'latent_z3': latch['latent_z3'],
                'old_mean_z3': latch['old_mean_z3'],
                'old_log_prob': float(latch['old_log_prob']),
                'old_value': float(latch['old_value']),
                'reward_components': components,
                'terminal_adjustment': adjustment if j == len(events)-1 else 0.,
                'reward': reward,
                'terminal': j == len(events)-1,
            }
            if (row['control_end'] <= row['control_start'] or
                    not math.isfinite(row['old_log_prob']) or
                    not math.isfinite(row['old_value'])):
                raise ValueError('C31 macro duration or old policy value is invalid')
            cycle_rows.append(row)
        for j in range(len(cycle_rows)-1, -1, -1):
            row = cycle_rows[j]
            next_v = 0. if row['terminal'] else cycle_rows[j+1]['old_value']
            next_gae = 0. if row['terminal'] else cycle_rows[j+1]['gae']
            delta = row['reward'] + next_v - row['old_value']
            row['delta'] = delta
            row['gae_factor'] = .95**row['duration_s']
            row['gae'] = delta + row['gae_factor']*next_gae
            row['value_target'] = row['gae'] + row['old_value']
        rows.extend(cycle_rows)
    if not 1 <= len(rows) <= 32:
        raise ValueError('C31 actual macro batch exceeds 32')
    advantages = np.asarray([r['gae'] for r in rows], dtype=np.float64)
    center = advantages - float(np.mean(advantages))
    divisor = float(np.sqrt(np.mean(center**2)) + 1e-8)
    for row, advantage in zip(rows, center/divisor):
        row['normalized_advantage'] = float(advantage)
    _matrix54([r['observation54'] for r in rows])
    return {'schema': SCHEMA31, 'cycle_indices': [c['cycle_index'] for c in closed_cycles],
            'macro_count': len(rows), 'advantage_mean': float(np.mean(advantages)),
            'advantage_divisor': divisor, 'rows': rows}


def _parameter_record(policy, optimizer):
    result = {}
    for group_name, layer in (('actor', policy.actor), ('value', policy.value)):
        result[group_name] = {}
        for name, parameter in layer.named_parameters():
            state = optimizer.state.get(parameter, {})
            step = state.get('step', 0)
            result[group_name][name] = {
                'parameter': parameter.detach().cpu().tolist(),
                'adam_step': int(step.item() if torch.is_tensor(step) else step),
                'exp_avg': (torch.zeros_like(parameter) if 'exp_avg' not in state
                            else state['exp_avg']).detach().cpu().tolist(),
                'exp_avg_sq': (torch.zeros_like(parameter) if 'exp_avg_sq' not in state
                               else state['exp_avg_sq']).detach().cpu().tolist(),
            }
    return result


def _gradient_record(parameters, raw_norm):
    gradients = [p.grad.detach().clone() for p in parameters]
    factor = min(1., .5/(float(raw_norm)+1e-6))
    return {'raw_norm': float(raw_norm), 'factor': factor,
            'raw': [g.cpu().tolist() for g in gradients],
            'clipped': [(g*factor).cpu().tolist() for g in gradients],
            'clipped_norm': float(torch.linalg.vector_norm(
                torch.cat([g.reshape(-1) for g in gradients]))*factor)}


def update_batch31(policy: LinearEventPolicy31, optimizer, batch: dict,
                   *, batch_index: int, step_count_before: int,
                   permutation_seed: int) -> dict:
    """Two epochs, minibatch 8; KL rejection is a forward-only attempt."""
    if (not isinstance(policy, LinearEventPolicy31) or
            type(batch_index) is not int or not 0 <= batch_index < 8 or
            type(step_count_before) is not int or not 0 <= step_count_before <= 64 or
            batch.get('schema') != SCHEMA31 or
            not 1 <= batch.get('macro_count', 0) <= 32 or
            batch.get('macro_count') != len(batch.get('rows', ())) or
            batch.get('cycle_indices') != list(range(8*batch_index,8*batch_index+8))):
        raise ValueError('C31 update identity/budget differs')
    rows = batch['rows']
    n = len(rows)
    x = _matrix54([r['observation54'] for r in rows])
    z = torch.as_tensor(np.asarray([r['latent_z3'] for r in rows]), dtype=torch.float64)
    old_mean = torch.as_tensor(np.asarray([r['old_mean_z3'] for r in rows]), dtype=torch.float64)
    old_logp = torch.as_tensor([r['old_log_prob'] for r in rows], dtype=torch.float64)
    adv = torch.as_tensor([r['normalized_advantage'] for r in rows], dtype=torch.float64)
    target = torch.as_tensor([r['value_target'] for r in rows], dtype=torch.float64)
    if (z.shape != (n,3) or old_mean.shape != (n,3) or
            not all(torch.isfinite(v).all() for v in (z,old_mean,old_logp,adv,target))):
        raise ValueError('C31 minibatch tensors are invalid')
    if not torch.allclose(_log_prob(z,old_mean),old_logp,
                          rtol=0.,atol=1e-12):
        raise ValueError('C31 saved old latent likelihood differs before actor forward')
    gen = torch.Generator(device='cpu').manual_seed(int(permutation_seed))
    attempts = []
    post_kl_rows = 0
    actual_steps = 0
    normal_stop = False
    hard_stop = None
    for epoch in range(2):
        permutation = torch.randperm(n, generator=gen).tolist()
        for minibatch, offset in enumerate(range(0,n,8)):
            if step_count_before+actual_steps >= 64:
                raise RuntimeError('C31 actual optimizer cap reached before batch completion')
            indices = permutation[offset:offset+8]
            ident = {'batch_index': batch_index, 'epoch': epoch,
                     'minibatch': minibatch, 'indices': indices}
            means = policy.actor(x[indices])  # one actual pre-update actor forward
            pre_kl = float(_mean_kl(old_mean[indices],means).detach())
            attempt = {**ident, 'actor_forward_rows': len(indices),
                       'value_forward_rows': 0, 'pre_kl': pre_kl,
                       'optimizer_step': False, 'backward': False,
                       'parameters_before': _parameter_record(policy,optimizer)}
            if not math.isfinite(pre_kl):
                hard_stop = 'nonfinite_pre_update_kl'
                attempts.append(attempt)
                break
            if pre_kl > .03:
                normal_stop = True
                attempt['normal_kl_stop'] = True
                attempts.append(attempt)
                break
            optimizer.zero_grad(set_to_none=True)
            values = policy.value(x[indices]).squeeze(-1)
            attempt['value_forward_rows'] = len(indices)
            logp = _log_prob(z[indices],means)
            ratio = torch.exp(logp-old_logp[indices])
            surrogate = torch.minimum(ratio*adv[indices],
                torch.clamp(ratio,.8,1.2)*adv[indices])
            policy_loss = -surrogate.mean()
            value_loss = .5*((values-target[indices])**2).mean()
            loss = policy_loss+value_loss
            attempt['losses'] = {'policy':float(policy_loss.detach()),
                                 'value':float(value_loss.detach()),
                                 'total':float(loss.detach())}
            if not torch.isfinite(loss):
                hard_stop = 'nonfinite_loss'
                attempts.append(attempt)
                break
            loss.backward()
            attempt['backward'] = True
            groups = {}
            for name, layer in (('actor',policy.actor),('value',policy.value)):
                params = list(layer.parameters())
                if any(p.grad is None for p in params):
                    raise RuntimeError('C31 actor/value gradient missing')
                raw_norm = torch.linalg.vector_norm(torch.cat(
                    [p.grad.detach().reshape(-1) for p in params]))
                if not torch.isfinite(raw_norm):
                    hard_stop = 'nonfinite_gradient'
                    break
                groups[name] = _gradient_record(params,raw_norm)
                torch.nn.utils.clip_grad_norm_(params,.5,norm_type=2.,
                    error_if_nonfinite=True,foreach=False)
                groups[name]['actual_clipped'] = [p.grad.detach().cpu().tolist()
                                                   for p in params]
            attempt['gradients'] = groups
            if hard_stop is not None:
                attempts.append(attempt)
                break
            optimizer.step()  # one call updates independent actor/value groups
            actual_steps += 1
            attempt['optimizer_step'] = True
            attempt['parameters_after'] = _parameter_record(policy,optimizer)
            if any(not torch.isfinite(p).all() for p in policy.parameters()):
                hard_stop = 'nonfinite_parameter'
                attempts.append(attempt)
                break
            with torch.no_grad():
                post_kl = float(_mean_kl(old_mean,policy.actor(x)))
            post_kl_rows += n
            attempt['post_kl_all_rows'] = post_kl
            attempt['post_kl_actor_forward_rows'] = n
            attempts.append(attempt)
            if not math.isfinite(post_kl) or post_kl > .10:
                hard_stop = 'nonfinite_or_excessive_post_update_kl'
                break
        if hard_stop is not None or normal_stop:
            break
    return {'schema':SCHEMA31,'batch_index':batch_index,'macro_count':n,
            'permutation_seed':int(permutation_seed),'attempts':attempts,
            'actual_evaluated_minibatches':len(attempts),
            'actual_backward':sum(bool(a['backward']) for a in attempts),
            'actual_optimizer_steps':actual_steps,
            'actual_pre_actor_rows':sum(a['actor_forward_rows'] for a in attempts),
            'actual_pre_value_rows':sum(a['value_forward_rows'] for a in attempts),
            'actual_post_kl_actor_rows':post_kl_rows,
            'normal_kl_stop':normal_stop,'hard_stop':hard_stop,
            'valid_train_call':actual_steps>0 and hard_stop is None}
