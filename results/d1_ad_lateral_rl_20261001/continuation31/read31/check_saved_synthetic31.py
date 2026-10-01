"""Read only the already executed C31 synthetic updates/checkpoint; no Torch."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

FORBIDDEN = {'torch', 'mujoco', 'stable_baselines3', 'gymnasium', 'gym', 'wheel_legged_control'}
if any(name.partition('.')[0] in FORBIDDEN for name in sys.modules):
    raise RuntimeError('synthetic saved readback must start cold')


class NoExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in FORBIDDEN:
            raise RuntimeError('saved synthetic readback prohibits '+fullname)


sys.meta_path.insert(0, NoExecution())
sys.dont_write_bytecode = True
import numpy as np
from checkpoint_read31 import checkpoint_parameters31
from learning_read31 import same, verify_update31, zero_state31
from math_read31 import need, reward31, advantages31, linear_outputs


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def run(folder):
    inputs = {}
    def document(path):
        inputs[str(path)] = identity(path)
        return json.loads(Path(path).read_text())
    receipt = document(folder/'receipt.json')
    need(receipt['passed'] is True and receipt['exit_code'] == 0 and receipt['source_mismatches'] == []
         and receipt['robot_controls'] == receipt['robot_models'] == receipt['normal_native'] == receipt['compiler_native'] == 0,
         'saved synthetic execution was not successful and robot-free')
    for path, expected in receipt['source_identities'].items():
        inputs[path] = identity(path)
        need(inputs[path] == expected, 'already executed synthetic source changed')
    updates = []
    for index in (0, 1):
        saved = document(folder/f'update_{index:02d}.json')
        need(saved['synthetic'] is True, 'saved fixture identity differs')
        batch, update = saved['batch'], saved['update']
        rows = batch['rows']
        need(len(rows) == 8 and batch['cycle_indices'] == list(range(8)), 'actual saved eight-cycle fixture differs')
        rewards = []
        for cycle, row in enumerate(rows):
            need(row['cycle_index'] == cycle and row['macro_index'] == 0
                 and row['control_start'] == 200 and row['control_end'] == 1000
                 and row['elapsed_controls'] == 800 and row['terminal'] is True,
                 'saved synthetic macro boundaries changed')
            outcome = 'controlled_failure' if cycle == 1 else 'success'
            calculated = reward31(row['reward_components'], outcome, 10., True)
            same(row['reward'], calculated['reward'], 'saved reward is not canonical-component sum')
            same(row['terminal_adjustment'], calculated['terminal_adjustment'], 'failure charge omits unelapsed horizon')
            rewards.append(calculated['reward'])
        gae = advantages31(rewards, [r['old_value'] for r in rows], [800]*8, list(range(8)), [True]*8)
        for i, row in enumerate(rows):
            for key in ('delta', 'gae', 'value_target', 'normalized_advantage'):
                same(row[key], gae[key][i], 'saved GAE/normalization differs')
        parameters, adam = zero_state31()
        if index == 1:
            parameters['actor.bias'][:] = .5
        verified = verify_update31(batch, update, parameters, adam, batch_index=0, permutation_seed=310031)
        need(verified['counts']['actual_optimizer_steps'] == (2 if index == 0 else 0), 'fixture actual steps differ')
        updates.append(dict(index=index, passed=verified['passed'], valid_train_call=verified['valid_train_call'],
                            counts=verified['counts'], normal_kl_stop=update['normal_kl_stop']))
        if index == 0:
            tampered = copy.deepcopy(update)
            tampered['attempts'][0]['gradients']['actor']['actual_clipped'][1][0] += .01
            try:
                verify_update31(batch, tampered, parameters, adam, batch_index=0, permutation_seed=310031)
            except ValueError:
                tamper_rejected = True
            else:
                raise ValueError('changed actual gradient escaped independent verification')
    final = folder/'pytest_tmp/test_sampling_train_vs_eval_an0/final'
    manifest = document(final/'manifest.json')
    for name, expected in manifest['files'].items():
        inputs[str(final/name)] = identity(final/name)
        need(inputs[str(final/name)] == expected, 'saved synthetic checkpoint/probe payload changed')
    parameters = checkpoint_parameters31(final/'lateral_final.pt')
    need(all(np.array_equal(value, np.zeros_like(value)) for value in parameters.values()),
         'this separate saved checkpoint fixture must contain all four initial zero tensors')
    before, after = document(final/'probe_before.json'), document(final/'probe_after.json')
    need(before == after, 'saved reload probe differs')
    mean, value = linear_outputs(parameters, before['observation54'])
    same(before['actor_mean_z3'], mean, 'decoded checkpoint actor differs from probe')
    same(before['value'], value, 'decoded checkpoint value differs from probe')
    for name in ('check_saved_synthetic31.py', 'math_read31.py', 'learning_read31.py', 'checkpoint_read31.py'):
        path = Path(__file__).parent/name
        inputs[str(path)] = identity(path)
    return dict(schema='d1-c31-saved-synthetic-independent-v1', passed=True, updates=updates,
                changed_actual_gradient_rejected=tamper_rejected, checkpoint_all_four_tensors_verified=True,
                canonical_reward_and_GAE_verified=True,
                checkpoint_fixture_is_separate_untrained_policy=True,
                prior_synthetic_torch_counts=receipt['counts'], new_torch_or_model_calls=0,
                new_robot_controls=0, new_normal_native=0, new_compiler_native=0, inputs=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'exclusive readback output already exists')
    report = run(args.folder.resolve(strict=True))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
