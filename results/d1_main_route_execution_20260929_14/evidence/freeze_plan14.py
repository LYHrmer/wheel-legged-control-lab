"""Bind the reviewed source and existing data, without importing any model code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

C = Path(__file__).resolve().parent
W = C.parent
R = Path('/home/lyh/wheel-legged-control-lab')
RUN = W / 'rl11/training_run_01'


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            size += len(block)
            digest.update(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def main():
    closed_path = W / 'rl11/partial_closed_manifest_20260929_05.json'
    closed = json.loads(closed_path.read_text())['files']
    selection = json.loads((C / 'sample_01/selection.json').read_text())
    assert selection['closed_manifest_identity'] == identity(closed_path)
    inputs = {}
    for relative, expected in selection['verified_input_files'].items():
        assert expected == closed[relative]
        inputs[str(RUN / relative)] = expected
    for relative in ('final_checkpoint/final_model.zip',
                     'final_checkpoint/final_metadata.json',
                     'final_checkpoint_manifest.json', 'session.json'):
        actual = identity(RUN / relative)
        assert actual == closed[relative], relative
        inputs[str(RUN / relative)] = actual
    model = RUN / 'final_checkpoint/final_model.zip'
    assert inputs[str(model)]['sha256'] == '6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70'
    metadata = RUN / 'final_checkpoint/final_metadata.json'
    trusted = json.loads((RUN / 'final_checkpoint_manifest.json').read_text())
    assert trusted['metadata_sha256'] == inputs[str(metadata)]['sha256']
    assert trusted['files']['final_model.zip'] == inputs[str(model)]
    session = json.loads((RUN / 'session.json').read_text())
    inputs.update(session['dependency_hashes'])
    for name in ('rl11/rl16_learning_11.py', 'rl11/budget_spec_11.py',
                 'course_impl08/residual16_math_08.py'):
        path = W / name
        actual = identity(path)
        assert actual == session['source_hashes'][str(path)]
        inputs[str(path)] = actual
    frozen_path = W.parent / 'rl_improvement_20260914/frozen_source_before.json'
    frozen = json.loads(frozen_path.read_text())['sha256']
    assert len(frozen) == 77
    for relative, expected in frozen.items():
        path = R / relative
        actual = identity(path)
        assert actual['sha256'] == expected
        inputs[str(path)] = actual
    # Cloudpickle is used by the trusted SB3 zip loader; record current source
    # explicitly when it is outside the historical dependency inventory.
    for root in (R / '.local-deps', Path('/home/lyh/.local/lib/python3.10/site-packages'),
                 Path('/usr/lib/python3/dist-packages')):
        for path in (root / 'cloudpickle').glob('*.py'):
            inputs.setdefault(str(path), identity(path))
    assert selection['samples_identity'] == identity(C / 'sample_01/samples.npz')
    for source, key in (('prepare_value14_data.py', 'preparation_source_identity'),
                        ('value14_math.py', 'math_source_identity'),
                        ('test_value14_pure.py', 'pure_test_source_identity')):
        assert selection[key] == identity(C / source)
    worker_sha = identity(C / 'run_value14.py')['sha256']
    assert worker_sha in (C / 'go_review_14.md').read_text()
    for path in (closed_path, frozen_path, *[C / name for name in (
            'sample_01/selection.json', 'sample_01/samples.npz', 'astra_plan_14.md',
            'run_value14.py', 'prepare_value14_data.py', 'value14_math.py',
            'test_value14_pure.py', 'freeze_plan14.py', 'go_review_14.md',
            'pure_tests_receipt_01.json', 'frozen77_before_14.json')]):
        inputs[str(path)] = identity(path)
    plan = {'schema': 'd1-value14-plan-v1', 'status': 'GO',
            'base_commit': '129f84bc6532f2622530c7ad0ed4435342d7edd4',
            'model': str(model), 'metadata': str(metadata),
            'samples': str(C / 'sample_01/samples.npz'),
            'output': str(C / 'model_01'), 'retry_permitted': False,
            'limits': {'ppo_loads': 1, 'torch_loads': 3, 'value_batches': 5,
                       'value_states': 1024, 'backwards': 4, 'actor_forwards': 0,
                       'optimizer_steps': 0, 'learn': 0, 'train': 0, 'save': 0,
                       'physics': 0, 'worker_wall_s': 120,
                       'host_wall_s': 150, 'host_kill_grace_s': 5},
            'inputs': dict(sorted(inputs.items()))}
    with (C / 'plan_go_14.json').open('x') as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'inputs': len(inputs), 'input_bytes': sum(x['bytes'] for x in inputs.values()),
                      'plan_identity': identity(C / 'plan_go_14.json'), 'model_calls': 0}))


if __name__ == '__main__':
    main()
