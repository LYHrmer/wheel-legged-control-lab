"""Build an unsigned C30 inventory; only the reviewed host may authorize a run."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from worker30 import identity, write, CONTRACT

P = Path(__file__).resolve().parents[1]
W = P.parent
R = Path('/home/lyh/wheel-legged-control-lab')


def make(stage):
    parent = W/'continuation29/source_go_29.json'
    old = json.loads(parent.read_text())
    candidate = copy.deepcopy(old)
    inputs = dict(old['inputs'])
    inputs[str(parent)] = identity(parent)
    # Inherit signed expected identities. The launcher checks every byte before
    # and after execution; inventory creation does not claim another full check.
    new_files = []
    for path in P.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts:
            continue
        relative = path.relative_to(P)
        if relative.parts[0].startswith(('source_go', 'fixed_', 'zero_', 'cancel_')):
            continue
        if relative.parts[0] == 'read30' and stage == 'bootstrap':
            continue
        if relative.parts[0] == 'bootstrap_01' and stage == 'bootstrap':
            continue
        if path.suffix in ('.py', '.md', '.json', '.log'):
            new_files.append(path)
    for name in ('final_profile_review_29.json', 'failed_saved_diagnostic_01.json',
                 'formal_reader_failure_01.json'):
        new_files.append(W/'continuation29'/name)
    new_files.extend((W/'continuation29/profile_left_01').rglob('*'))
    for path in new_files:
        if path.is_file():
            inputs[str(path.resolve())] = identity(path)
    env = dict(old['runtime_environment'])
    paths = env['PYTHONPATH'].split(':')
    env['PYTHONPATH'] = ':'.join([paths[0], str(P/'root30'), str(P/'sol30'), *paths[1:]])
    env['DISPLAY'] = None
    env['XAUTHORITY'] = None
    shared = copy.deepcopy(old['shared_session'])
    shared.update(execution_contract_id=CONTRACT, numerical_protocol_contract_id=CONTRACT,
                  outer_execution_contract_id=CONTRACT, side_arm='teacher',
                  parent_failed_attempt=str(W/'continuation29/profile_left_01'),
                  spawn_position_m=[-8., -4.7, .455])
    previous_arm = old['arms']['profile_left']
    limits = dict(previous_arm['model_limits'])
    limits['predict'] = 2201
    arm = dict(schema='d1-c30-headless-worker-v1', execution_contract_id=CONTRACT,
        actor='B', arm='fixed_nonzero_left', direction=1, side_arm='teacher', seed=271001,
        output_directory=str(P/'fixed_nonzero_left_01'), control_limit=2200,
        seconds=22., render=False, require_event_driver=False, mode='script',
        qualification=False, model_limits=limits, soft_s=240, close_s=270, hard_s=300,
        spawn_position_m=[-8., -4.7, .455])
    candidate.update(decision='PENDING', execution_contract_id=CONTRACT,
        outer_execution_contract_id=CONTRACT, schema='d1-c30-source-candidate-v1',
        bootstrap_only_candidate=stage == 'bootstrap', worker=str(P/'root30/worker30.py'),
        repository=str(R), host=str(W/'continuation27/director27/host27.py'),
        runtime_environment=env, shared_session=shared, arms={'fixed_nonzero_left': arm},
        x11_events_by_arm={}, inputs=inputs,
        parent_source_go_identity=identity(parent), training_authorized=False,
        candidate_inventory_scope='inherited signed expected hashes plus new exact identities; host performs full pre/post verification',
        review_status='not reviewed; no physics or training GO')
    for key in ('review', 'source_review', 'reviewed_by', 'reviewer', 'approved_arms',
                'review_receipt', 'source_review_receipt', 'review_status_detail', 'bootstrap_receipt'):
        candidate.pop(key, None)
    if stage == 'physical':
        receipt = json.loads((P/'bootstrap_01/receipt.json').read_text())
        host = json.loads((P/'bootstrap_01/host_receipt.json').read_text())
        if not receipt['passed'] or host['exit_code'] != 0 or host['timed_out'] or host['source_mismatches']:
            raise RuntimeError('C30 actual cold bootstrap did not pass')
        for name in ('source_go_bootstrap_candidate_30.json',):
            path = P/name
            candidate['inputs'][str(path)] = identity(path)
        candidate['actual_bootstrap_identity'] = identity(P/'bootstrap_01/receipt.json')
    target = P/f'source_go_{stage}_candidate_30.json'
    write(target, candidate)
    print(json.dumps(dict(path=str(target), sources=len(candidate['inputs']), **identity(target))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('bootstrap', 'physical'), required=True)
    make(parser.parse_args().stage)
