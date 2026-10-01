"""Freeze a review candidate; this file cannot authorize or launch physics."""
import argparse
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
R = Path('/home/lyh/wheel-legged-control-lab')
CONTRACT = 'C31_event_lateral_RL_pilot_v1'


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parent_path = P.parent/'continuation30/source_go_30.json'
    parent = json.loads(parent_path.read_text())
    sources = dict(parent['inputs'])
    additions = [Path(__file__), P/'pilot_contract_31_draft.md', P/'spec31_draft.json',
        P/'synthetic_torch_contract_31.json', P/'synthetic_torch_01/receipt.json',
        P/'synthetic_saved_readback_01.json', P/'pure_interfaces_01/receipt.json',
        P/'saved_task_check_01/result.json', P/'saved_task_check_01/receipt.json',
        P.parent/'continuation30/paired_trigger_audit_30.json',
        P.parent/'continuation30/final_review_30.json',
        P.parent/'publication30/final_publication_receipt_30.json']
    additions.extend(sorted((P/'root31').glob('*.py')))
    additions.extend(sorted((P/'sol31').glob('*.py')))
    additions.extend(P/'read31'/name for name in ('math_read31.py',
        'learning_read31.py', 'checkpoint_read31.py', 'check_saved_synthetic31.py'))
    for path in additions:
        if str(path) in sources:
            raise RuntimeError('new C31 path overlaps inherited freeze')
        sources[str(path)] = identity(path)
    environment = dict(parent['runtime_environment'])
    old_paths = environment['PYTHONPATH'].split(':')
    environment['PYTHONPATH'] = ':'.join([old_paths[0], str(P/'root31'),
        str(P/'sol31'), *old_paths[1:]])
    environment['CUDA_VISIBLE_DEVICES'] = ''
    shared = dict(parent['shared_session'])
    shared.pop('parent_failed_attempt', None)
    for key in ('execution_contract_id', 'numerical_protocol_contract_id', 'outer_execution_contract_id'):
        shared[key] = CONTRACT
    shared.update(repository_head='a5e6c6b22a37f2dc0417e591b0b3b592f64b2f39',
        seed=271001, policy_seed=310031,
        final_probe_observations54=[[1. if i > 0 and j == i-1 else 0.
                                     for j in range(54)] for i in range(16)])
    cases = [dict(case_id='nominal_learned_left', mode='learned', direction=1, initial_yaw_rad=0., cancel=False),
             dict(case_id='nominal_learned_right', mode='learned', direction=-1, initial_yaw_rad=0., cancel=False)]
    for side, direction, yaw in (('left', 1, .04), ('right', -1, -.04)):
        for mode in ('zero', 'fixed', 'learned'):
            cases.append(dict(case_id=f'heldout_{side}_{mode}', mode=mode,
                              direction=direction, initial_yaw_rad=yaw, cancel=False))
    for side, direction in (('left', 1), ('right', -1)):
        cases.append(dict(case_id=f'cancel_learned_{side}', mode='learned',
                          direction=direction, initial_yaw_rad=0., cancel=True))
    shared['evaluation_cases'] = cases
    limits = dict(load=1, torch_load=4, predict=38401, forward=0, evaluate_actions=0,
                  predict_values=0, backward=64, learn=0, train=0, save=0)
    train = dict(schema='d1-c31-headless-worker-v1', arm='train', actor='B',
        mode='train', render=False, qualification=False, require_event_driver=False,
        control_limit=140800, cycles_limit=64, macros_limit=256,
        soft_s=4200, close_s=4500, hard_s=4560, outer_s=4800,
        seed=271001, policy_seed=310031, side_arm='teacher',
        execution_contract_id=CONTRACT, spawn_position_m=[-8., -4.7, .455],
        model_limits=limits, output_directory=str(P/'train_01'))
    document = dict(schema='d1-c31-source-candidate-v1', decision='PENDING',
        bootstrap_only_candidate=True, training_authorized=False,
        repository=str(R), repository_head=shared['repository_head'],
        execution_contract_id=CONTRACT, outer_execution_contract_id=CONTRACT,
        parent_source_go=dict(path=str(parent_path), **identity(parent_path)),
        worker=str(P/'root31/worker31.py'), host=str(P/'root31/host31.py'),
        runtime_environment=environment, shared_session=shared, arms=dict(train=train),
        x11_events_by_arm={}, inputs=sources,
        future_evaluation_cases_frozen_before_training=cases,
        requires_reviewed_GO_before_model_or_physics=True)
    with args.output.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(candidate=str(args.output), inputs=len(sources), **identity(args.output))))


if __name__ == '__main__':
    main()
