"""Build the preregistered final-evaluation candidate only after training readback."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-readback', type=Path, required=True)
    parser.add_argument('--training-execution', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train = P/'train_01'
    report_path = args.training_readback.resolve(strict=True)
    execution_path = args.training_execution.resolve(strict=True)
    report, execution = read(report_path), read(execution_path)
    worker, host = read(train/'worker_receipt.json'), read(train/'host_receipt.json')
    supervisor = read(train/'supervisor_receipt.json')
    if (report.get('training_valid') is not True or report.get('record_valid') is not True
            or report.get('real_actor_update') is not True or Path(report['run']) != train
            or execution.get('exit_code') != 0 or worker['execution_complete'] is not True
            or host['exit_code'] != 0 or host['failure'] is not None
            or host['source_mismatches'] or not host['owned_no_orphans']
            or supervisor['exit_code'] != 0 or supervisor['failure'] is not None
            or supervisor['cleanup']['remaining']):
        raise RuntimeError('C31 requires complete independently valid unique training first')
    expected_command = ['rtk', 'proxy', '/usr/bin/python3', '-B', str(P/'read31/read31_v2.py'),
                        '--run', str(train), '--output', str(report_path)]
    if (report['session_identity'] != identity(train/'session.json')
            or report['worker_identity'] != identity(train/'worker_receipt.json')
            or report['checkpoint']['passed'] is not True
            or report['checkpoint']['checkpoint_path'] != str(train/'final_checkpoint/lateral_final.pt')
            or report['checkpoint']['checkpoint_identity'] != identity(train/'final_checkpoint/lateral_final.pt')
            or execution.get('schema') != 'd1-c31-saved-reader-execution-v1'
            or execution.get('arm') != 'training' or execution.get('failure') is not None
            or execution.get('source_mismatches') != []
            or execution.get('owned_child_reaped') is not True
            or execution.get('command') != expected_command
            or execution.get('source_review_identity') != identity(P/'reader_source_review_31_v2.json')
            or execution.get('output_path') != str(report_path)
            or execution.get('output_identity') != identity(report_path)):
        raise RuntimeError('training readback, execution receipt and actual final files are not bound')
    go = copy.deepcopy(read(P/'source_go_train_31.json'))
    for key in ('review', 'reviewer', 'reviewed_at_utc', 'reviewed_at'):
        go.pop(key, None)
    go.update(decision='PENDING', bootstrap_only_candidate=False, training_authorized=False,
              evaluation_authorized=False, review_status='awaiting actual final-evaluation source review',
              requires_reviewed_GO_before_model_or_physics=True,
              parent_train_source_go=dict(path=str(P/'source_go_train_31.json'),
                                         **identity(P/'source_go_train_31.json')),
              candidate_kind='same frozen worker and preregistered list; unique final checkpoint only')
    sources = go['inputs']
    additions = [Path(__file__), P/'launch_saved_reader31_v2.py', report_path, execution_path,
        P/'source_go_train_31.json', P/'request_train_31.json', P/'train_01_reservation.json',
        P/'reader_source_review_31_v2.json', P/'eval_reader_source_review_31_v2.json',
        P/'eval_reader_interfaces_01/receipt.json',
        train/'session.json', train/'worker_receipt.json', train/'host_receipt.json',
        train/'supervisor_receipt.json', train/'training_receipt31.json',
        train/'final_parameters31.json', train/'final_parameters31.json.manifest.json',
        train/'final_checkpoint_receipt31.json', train/'final_checkpoint_receipt31.json.manifest.json',
        P.parent/'continuation30/pair_read30/audit_pairs30.py']
    additions.extend(sorted((train/'final_checkpoint').iterdir()))
    additions.extend(sorted((P/'read31').glob('*.py')))
    for review_name in ('reader_source_review_31_v2.json', 'eval_reader_source_review_31_v2.json'):
        review = read(P/review_name)
        if review['decision'] != 'GO':
            raise RuntimeError('C31 independent reader source is not reviewed')
        for file, expected in review['inputs'].items():
            if identity(file) != expected:
                raise RuntimeError('C31 reviewed saved-reader input changed: '+file)
            additions.append(Path(file))
    for case in ('fixed_nonzero_left', 'fixed_nonzero_right', 'zero_left', 'zero_right'):
        run = P.parent/'continuation30'/f'{case}_01'
        additions.extend(run/name for name in ('session.json', 'worker_receipt.json',
            'host_receipt.json', 'supervisor_receipt.json', 'construction_receipt.json',
            'construction_receipt.json.manifest.json', 'runtime_module_origins.json',
            'runtime_module_origins.json.manifest.json', 'loaded_origins_final.json',
            'loaded_origins_final.json.manifest.json'))
        additions.append(P.parent/'continuation30'/f'independent_{case}_01.json')
        for name in ('initial_state.npz', 'states.npz', 'controls.jsonl.gz', 'reset.json',
                     'macro_transitions30.json', 'segment_receipt.json'):
            additions.extend((run/'episode_0'/name, run/'episode_0'/(name+'.manifest.json')))
    for path in additions:
        current = identity(path)
        if str(path) in sources and sources[str(path)] != current:
            raise RuntimeError('C31 frozen predecessor changed: '+str(path))
        sources[str(path)] = current
    shared = go['shared_session']
    shared.update(training_readback_path=str(report_path), training_readback_identity=identity(report_path),
        training_readback_execution_path=str(execution_path), training_run=str(train),
        lateral_checkpoint_folder=str(train/'final_checkpoint'),
        lateral_checkpoint_manifest=read(train/'final_checkpoint_receipt31.json'))
    arm = dict(go['arms']['train'])
    arm.update(arm='evaluation', mode='eval', control_limit=22000, cycles_limit=10, macros_limit=40,
        soft_s=900, close_s=960, hard_s=1020, outer_s=1500, output_directory=str(P/'eval_01'),
        model_limits=dict(load=1, torch_load=4, predict=6001, forward=0, evaluate_actions=0,
                          predict_values=0, backward=0, learn=0, train=0, save=0))
    go['arms'] = dict(evaluation=arm)
    with args.output.open('x') as stream:
        json.dump(go, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(candidate=str(args.output), inputs=len(sources), **identity(args.output))))


if __name__ == '__main__':
    main()
