"""Bind a separate C18 contract to completed C17 evidence and exact new sources."""
from pathlib import Path
import argparse
import json
from worker18 import identity, write

C = Path(__file__).resolve().parent
OLD = C.parent / 'continuation17'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('development', 'qualification'), required=True)
    args = parser.parse_args()
    qualification = args.phase == 'qualification'
    spec_path = C / (args.phase + '_spec_18.json')
    spec = json.loads(spec_path.read_text())
    parent = C / 'plan_go_B18.json' if qualification else OLD / 'plan_go_C17.json'
    frozen = json.loads(parent.read_text())['inputs']
    for path, expected in frozen.items():
        if identity(path) != expected:
            raise ValueError('frozen source changed: ' + path)
    tests = json.loads((C / spec['tests_receipt_filename']).read_text())
    if not tests['passed'] or tests['changed_sources']:
        raise ValueError('new pure tests did not pass')
    for path, digest in tests['source_sha256'].items():
        if identity(path)['sha256'] != digest:
            raise ValueError('tested source changed: ' + path)
    diagnostic = json.loads((C / 'diagnosis_receipt_18.json').read_text())
    if diagnostic['failure'] is not None or diagnostic['changed_inputs']:
        raise ValueError('saved rough diagnosis failed')
    if identity(C / 'saved_rough_diagnosis_18.json')['sha256'] != diagnostic['output_sha256']:
        raise ValueError('saved rough diagnosis output changed')
    extras = [parent, spec_path, C / spec['contract_filename'], C / spec['go_filename'],
              C / 'plan_18.md', C / spec['tests_receipt_filename']]
    extras += [p for p in C.glob('*.py')]
    extras += [C / name for name in ('diagnosis_receipt_18.json', 'diagnosis_reservation_18.json',
                                    'saved_rough_diagnosis_18.json')]
    diagnosis = json.loads((C / 'saved_rough_diagnosis_18.json').read_text())
    for path, digest in diagnosis['source_sha256'].items():
        if identity(path)['sha256'] != digest:
            raise ValueError('diagnostic input changed: ' + path)
        extras.append(Path(path))
    development_pairs = {}
    if qualification:
        run = C / 'development_filtered_01'
        reader_path = C / 'development_readback_18.json'
        reader = json.loads(reader_path.read_text())
        rh = json.loads((C / 'development_readback_18_host_receipt.json').read_text())
        host = json.loads((run / 'host_receipt.json').read_text())
        if (not reader['source_closure_verified'] or rh['failure'] is not None
                or rh['output_identity'] != identity(reader_path)
                or host['failure'] is not None or host['cleanup_errors'] or host['changed_sources']
                or not host['postcheck_complete'] or not host['no_live_owned_processes']):
            raise ValueError('new development closure failed')
        scores = reader['numeric_scores']
        if ({r['case_id'] for r in scores} != {'rough_0p35', 'ramp_0p45_complete'}
                or len(scores) != 2 or not all(r['task_passed'] for r in scores)
                or not all(r['experiment_actor'] == 'grouped_continue' for r in scores)):
            raise ValueError('both required development tasks must pass before qualification')
        import numpy as np
        for case_id in ('rough_0p35', 'ramp_0p45_complete'):
            relative = Path('heldout') / (case_id + '_grouped_continue') / 'initial_state.npz'
            before = OLD / 'qualification_primary_01' / relative
            after = run / relative
            with np.load(before, allow_pickle=False) as a, np.load(after, allow_pickle=False) as b:
                fields = {key: a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                          and a[key].tobytes() == b[key].tobytes()
                          for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'observation')}
            if not all(fields.values()):
                raise ValueError('development initial state differs from C17: ' + case_id)
            development_pairs[case_id] = {'fields_bitwise_equal': fields,
                'C17_initial': identity(before), 'C18_initial': identity(after)}
        extras += [p for p in run.rglob('*') if p.is_file()]
        extras += [reader_path, C / 'development_readback_18_host_receipt.json',
                   C / 'development_readback_18_reservation.json', C / 'development_filtered_01_reservation.json']
    else:
        extras += [p for p in OLD.rglob('*') if p.is_file()
                   and not any(part in {'.ruff_cache', '__pycache__'} for part in p.relative_to(OLD).parts)]
        outcome = json.loads((OLD / 'qualification_summary_17.json').read_text())
        if outcome['qualification_passed'] or outcome['grouped_passed_scene_count'] != 6:
            raise ValueError('C17 preserved development premise changed')
    if 'GO' not in (C / spec['go_filename']).read_text():
        raise ValueError('exact-source Astra GO absent')
    expected_total = 24000 if qualification else 4000
    if sum(a['controls'] for a in spec['arms'].values()) != expected_total:
        raise ValueError('reserved total differs')
    for arm in spec['arms'].values():
        if arm['controller_variant'] != 'combined' or arm['controls'] != 600 + sum(
                row['horizon'] * len(row['actors']) for row in arm['case_specs']):
            raise ValueError('fixed controller/segments differ')
    for path in extras:
        frozen[str(path)] = identity(path)
    plan = {**spec, 'schema': 'd1-control-repair-plan-18-v1', 'status': 'GO',
            'retry_permitted': False, 'inputs': dict(sorted(frozen.items())),
            'development_precondition_verified': qualification,
            'development_initial_state_pairs': development_pairs,
            'qualification_does_not_treat_seed_as_physical_randomization': True}
    output = C / spec['plan_filename']
    write(output, plan)
    print(json.dumps({'plan': str(output), 'identity': identity(output),
                      'inputs': len(frozen), 'reserved_controls': expected_total}))


if __name__ == '__main__':
    main()
