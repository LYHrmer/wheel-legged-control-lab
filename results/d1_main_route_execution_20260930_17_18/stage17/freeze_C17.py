"""Freeze combined qualification only after both bounded development fixes pass."""
from pathlib import Path
import json
from worker17 import identity, write

C = Path(__file__).resolve().parent


def main():
    bplan = C/'plan_go_B17.json'
    frozen = json.loads(bplan.read_text())['inputs']
    for source, expected in frozen.items():
        if identity(source) != expected:
            raise RuntimeError('B frozen input changed: '+source)
    summarypath = C/'development_summary_17.json'
    summary = json.loads(summarypath.read_text())
    if (not all(row['all_five_equal'] for row in summary['cross_worker_initial_state'].values())
            or not summary['task_gates']['yaw']['flat_1p2_yaw']['task_passed']
            or not summary['task_gates']['ramp']['ramp_0p45_complete']['task_passed']):
        raise RuntimeError('separate B fixes and matched initial states did not qualify')
    specpath = C/'qualification_spec_17.json'
    spec = json.loads(specpath.read_text())
    if sum(a['controls'] for a in spec['arms'].values()) != 24000:
        raise RuntimeError('C reserved total differs')
    for arm in spec['arms'].values():
        if (arm['controller_variant'] != 'combined'
                or arm['controls'] != 600+sum(s['horizon']*len(s['actors']) for s in arm['case_specs'])):
            raise RuntimeError('C fixed configuration or control ledger differs')
    if 'GO' not in (C/spec['go_filename']).read_text():
        raise RuntimeError('C exact-source review absent')
    extras = [bplan, summarypath, specpath, C/spec['contract_filename'], C/spec['go_filename'],
              *C.rglob('*.py')]
    for arm in ('baseline', 'yaw', 'ramp'):
        run = C/f'development_{arm}_01'
        host = json.loads((run/'host_receipt.json').read_text())
        worker = json.loads((run/'worker_receipt.json').read_text())
        readback = json.loads((C/f'development_{arm}_readback_17.json').read_text())
        if (host['exit_code'] != 0 or host['changed_sources'] or host['cleanup_errors']
                or not worker['execution_complete'] or not readback['source_closure_verified']):
            raise RuntimeError('B worker source/physical/readback chain not closed')
        extras.extend(p for p in run.rglob('*') if p.is_file())
        extras.extend((C/f'development_{arm}_readback_17.json',
                       C/f'development_{arm}_readback_17_host_receipt.json'))
    for p in extras:
        frozen[str(p)] = identity(p)
    plan = {**spec, 'schema': 'd1-control-repair-plan-17-v1', 'status': 'GO',
            'inputs': dict(sorted(frozen.items())), 'retry_permitted': False,
            'development_precondition_verified': True,
            'qualification_does_not_treat_seed_as_physical_randomization': True}
    path = C/spec['plan_filename']
    write(path, plan)
    print(json.dumps({'plan': str(path), 'identity': identity(path),
                      'source_count': len(frozen), 'reserved_controls': 24000}))


if __name__ == '__main__':
    main()
