"""One finite, pure saved-data C21 geometry preflight. No plant is built.

Usage: python run_preflight21.py --output NEW_DIRECTORY
The output directory must not exist. A failing qualification is recorded for
every listed case and produces exit 2; it is never silently filtered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
W = HERE.parent
for source in (HERE,W/'course_impl08',W/'rl11',W/'continuation18'):
    sys.path.insert(0,str(source))

import numpy as np
from geometry21 import preflight_finite21, source_derived_templates21
from recipes21 import CASE_SPECS, floor_schedule, make_schedule, select_episode

FORBIDDEN = ('mujoco','torch','stable_baselines3','gym','gymnasium',
             'engine_binding','wheel_legged_control')


def _source(path: Path) -> dict:
    content = path.read_bytes()
    return {'path':str(path),'sha256':hashlib.sha256(content).hexdigest(),
            'bytes':len(content)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('C21 finite preflight output must be a new directory')
    begun = time.monotonic()
    output.mkdir(parents=True)
    traces = {}
    try:
        spec_path = HERE/'spec21.json'
        spec = json.loads(spec_path.read_text())
        if spec['execution_contract_id']!='C21_fixed_B_repair_v1':
            raise ValueError('C21 execution contract identity differs')
        finite = spec['finite_geometry']
        templates,reference_identities = source_derived_templates21(spec)
        fixed = {name:make_schedule(name) for name in CASE_SPECS}
        fixed['floor_0p4_600']=floor_schedule(spec['evaluation']['floor']['seed_stage1'])
        report = preflight_finite21(lambda i:select_episode(i,'A'),
                                    lambda i:select_episode(i,'B'),
                                    fixed,templates,trace_sink=traces)
        if report['pure_servo_advance_count']!=finite['total_pure_servo_advance_max']:
            raise ValueError('finite C21 control count differs from sealed budget')
        if len(report['train_rows'])!=160 or len(report['fixed_eval_rows'])!=16:
            raise ValueError('finite C21 case enumeration is incomplete')
        forbidden_loaded = sorted(name for name in sys.modules
                                  if name.split('.')[0] in FORBIDDEN)
        if forbidden_loaded:
            raise RuntimeError('model/physics/training module imported by pure preflight: '+
                               ','.join(forbidden_loaded))
        sources = [HERE/name for name in ('run_preflight21.py','geometry21.py',
                                          'recipes21.py','spec21.json')]
        sources += [Path(finite[key]) for key in ('flat_saved_geometry',
                                                  'candidate_saved_geometry')]
        for terrain,paths in reference_identities.items():
            if terrain in ('flat','bumps','rough','ramp'):
                sources += [Path(paths['initial_state_path']),Path(paths['control_reset_path'])]
                manifest,initial = templates[terrain]
                manifest_path = output/f'expected_{terrain}_geometry21.json'
                manifest_path.write_text(json.dumps(manifest,sort_keys=True,allow_nan=False)+'\n')
                initial_path = output/f'expected_{terrain}_initial_state21.npz'
                np.savez_compressed(initial_path,**initial)
                control_path = output/f'expected_{terrain}_control_reset21.json'
                control_path.write_text(Path(paths['control_reset_path']).read_text())
                sources += [manifest_path,initial_path,control_path]
        trace_path = output/'all_nominal_traces21.npz'
        np.savez_compressed(trace_path,**traces)
        sources.append(trace_path)
        report.update({'execution_contract_id':spec['execution_contract_id'],
                       'reference_identities':reference_identities,
                       'inputs':{str(path.resolve()):{k:v for k,v in _source(path).items()
                                                       if k!='path'} for path in sources},
                       'forbidden_modules_loaded':forbidden_loaded,
                       'model_calls':0,'physics_calls':0,
                       'elapsed_s':time.monotonic()-begun,
                       'trace_file':str(trace_path),
                       'trace_arrays':len(traces),
                       'qualification_scope':'finite source0..79 and 16 fixed cases only',
                       'future_reset_status':'not observed; every actual reset must be revalidated',
                       'passed':bool(report['all_conditional_template_checks_passed'])})
        (output/'finite_preflight21.json').write_text(
            json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
        return 0 if report['passed'] else 2
    except Exception as error:
        attempted = sum(len(array) for name,array in traces.items()
                        if name.endswith('_raw_forward_yaw'))
        (output/'preflight_exception21.json').write_text(json.dumps({
            'schema':'d1-c21-finite-preflight-exception-v1','passed':False,
            'exception_type':type(error).__name__,'reason':str(error),
            'pure_servo_advance_count_visible':attempted,
            'model_calls':0,'physics_calls':0,'elapsed_s':time.monotonic()-begun,
        },sort_keys=True,indent=2)+'\n')
        raise


if __name__=='__main__':
    raise SystemExit(main())
