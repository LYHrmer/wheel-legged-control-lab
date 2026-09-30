"""Delta-only C22 finite geometry preflight: four revised yaw cases, 6400 ticks.

Usage: python run_preflight22.py --output NEW_DIRECTORY
The sealed C21 result is reused only for its 160 passed training rows and
12 passed unchanged evaluation rows. Its four failed rows remain historical.
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
for directory in (HERE,W/'continuation21',W/'continuation18',W/'course_impl08',W/'rl11'):
    sys.path.insert(0,str(directory))

import numpy as np
from geometry21 import (nominal_path21,_qualify_path21,command_sha256,
                        SAFE_X_M,SAFE_Y_M)
from geometry_runtime22 import load_sealed_templates22
from recipes22 import CASE_SPECS, floor_schedule, make_schedule, select_episode

FORBIDDEN = ('mujoco','torch','stable_baselines3','gym','gymnasium',
             'engine_binding','wheel_legged_control')


def identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def _check_prior(prior: dict, spec: dict) -> tuple[dict,dict,dict]:
    finite = spec['finite_geometry']
    path = Path(finite['prior_preflight_path'])
    if identity(path)!=finite['prior_preflight_identity']:
        raise ValueError('C21 preflight report identity differs')
    if identity(Path(finite['prior_spec_path']))!=finite['prior_spec_identity']:
        raise ValueError('C21 frozen spec identity differs')
    if (prior.get('passed') is not False
            or prior.get('pure_servo_advance_count')!=184800
            or len(prior.get('train_rows',[]))!=160
            or len(prior.get('fixed_eval_rows',[]))!=16):
        raise ValueError('C21 preflight provenance/count differs')
    for raw_path,expected in prior['inputs'].items():
        if identity(Path(raw_path))!=expected:
            raise ValueError('C21 preflight source/template/trace changed: '+raw_path)
    train = {(row['arm'],row['source_episode_index']):row
             for row in prior['train_rows']}
    if len(train)!=160 or not all(row['conditional_template_passed'] for row in train.values()):
        raise ValueError('C21 training source0..79 did not all qualify')
    for arm in ('A','B'):
        for source in range(80):
            row = train[(arm,source)]
            if command_sha256(select_episode(source,arm))!=row['command_sha256']:
                raise ValueError('C22 training command differs from qualified C21 source')
    old_eval = {row['case_id']:row for row in prior['fixed_eval_rows']}
    changed = set(finite['new_preflight_case_ids'])
    if (len(old_eval)!=16 or changed!=set(finite['prior_failed_evaluation_case_ids'])
            or any(old_eval[case]['conditional_template_passed'] for case in changed)):
        raise ValueError('C21 four failed historical yaw rows differ')
    unchanged = {}
    for case_id,row in old_eval.items():
        if case_id in changed:
            continue
        schedule = (floor_schedule(spec['evaluation']['floor']['seed_stage1'])
                    if case_id=='floor_0p4_600' else make_schedule(case_id))
        if not row['conditional_template_passed'] or command_sha256(schedule)!=row['command_sha256']:
            raise ValueError('C22 unchanged evaluation command differs from C21 qualified row')
        unchanged[case_id]=row
    if len(unchanged)!=12:
        raise ValueError('C21 qualified unchanged evaluation count differs')
    templates,controls,template_ids = load_sealed_templates22(spec)
    return unchanged,old_eval,{'templates':templates,'controls':controls,
                               'template_identities':template_ids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('C22 delta preflight output must be a new directory')
    output.mkdir(parents=True)
    begun = time.monotonic()
    traces = {}
    try:
        spec_path = HERE/'spec22.json'
        spec = json.loads(spec_path.read_text())
        finite = spec['finite_geometry']
        if spec['execution_contract_id']!='C22_fixed_B_eval_repair_v1':
            raise ValueError('C22 execution contract identity differs')
        prior = json.loads(Path(finite['prior_preflight_path']).read_text())
        unchanged,old_eval,loaded = _check_prior(prior,spec)
        templates = loaded['templates']
        rows = []
        controls = 0
        for case_id in finite['new_preflight_case_ids']:
            schedule = make_schedule(case_id)
            if schedule.terrain!='flat' or len(schedule.raw_commands)!=1600:
                raise ValueError('C22 new yaw case terrain/horizon differs')
            path,traced = nominal_path21(schedule,trace=True)
            controls += len(schedule.raw_commands)
            for suffix,array in (('path_xy_m',path),
                                 ('raw_forward_yaw',np.asarray(traced['raw_forward_yaw'])),
                                 ('servo_forward_yaw',np.asarray(traced['servo_forward_yaw']))):
                traces[case_id+'_'+suffix]=array
            nominal = {k:v for k,v in traced.items() if k not in
                       ('raw_forward_yaw','servo_forward_yaw')}
            manifest,initial = templates['flat']
            base = np.asarray(manifest['base_world_position_m'])
            radius = max(float(np.linalg.norm(np.asarray(g['world_center_m'])[:2]-base[:2])
                               +g['rbound_m']) for g in manifest['robot_collision_geoms'])
            xexp = np.abs(path[:,0])+radius
            yexp = np.abs(path[:,1])+radius
            bad = np.flatnonzero((xexp>=SAFE_X_M)|(yexp>=SAFE_Y_M))
            row = {'case_id':case_id,'terrain':'flat','controls':1600,
                   'command_sha256':command_sha256(schedule),
                   'path_sha256':nominal['path_sha256'],
                   'old_C21_failed_command_sha256':old_eval[case_id]['command_sha256'],
                   'evidence_basis':'source_derived_expected_reset_not_observed_future_reset',
                   'reset_robot_radial_bound_m':radius,
                   'maximum_expanded_abs_x_m':float(xexp.max()),
                   'maximum_expanded_abs_y_m':float(yexp.max()),
                   'x_safety_margin_m':float(SAFE_X_M-xexp.max()),
                   'y_safety_margin_m':float(SAFE_Y_M-yexp.max()),
                   'first_safety_bound_bad_point':int(bad[0]) if len(bad) else None}
            try:
                receipt,_ = _qualify_path21(schedule,manifest,initial,path,nominal,
                    kind='fixed_eval',case_id=case_id,evidence_basis=row['evidence_basis'])
                row['conditional_template_passed']=True
                row['conditional_receipt']=receipt
            except (ValueError,TypeError,KeyError) as error:
                row['conditional_template_passed']=False
                row['reason']=str(error)
            rows.append(row)
        if controls!=6400 or controls!=finite['total_pure_servo_advance_max']:
            raise ValueError('C22 new pure servo budget differs')
        loaded_forbidden = sorted(name for name in sys.modules
                                  if name.split('.')[0] in FORBIDDEN)
        if loaded_forbidden:
            raise RuntimeError('pure delta preflight loaded forbidden model/physics module')
        trace_path = output/'four_revised_nominal_traces22.npz'
        np.savez_compressed(trace_path,**traces)
        sources = [HERE/name for name in ('run_preflight22.py','geometry_runtime22.py',
                                          'recipes22.py','spec22.json')]
        sources += [W/'continuation21'/'geometry21.py',
                    Path(finite['prior_preflight_path']),Path(finite['prior_spec_path']),
                    trace_path]
        report = {'schema':'d1-c22-delta-finite-geometry-preflight-v1',
                  'execution_contract_id':spec['execution_contract_id'],
                  'passed':all(row['conditional_template_passed'] for row in rows),
                  'new_rows':rows,'new_pure_servo_advance_count':controls,
                  'reused_C21_training_rows':160,
                  'reused_C21_unchanged_evaluation_rows':unchanged,
                  'historical_C21_failed_rows':{case:old_eval[case] for case in
                                                finite['new_preflight_case_ids']},
                  'reused_nominal_intervals':finite['reused_nominal_intervals'],
                  'qualified_schedule_nominal_intervals':finite['qualified_schedule_nominal_intervals'],
                  'old_C21_actual_pure_servo_calls':finite['prior_C21_actual_pure_servo_calls'],
                  'actual_future_resets_verified':False,
                  'inputs':{**prior['inputs'],**loaded['template_identities'],
                            **{str(path.resolve()):identity(path) for path in sources}},
                  'sealed_template_inputs':loaded['template_identities'],
                  'trace_file':str(trace_path),'trace_arrays':len(traces),
                  'model_calls':0,'physics_calls':0,'forbidden_modules_loaded':loaded_forbidden,
                  'elapsed_s':time.monotonic()-begun}
        (output/'finite_delta_preflight22.json').write_text(
            json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
        return 0 if report['passed'] else 2
    except Exception as error:
        attempted = sum(len(array) for name,array in traces.items()
                        if name.endswith('_raw_forward_yaw'))
        (output/'preflight_exception22.json').write_text(json.dumps({
            'schema':'d1-c22-delta-preflight-exception-v1','passed':False,
            'exception_type':type(error).__name__,'reason':str(error),
            'new_pure_servo_advance_count_visible':attempted,
            'model_calls':0,'physics_calls':0,'elapsed_s':time.monotonic()-begun,
        },sort_keys=True,indent=2)+'\n')
        raise


if __name__=='__main__':
    raise SystemExit(main())
