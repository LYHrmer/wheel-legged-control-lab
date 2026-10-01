"""Three fixed C30 saved-only cases against new warm-safe C31 task arithmetic.

Root runs this once; it only reads archives and writes one exclusive receipt.
No model, policy, reader import, native call, or physics integration occurs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from task31 import GATES31, score_task31


W=Path(__file__).resolve().parents[2]
CASES=('fixed_nonzero_left','fixed_nonzero_right','cancel_left')
NUMERIC=('signed_lateral_m','final_xy_error_m','final_yaw_error_rad',
         'finish_support_margin_m','side_duration_s','retention_ratio',
         'max_rollback_m')
GATE_MAP={'entry':'entry','side_physical_safety':'side_physical_safety',
          'rolling_physical_safety':'rolling_physical_safety',
          'all_raw_servo_motion_zero':'zero_raw_and_servo_motion',
          'side_controls_within_1500':'one_side_within_1500',
          'safe_landed_handoff':'safe_landed_handoff',
          'exact_400_retention_complete':'exact_400_retention',
          'original_tail_stopped_and_loaded':'tail_stopped_loaded',
          'B22_predict_restored_in_retention':'B22_predict_restored'}


def identity(path):
    data=Path(path).read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def lines(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream]


def one(case):
    base=W/'continuation30'
    folder=base/f'{case}_01'
    episode=folder/'episode_0'
    paths=[folder/'construction_receipt.json',
           base/f'independent_{case}_01.json',
           episode/'states.npz',episode/'controls.jsonl.gz',
           episode/'macro_transitions30.json']
    paths.extend(sorted(episode.glob('native_block_*.jsonl.gz')))
    if not all(p.is_file() for p in paths) or len(paths)<6:
        raise FileNotFoundError('C31 saved task regression lacks a complete actual case')
    construction=json.loads(paths[0].read_text())
    independent=json.loads(paths[1].read_text())
    with np.load(paths[2],allow_pickle=False) as saved:
        states={key:saved[key].copy() for key in saved.files}
    rows=lines(paths[3])
    macro=json.loads(paths[4].read_text())
    native=[row for path in paths[5:] for row in lines(path)]
    direction=int(macro['direction'])
    cancel_index=next((i for i,row in enumerate(rows)
                       if row['cancel_requested30']),None)
    actual=score_task31(rows,states,native,
        binding=construction['side_kinematic_binding'],
        geometry={int(r['geom_id']):r for r in
                  construction['compiled_geometry']['world_collision_geoms']},
        gates=GATES31,direction=direction,global_control_offset=0,
        cancel_expected=case.startswith('cancel_'),cancel_index=cancel_index)
    old=independent['task']
    for old_key,new_key in GATE_MAP.items():
        if bool(old['gates'][old_key]) != bool(actual['mandatory_gates'][new_key]):
            raise AssertionError(f'{case}: gate {old_key} differs')
    if old['entry_gates']!=actual['entry_gates']:
        raise AssertionError(f'{case}: actual start entry gate differs')
    for key in NUMERIC:
        left,right=old[key],actual[key]
        if (left is None)!=(right is None) or (left is not None and
                abs(float(left)-float(right))>2e-9):
            raise AssertionError(f'{case}: {key} differs')
    if old['side_control_window']!=actual['side_control_window'] or (
            old['retention_control_window']!=actual['retention_control_window']):
        raise AssertionError(f'{case}: physical side/hold window differs')
    passed=(actual['online_safe_cancel'] if case.startswith('cancel_')
            else actual['online_success'])
    if not passed or old['base_task_gates_passed'] is not True:
        raise AssertionError(f'{case}: old qualified task is not online qualified')
    return {'case':case,'passed':True,'online':actual,
            'old_base_task_gates_passed':old['base_task_gates_passed'],
            'inputs':{str(p):identity(p) for p in paths},
            'saved_control_rows':len(rows),'saved_native_rows':len(native)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases=[one(case) for case in CASES]
    result={'schema':'d1-c31-three-saved-task-regression-v1',
            'passed':all(case['passed'] for case in cases),
            'cases':cases,'model_calls':0,'physics_calls':0,
            'scope':'three preexisting complete saved C30 cases only'}
    with args.output.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


if __name__=='__main__':
    main()
