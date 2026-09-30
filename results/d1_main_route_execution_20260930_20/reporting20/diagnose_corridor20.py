"""Offline, bounded C20 command corridor audit; no model or physics calls.

All 2*80 original source recipes are reported. Only flat recipes with their
own saved reset geometry/path can receive the actual compiled-geometry guard;
unstarted episodes remain unqualified. An optional single, preselected B18
three-phase candidate is evaluated on B18's saved reset geometry and is never
treated as a training result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import sys

FORBIDDEN = {'mujoco','glfw','torch','stable_baselines3','gym','gymnasium',
             'engine_binding','wheel_legged_control'}
class _NoModelPhysics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in FORBIDDEN:
            raise RuntimeError('offline corridor audit forbids model/physics import: '+fullname)
if any(name.partition('.')[0] in FORBIDDEN for name in sys.modules):
    raise RuntimeError('model/physics already imported before offline audit')
sys.meta_path.insert(0, _NoModelPhysics())
sys.dont_write_bytecode = True

C = Path(__file__).resolve().parent.parent
W = C.parent
for path in (C, W/'course_impl08', W/'rl11', W/'continuation18'):
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))

import numpy as np
from full_drive_command_08 import FullDriveCommand, QualifiedCommandCaps
from full_drive_servo_08 import FullDriveCommandServo
from course_ground_08 import CourseGroundMap
from recipes20 import select_episode
from short_corridor_11 import assert_nominal_corridor_clearance

RUNS = {'A':C/'train_A_1'/'training', 'B':C/'train_B_1'/'training'}
PLAN = C/'plan_go_training_1_20.json'
TABLE = C/'sealed_command_tables_20.json'
CAPS = QualifiedCommandCaps(1.6,0.,0.,.3,False)
SAFE_X, SAFE_Y = 11.,6.
THREE_PHASE = ((425,525,-.25),(525,725,.25),(725,825,-.25))


def identity(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError('missing regular saved source: '+str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    return {'sha256':digest.hexdigest(),'bytes':path.stat().st_size}


def _committed(path):
    sidecar = path.with_name(path.name+'.manifest.json')
    manifest = json.loads(sidecar.read_text())
    if manifest['schema'] != 'd1-archive-transaction-13-v1':
        raise ValueError('archive manifest schema differs: '+str(path))
    members = {item['file']:item for item in manifest['payloads']}
    if path.name not in members or identity(path) != {
            key:members[path.name][key] for key in ('sha256','bytes')}:
        raise ValueError('saved archive transaction differs: '+str(path))
    return identity(path)


def _path(commands, spawn, trace=None):
    servo = FullDriveCommandServo(CAPS)
    x,y = map(float,spawn[:2])
    yaw = 0.
    points = [(x,y)]
    for tick,raw in enumerate(commands):
        applied = servo.advance(tick,raw).applied
        x += math.cos(yaw)*applied.forward_velocity_mps*.01
        y += math.sin(yaw)*applied.forward_velocity_mps*.01
        yaw += applied.yaw_rate_rps*.01
        points.append((x,y))
        if trace is not None:
            trace.append({'tick':tick,'servo':asdict(applied),
                          'heading_rad':yaw,'nominal_xy_m':[x,y]})
    return np.asarray(points,dtype=np.float64),yaw


def _radius(geometry):
    base = np.asarray(geometry['base_world_position_m'][:2],dtype=float)
    return max(float(np.linalg.norm(np.asarray(g['world_center_m'][:2])-base)
                     +g['rbound_m']) for g in geometry['robot_collision_geoms'])


def _bounds(path,radius):
    abs_x = np.abs(path[:,0])+radius
    abs_y = np.abs(path[:,1])+radius
    bad = np.flatnonzero((abs_x>=SAFE_X)|(abs_y>=SAFE_Y))
    return {'radius_m':radius,
            'nominal_x_min_m':float(path[:,0].min()),
            'nominal_x_max_m':float(path[:,0].max()),
            'nominal_y_min_m':float(path[:,1].min()),
            'nominal_y_max_m':float(path[:,1].max()),
            'max_abs_x_plus_radius_m':float(abs_x.max()),
            'max_abs_y_plus_radius_m':float(abs_y.max()),
            'x_safety_margin_m':float(SAFE_X-abs_x.max()),
            'y_safety_margin_m':float(SAFE_Y-abs_y.max()),
            'first_safety_bound_touch_point':None if len(bad)==0 else int(bad[0]),
            'safety_bound_touch_points':int(len(bad)),
            'endpoint_xy_m':path[-1].tolist()}


def _obstacle_clearance_lower_bound(path,geometry,radius):
    """Conservative continuous-path lower bound; original guard remains authority."""
    terrain = CourseGroundMap(geometry['terrain_world_geoms'])
    best = float('inf')
    closest = None
    for geom,low,high in zip(terrain._boxes,terrain._box_xy_low,
                             terrain._box_xy_high,strict=True):
        lo,hi = low-radius,high+radius
        delta = np.maximum(np.maximum(lo-path,path-hi),0.)
        distances = np.linalg.norm(delta,axis=1)
        at = int(np.argmin(distances))
        if float(distances[at]) < best:
            best,closest = float(distances[at]),{'geom_id':geom.geom_id,'point':at}
    max_step = float(np.linalg.norm(np.diff(path,axis=0),axis=1).max())
    return {'minimum_sampled_point_distance_m':best,
            'continuous_segment_clearance_lower_bound_m':best-.5*max_step,
            'maximum_centerline_segment_m':max_step,'closest_sample':closest,
            'meaning':'Lower bound to radius-expanded compiled XY boxes; exact collision decision uses original guard'}


def _saved_flat(arm,source,expected_path):
    stem = f'training_episode_{source:06d}_flat_corridor'
    folder = RUNS[arm]
    geom_file = folder/f'{stem}_geometry.json'
    path_file = folder/f'{stem}_nominal_path.npz'
    if not geom_file.exists() and not path_file.exists():
        return None
    if not geom_file.is_file() or not path_file.is_file():
        return {'status':'unqualified_incomplete_saved_reset_geometry_pair'}
    geometry_id = _committed(geom_file)
    path_id = _committed(path_file)
    geometry = json.loads(geom_file.read_text())
    with np.load(path_file,allow_pickle=False) as saved:
        if saved.files != ['xy_m']:
            raise ValueError('saved nominal path field differs: '+str(path_file))
        observed = saved['xy_m']
    if observed.shape != (1001,2) or not np.array_equal(observed,expected_path):
        raise ValueError('saved actual-reset nominal path differs from frozen recipe: '+str(path_file))
    if not np.array_equal(np.asarray(geometry['base_world_position_m'][:2]),expected_path[0]):
        raise ValueError('saved reset base is not this recipe spawn: '+str(geom_file))
    radius = _radius(geometry)
    outcome = {'status':None,'geometry_file':str(geom_file),'geometry_identity':geometry_id,
               'nominal_path_file':str(path_file),'nominal_path_identity':path_id,
               'saved_path_bitwise_equal_to_recomputed':True,
               'actual_reset_geometry_not_translated':True,
               'obstacle_clearance':_obstacle_clearance_lower_bound(expected_path,geometry,radius),
               **_bounds(expected_path,radius)}
    try:
        proof = assert_nominal_corridor_clearance(geometry,expected_path)
        outcome['status'] = 'actual_saved_reset_guard_pass'
        outcome['guard_report'] = proof
    except ValueError as error:
        outcome['status'] = 'actual_saved_reset_guard_reject'
        outcome['guard_error'] = str(error)
    return outcome


def _commands_sha(commands):
    return hashlib.sha256(json.dumps([asdict(c) for c in commands],
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def diagnose(*,output,candidate_three_phase=False):
    output = Path(output).resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError('offline diagnostic output must be a new file in existing directory')
    plan = json.loads(PLAN.read_text())
    table = json.loads(TABLE.read_text())
    if plan['status'] != 'GO' or plan['retry_permitted'] is not False:
        raise ValueError('frozen C20 source GO identity differs')
    needed = (C/'recipes20.py',W/'course_impl08/full_drive_servo_08.py',
              W/'rl11/short_corridor_11.py',TABLE)
    for path in needed:
        if plan['inputs'].get(str(path)) != identity(path):
            raise ValueError('frozen pure input changed: '+str(path))
    rows = []
    for arm in ('A','B'):
        frozen = table['training_first_80_sources'][arm]
        if len(frozen) != 80:
            raise ValueError('frozen first80 command table count differs')
        for source in range(80):
            recipe = select_episode(source,arm)
            choice = json.loads(recipe.choice_json)
            if (frozen[source]['source_index'] != source
                    or frozen[source]['choice'] != choice
                    or frozen[source]['command_sha256'] != recipe.command_sha256):
                raise ValueError('first80 command hash/choice changed')
            path,heading = _path(recipe.raw_commands,recipe.spawn_position_m)
            coarse = {
                'abs_x_max_m':float(np.abs(path[:,0]).max()),
                'abs_y_max_m':float(np.abs(path[:,1]).max()),
                'flat_y_max_m':float(path[:,1].max()),
                'passed':bool(np.abs(path[:,0]).max()<10.5
                         and np.abs(path[:,1]).max()<5.8
                         and (recipe.terrain!='flat' or path[:,1].max() < -3.6))}
            row = {'arm':arm,'source_episode_index':source,'terrain':recipe.terrain,
                   'choice':choice,'command_sha256':recipe.command_sha256,
                   'spawn_position_m':list(recipe.spawn_position_m),
                   'nominal_endpoint_xy_m':path[-1].tolist(),
                   'nominal_endpoint_heading_rad':heading,
                   'pre_geometry_nominal_path_gate':coarse,
                   'actual_reset_flat_guard':'not_applicable_nonflat' if recipe.terrain!='flat'
                                              else 'unqualified_missing_actual_saved_reset_geometry'}
            if recipe.terrain=='flat':
                saved = _saved_flat(arm,source,path)
                if saved is not None:
                    row['actual_reset_flat_guard'] = saved
            rows.append(row)
    failure = next(row for row in rows if row['arm']=='B' and row['source_episode_index']==18)
    if not isinstance(failure['actual_reset_flat_guard'],dict) or failure['actual_reset_flat_guard']['status'] != 'actual_saved_reset_guard_reject':
        raise ValueError('B18 saved failure guard is not reproduced from actual reset geometry')
    candidate = None
    if candidate_three_phase:
        base = select_episode(18,'B')
        commands = tuple(FullDriveCommand(
            forward_velocity_mps=old.forward_velocity_mps,
            yaw_rate_rps=next((yaw for start,end,yaw in THREE_PHASE if start<=i<end),0.))
            for i,old in enumerate(base.raw_commands))
        trace = []
        path,heading = _path(commands,base.spawn_position_m,trace)
        saved = failure['actual_reset_flat_guard']
        geometry = json.loads(Path(saved['geometry_file']).read_text())
        candidate = {'identity':'single_B_source18_three_phase_100_200_100',
                     'yaw_segments':[list(x) for x in THREE_PHASE],
                     'speed_mps':1.0,'command_sha256':_commands_sha(commands),
                     'nominal_path_xy_m':path.tolist(),
                     'nominal_endpoint_heading_rad':heading,
                     'raw_commands':[asdict(c) for c in commands],
                     'servo_integration_trace':trace,
                     'actual_B18_reset_geometry_identity':saved['geometry_identity'],
                     'same_saved_reset_geometry_counterfactual_not_physics':True,
                     'obstacle_clearance':_obstacle_clearance_lower_bound(path,geometry,_radius(geometry)),
                     **_bounds(path,_radius(geometry))}
        try:
            proof = assert_nominal_corridor_clearance(geometry,path)
            candidate['status'] = 'offline_actual_B18_geometry_guard_pass'
            candidate['guard_report'] = proof
        except ValueError as error:
            candidate['status'] = 'offline_actual_B18_geometry_guard_reject'
            candidate['guard_error'] = str(error)
    result = {'schema':'d1-c20-frozen-first80-offline-corridor-audit-v1',
              'plan_identity':identity(PLAN),'command_table_identity':identity(TABLE),
              'pure_source_identities':{str(p):identity(p) for p in needed},
              'arms':['A','B'],'sources_per_arm':80,'nominal_control_intervals_considered':160000,
              'actual_pure_servo_advance_calls':160000+1000*int(candidate_three_phase),
              'candidate_pure_intervals':1000*int(candidate_three_phase),
              'all_original_rows_included':len(rows)==160,'rows':rows,
              'B18_failure':failure,'single_preselected_candidate':candidate,
              'model_calls':0,'physics_calls':0,'training_updates':0,
              'limits':'Nominal command/actual saved reset geometry only; no new trajectory or model'}
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--candidate-three-phase',action='store_true')
    args = parser.parse_args()
    result = diagnose(output=args.output,candidate_three_phase=args.candidate_three_phase)
    print(json.dumps({'rows':len(result['rows']),
                      'B18':result['B18_failure']['actual_reset_flat_guard']['status'],
                      'candidate':None if result['single_preselected_candidate'] is None else
                                  result['single_preselected_candidate']['status']}))


if __name__=='__main__':
    main()
