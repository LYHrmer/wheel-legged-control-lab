"""Pure C21 finite-course geometry qualification and new-reset bridge.

This module accepts typed saved/pure schedules and a caller-supplied reset
manifest. It never constructs a plant. A template result predicts eligibility
only; the runtime must call ``qualify_reset21`` again with each newly observed
reset before starting that episode's first physical control.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from course_ground_08 import CourseGroundMap
from full_drive_command_08 import QualifiedCommandCaps
from full_drive_servo_08 import FullDriveCommandServo
from short_corridor_11 import (_segment_hits_aabb, SAFE_X_M, SAFE_Y_M,
                               assert_nominal_corridor_clearance)

SCHEMA = 'd1-c21-finite-command-compiled-reset-geometry-v1'
FIRST_SOURCE, AFTER_LAST_SOURCE = 0, 80
CAPS = QualifiedCommandCaps(1.6, 0., 0., .3, False)
ALLOWED_TERRAINS = frozenset({'flat','bumps','rough','ramp'})
INITIAL_FIELDS = ('qpos','qvel','ctrl','qacc_warmstart','observation','time')


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def command_sha256(schedule: Any) -> str:
    return _canonical_sha([asdict(row) for row in schedule.raw_commands])


def nominal_path21(schedule: Any, *, caps: QualifiedCommandCaps = CAPS,
                   trace: bool = False) -> tuple[np.ndarray, dict]:
    """Reproduce the frozen command-servo integration, without a live plant."""
    _need(isinstance(caps, QualifiedCommandCaps) and caps == CAPS,
          'C21 caps differ from the frozen forward/yaw-only authority')
    commands = schedule.raw_commands
    spawn = tuple(schedule.spawn_position_m)
    _need(schedule.terrain in ALLOWED_TERRAINS
          and isinstance(commands,(tuple,list)) and len(commands) in (600,1000,1600,1800)
          and len(spawn)==3 and all(math.isfinite(float(x)) for x in spawn),
          'C21 schedule/horizon/spawn invalid')
    servo = FullDriveCommandServo(caps)
    x,y = float(spawn[0]),float(spawn[1])
    heading = 0.
    path = [(x,y)]
    applied_rows = []
    for tick,raw in enumerate(commands):
        applied = servo.advance(tick,raw).applied
        if trace:
            applied_rows.append((applied.forward_velocity_mps, applied.yaw_rate_rps))
        x += math.cos(heading)*applied.forward_velocity_mps*.01
        y += math.sin(heading)*applied.forward_velocity_mps*.01
        heading += applied.yaw_rate_rps*.01
        path.append((x,y))
    points = np.asarray(path,dtype=np.float64)
    _need(points.shape==(len(commands)+1,2) and np.isfinite(points).all(),
          'C21 integrated nominal path invalid')
    details = {'command_sha256':command_sha256(schedule),
                    'controls':len(commands),'path_sha256':hashlib.sha256(
                        points.tobytes(order='C')).hexdigest(),
                    'endpoint_heading_rad':heading,
                    'endpoint_xy_m':points[-1].tolist(),
                    'spawn_position_m':list(spawn)}
    if trace:
        details['raw_forward_yaw'] = [[c.forward_velocity_mps,c.yaw_rate_rps] for c in commands]
        details['servo_forward_yaw'] = applied_rows
    return points,details


def _reset_manifest(manifest: dict, initial_state: dict, schedule: Any) -> tuple[float,CourseGroundMap]:
    _need(type(manifest) is dict and set(manifest)=={
        'model_identity','base_world_position_m','robot_collision_geoms','terrain_world_geoms'},
          'actual reset geometry manifest lacks exact fields')
    ident = manifest['model_identity']
    _need(type(ident) is dict and ident.get('source')=='actual_compiled_course_reset_geometry_v1'
          and type(ident.get('model_ngeom')) is int and ident['model_ngeom']>92,
          'actual compiled reset model identity invalid')
    base = np.asarray(manifest['base_world_position_m'],dtype=np.float64)
    qpos = np.asarray(initial_state['qpos'],dtype=np.float64)
    _need(base.shape==(3,) and qpos.ndim==1 and len(qpos)>=7
          and np.isfinite(base).all() and np.isfinite(qpos).all()
          and np.array_equal(base,qpos[:3])
          and np.array_equal(base,np.asarray(schedule.spawn_position_m,dtype=float))
          and float(initial_state['time'])==0.,
          'actual new reset base/initial state differs from selected spawn')
    for field in ('qvel','ctrl','qacc_warmstart','observation'):
        arr = np.asarray(initial_state[field])
        _need(arr.size>0 and np.isfinite(arr).all(),
              'actual reset state field invalid: '+field)
    terrain = CourseGroundMap(manifest['terrain_world_geoms'])
    _need(len(terrain.geom_ids)==92,'actual compiled world geometry count differs')
    robot = manifest['robot_collision_geoms']
    _need(isinstance(robot,list) and len(robot)>0,'actual reset robot collision geoms missing')
    ids = set()
    radius = 0.
    for row in robot:
        gid,body = row['geom_id'],row['body_id']
        center = np.asarray(row['world_center_m'],dtype=np.float64)
        bound = float(row['rbound_m'])
        _need(type(gid) is int and 0<=gid<ident['model_ngeom']
              and gid not in ids and gid not in terrain.geom_ids
              and type(body) is int and body>0 and row['collision'] is True
              and center.shape==(3,) and np.isfinite(center).all()
              and math.isfinite(bound) and bound>=0.,
              'actual reset robot compiled collision row invalid')
        radius = max(radius,float(np.linalg.norm(center[:2]-base[:2])+bound))
        ids.add(gid)
    return radius,terrain


def _generic_flat_guard(path: np.ndarray, manifest: dict, radius: float,
                        terrain: CourseGroundMap) -> dict:
    _need(not np.any(np.abs(path[:,0])+radius>=SAFE_X_M)
          and not np.any(np.abs(path[:,1])+radius>=SAFE_Y_M),
          'nominal centerline plus actual reset robot envelope touches course safety bounds')
    for geom,low,high in zip(terrain._boxes,terrain._box_xy_low,
                             terrain._box_xy_high,strict=True):
        lo,hi = low-radius,high+radius
        for index in range(len(path)-1):
            _need(not _segment_hits_aabb(path[index],path[index+1],lo,hi),
                  f'nominal reset-footprint corridor intersects geom {geom.geom_id} at segment {index}')
    return {'flat_compiled_obstacle_segments_checked':len(path)-1,
            'flat_obstacle_guard':'pass'}


def qualify_reset21(schedule: Any, actual_manifest: dict, initial_state: dict,
                    *, kind: str, source_index: int|None=None,
                    case_id: str|None=None) -> tuple[dict,np.ndarray]:
    """Qualify *one* actual new reset; this function itself has no physics call.

    ``kind='train'`` accepts only source indices 0..79 and exactly 1000
    controls. ``kind='fixed_eval'`` needs a named frozen case and accepts the
    fixed 1600/1800-control schedules. Nonflat terrain is checked against the
    global envelope; its intended obstacle contacts are not forbidden.
    """
    path,nominal = nominal_path21(schedule)
    return _qualify_path21(schedule,actual_manifest,initial_state,path,nominal,
                           kind=kind,source_index=source_index,case_id=case_id,
                           evidence_basis='actual_new_compiled_reset_geometry_and_initial_state')


def _qualify_path21(schedule: Any, actual_manifest: dict, initial_state: dict,
                    path: np.ndarray, nominal: dict, *, kind: str,
                    source_index: int|None=None, case_id: str|None=None,
                    evidence_basis: str) -> tuple[dict,np.ndarray]:
    if kind=='train':
        _need(type(source_index) is int and FIRST_SOURCE<=source_index<AFTER_LAST_SOURCE
              and len(schedule.raw_commands)==1000 and case_id is None,
              'training geometry qualification outside frozen source0..79')
        target = {'kind':'train','source_episode_index':source_index}
    elif kind=='fixed_eval':
        _need(type(case_id) is str and case_id and source_index is None
              and len(schedule.raw_commands) in (600,1600,1800),
              'fixed evaluation geometry qualification lacks frozen case/horizon')
        target = {'kind':'fixed_eval','case_id':case_id}
    else:
        raise ValueError('unknown C21 geometry qualification kind')
    radius,terrain = _reset_manifest(actual_manifest,initial_state,schedule)
    absx = np.abs(path[:,0])+radius
    absy = np.abs(path[:,1])+radius
    _need(np.isfinite(absx).all() and np.isfinite(absy).all(),
          'nominal expanded bounds are nonfinite')
    # Course map/lane envelope applies to every frozen nominal path.
    _need(np.max(np.abs(path[:,0]))<10.5
          and np.max(np.abs(path[:,1]))<5.8
          and (schedule.terrain!='flat' or np.max(path[:,1])< -3.6),
          'selected nominal path exceeds its declared course lane')
    _need(float(absx.max())<SAFE_X_M and float(absy.max())<SAFE_Y_M,
          'nominal centerline plus actual reset robot envelope touches course safety bounds')
    if schedule.terrain=='flat':
        if kind=='train':
            old = assert_nominal_corridor_clearance(actual_manifest,path)
            obstacle = {'flat_obstacle_guard':'frozen_short_corridor_11_pass',
                        'frozen_guard_schema':old['schema']}
        else:
            obstacle = _generic_flat_guard(path,actual_manifest,radius,terrain)
    else:
        obstacle = {'flat_obstacle_guard':'not_applicable_nonflat_intended_terrain_contact'}
    receipt = {'schema':SCHEMA,**target,'terrain':schedule.terrain,
               'evidence_basis':evidence_basis,
               'nominal_only_not_actual_trajectory':True,
               'actual_reset_geometry_sha256':_canonical_sha(actual_manifest),
               'actual_initial_state_digest':_canonical_sha({
                   name:np.asarray(initial_state[name]).tolist() for name in
                   ('qpos','qvel','ctrl','qacc_warmstart','observation','time')}),
               'model_identity':dict(actual_manifest['model_identity']),
               'reset_robot_radial_bound_m':radius,
               'x_safety_margin_m':float(SAFE_X_M-absx.max()),
               'y_safety_margin_m':float(SAFE_Y_M-absy.max()),
               'nominal_bounds_m':{'x_min':float(path[:,0].min()),
                                   'x_max':float(path[:,0].max()),
                                   'y_min':float(path[:,1].min()),
                                   'y_max':float(path[:,1].max())},
               **nominal,**obstacle,'passed':True}
    return receipt,path


def reset_homotopy21(reference_initial: dict, new_initial: dict,
                     reference_control_reset: dict,
                     new_control_reset: dict) -> dict:
    """Compare two observed resets; seed equality alone is never the proof."""
    fields = ('qpos','qvel','ctrl','qacc_warmstart','observation','time')
    same = {}
    for field in fields:
        a,b = np.asarray(reference_initial[field]),np.asarray(new_initial[field])
        same[field] = (a.dtype==b.dtype and a.shape==b.shape and
                       a.tobytes()==b.tobytes())
    control_equal = (reference_control_reset==new_control_reset)
    return {'schema':'d1-c21-observed-reset-homotopy-v1',
            'five_initial_arrays_and_time_byte_equal':all(same.values()),
            'initial_fields_byte_equal':same,
            'complete_control_reset_state_equal':control_equal,
            'passed':all(same.values()) and control_equal,
            'meaning':'Two actually recorded resets compared; command seed alone is insufficient'}


def normalized_manifest21(manifest: dict) -> dict:
    """Keep compiled topology and numeric geometry; remove process addresses only."""
    clean = deepcopy(manifest)
    for key in ('model_address','data_address'):
        clean['model_identity'].pop(key,None)
    return clean


def compare_reset_to_template21(expected_manifest: dict, actual_manifest: dict,
                                expected_initial: dict, actual_initial: dict,
                                expected_control_reset: dict,
                                actual_control_reset: dict) -> dict:
    """Bind a newly observed reset to the sealed, source-derived expectation."""
    left,right = normalized_manifest21(expected_manifest),normalized_manifest21(actual_manifest)
    if left.keys()!=right.keys() or left['model_identity']!=right['model_identity']:
        geometry_same = False
    elif left['terrain_world_geoms']!=right['terrain_world_geoms']:
        geometry_same = False
    else:
        aa,bb = left['robot_collision_geoms'],right['robot_collision_geoms']
        geometry_same = len(aa)==len(bb) and all(
            {k:v for k,v in a.items() if k not in ('world_center_m','rbound_m')} ==
            {k:v for k,v in b.items() if k not in ('world_center_m','rbound_m')}
            and np.allclose(a['world_center_m'],b['world_center_m'],rtol=0.,atol=1e-12)
            and abs(a['rbound_m']-b['rbound_m'])<=1e-12
            for a,b in zip(aa,bb,strict=True))
        geometry_same = geometry_same and np.allclose(
            left['base_world_position_m'],right['base_world_position_m'],rtol=0.,atol=1e-12)
    homotopy = reset_homotopy21(expected_initial,actual_initial,
                                expected_control_reset,actual_control_reset)
    return {'schema':'d1-c21-expected-actual-reset-bridge-v1',
            'expected_manifest_sha256':_canonical_sha(expected_manifest),
            'actual_manifest_sha256':_canonical_sha(actual_manifest),
            'normalized_compiled_geometry_equal_atol_1e_minus_12':bool(geometry_same),
            'initial_and_control_reset':homotopy,
            'passed':bool(geometry_same and homotopy['passed']),
            'reference_kind':'source_derived_expected_reset_not_observed_future_reset'}


def source_derived_templates21(spec: dict) -> tuple[dict,dict]:
    """Translate only the robot reset geometry from an observed flat reset.

    World terrain rows remain fixed. The proof requires identical reset
    qpos posture and dynamic state in the saved A1 terrain references; no
    unseen reset is claimed to have been observed.
    """
    finite = spec['finite_geometry']
    root = Path(finite['reference_run'])
    flat_manifest = json.loads(Path(finite['flat_saved_geometry']).read_text())
    b_manifest = json.loads(Path(finite['candidate_saved_geometry']).read_text())
    references = {}
    identities = {}
    for terrain,index in finite['reference_episode_by_terrain'].items():
        stem = root/f'training_episode_{index:06d}'
        archive = stem.with_name(stem.name+'_initial_state.npz')
        with np.load(archive,allow_pickle=False) as saved:
            initial = {name:np.array(saved[name],copy=True) for name in INITIAL_FIELDS}
        control = json.loads(stem.with_name(stem.name+'_control_reset.json').read_text())
        references[terrain] = (initial,control)
        identities[terrain] = {'initial_state_path':str(archive),
                               'control_reset_path':str(stem.with_name(stem.name+'_control_reset.json'))}
    flat_initial = references['flat'][0]
    _need(normalized_manifest21(flat_manifest)['terrain_world_geoms'] ==
          normalized_manifest21(b_manifest)['terrain_world_geoms'],
          'saved C20 A0/B18 compiled world geometry differs')
    _need(normalized_manifest21(flat_manifest)['model_identity'] ==
          normalized_manifest21(b_manifest)['model_identity'],
          'saved C20 A0/B18 compiled model topology differs')
    _need(len(flat_manifest['robot_collision_geoms'])==len(b_manifest['robot_collision_geoms']),
          'saved C20 A0/B18 robot geom count differs')
    for a,b in zip(flat_manifest['robot_collision_geoms'],b_manifest['robot_collision_geoms'],strict=True):
        _need(a==b,'saved C20 A0/B18 flat reset robot geometry differs')
    templates = {}
    for terrain,(initial,_) in references.items():
        for field in ('qvel','ctrl','qacc_warmstart'):
            a,b = np.asarray(initial[field]),np.asarray(flat_initial[field])
            _need(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),
                  'observed terrain reset dynamic state differs: '+terrain+'/'+field)
        qa,qb = np.asarray(initial['qpos']),np.asarray(flat_initial['qpos'])
        _need(qa.shape==qb.shape and qa.dtype==qb.dtype and qa[3:].tobytes()==qb[3:].tobytes(),
              'observed terrain reset posture differs: '+terrain)
        delta = qa[:3]-qb[:3]
        manifest = deepcopy(flat_manifest)
        manifest['base_world_position_m'] = qa[:3].tolist()
        for geom in manifest['robot_collision_geoms']:
            geom['world_center_m']=(np.asarray(geom['world_center_m'])+delta).tolist()
        templates[terrain] = (manifest,initial)
        identities[terrain]['derived_manifest_sha256']=_canonical_sha(manifest)
        identities[terrain]['translation_m']=delta.tolist()
    identities['flat_compiled_manifest_path']=str(finite['flat_saved_geometry'])
    identities['B18_crosscheck_compiled_manifest_path']=str(finite['candidate_saved_geometry'])
    identities['basis']='source_derived_expected_reset_not_observed_future_reset'
    return templates,identities


def preflight_finite21(select_a:Callable[[int],Any],select_b:Callable[[int],Any],
                       fixed_eval_schedules:dict[str,Any],
                       template_by_terrain:dict[str,tuple[dict,dict]],
                       *, trace_sink:dict[str,np.ndarray]|None=None) -> dict:
    """Exactly A/B source0..79 plus the supplied fixed evaluation table.

    Templates are observed elsewhere but *not* an actual future reset. Their
    result is a conditional source-equivalence inference and must be replaced
    by ``qualify_reset21`` on every new runtime reset.
    """
    _need(set(template_by_terrain)==ALLOWED_TERRAINS,
          'four source-derived reset templates are required')
    _need(type(fixed_eval_schedules) is dict and len(fixed_eval_schedules)==16
          and sum(len(s.raw_commands) for s in fixed_eval_schedules.values())==24800,
          'exactly 16 fixed evaluation schedules totaling 24800 controls are required')
    expected_total = 160000+24800
    total_controls = 0
    rows = []
    for arm,selector in (('A',select_a),('B',select_b)):
        for source in range(FIRST_SOURCE,AFTER_LAST_SOURCE):
            schedule = selector(source)
            _need(schedule.terrain in ALLOWED_TERRAINS,
                  'finite source has unknown terrain')
            manifest,initial = template_by_terrain[schedule.terrain]
            path, traced = nominal_path21(schedule,trace=True)
            total_controls += len(schedule.raw_commands)
            key = f'train_{arm}_{source:06d}'
            if trace_sink is not None:
                trace_sink[key+'_path_xy_m'] = path
                trace_sink[key+'_raw_forward_yaw'] = np.asarray(traced['raw_forward_yaw'],dtype=np.float64)
                trace_sink[key+'_servo_forward_yaw'] = np.asarray(traced['servo_forward_yaw'],dtype=np.float64)
            nominal = {k:v for k,v in traced.items() if k not in ('raw_forward_yaw','servo_forward_yaw')}
            radius,_ = _reset_manifest(manifest,initial,schedule)
            bad = np.flatnonzero((np.abs(path[:,0])+radius>=SAFE_X_M)|
                                   (np.abs(path[:,1])+radius>=SAFE_Y_M))
            row = {'arm':arm,'source_episode_index':source,
                   'terrain':schedule.terrain,
                   'command_sha256':command_sha256(schedule),
                   'evidence_basis':'source_derived_expected_reset_not_observed_future_reset',
                   'actual_future_reset_verified':False,
                   'path_sha256':nominal['path_sha256'],
                   'reset_robot_radial_bound_m':radius,
                   'first_safety_bound_bad_point':int(bad[0]) if len(bad) else None,
                   'maximum_expanded_abs_x_m':float(np.max(np.abs(path[:,0]))+radius),
                   'maximum_expanded_abs_y_m':float(np.max(np.abs(path[:,1]))+radius),
                   'x_safety_margin_m':float(SAFE_X_M-np.max(np.abs(path[:,0]))-radius),
                   'y_safety_margin_m':float(SAFE_Y_M-np.max(np.abs(path[:,1]))-radius)}
            try:
                receipt,_ = _qualify_path21(schedule,manifest,initial,path,nominal,
                                            kind='train',source_index=source,
                                            evidence_basis=row['evidence_basis'])
                row['conditional_template_passed'] = True
                row['conditional_receipt'] = receipt
            except (ValueError,TypeError,KeyError) as error:
                row['conditional_template_passed'] = False
                row['reason'] = str(error)
            rows.append(row)
    eval_rows = []
    for case_id,schedule in sorted(fixed_eval_schedules.items()):
        _need(schedule.terrain in ALLOWED_TERRAINS,
              'fixed evaluation schedule has unknown terrain')
        manifest,initial = template_by_terrain[schedule.terrain]
        path,traced = nominal_path21(schedule,trace=True)
        total_controls += len(schedule.raw_commands)
        key = 'eval_'+case_id
        if trace_sink is not None:
            trace_sink[key+'_path_xy_m'] = path
            trace_sink[key+'_raw_forward_yaw'] = np.asarray(traced['raw_forward_yaw'],dtype=np.float64)
            trace_sink[key+'_servo_forward_yaw'] = np.asarray(traced['servo_forward_yaw'],dtype=np.float64)
        nominal = {k:v for k,v in traced.items() if k not in ('raw_forward_yaw','servo_forward_yaw')}
        radius,_ = _reset_manifest(manifest,initial,schedule)
        bad = np.flatnonzero((np.abs(path[:,0])+radius>=SAFE_X_M)|
                               (np.abs(path[:,1])+radius>=SAFE_Y_M))
        row = {'case_id':case_id,'terrain':schedule.terrain,
               'command_sha256':command_sha256(schedule),
               'evidence_basis':'source_derived_expected_reset_not_observed_future_reset',
               'actual_future_reset_verified':False,
               'path_sha256':nominal['path_sha256'],
               'reset_robot_radial_bound_m':radius,
               'first_safety_bound_bad_point':int(bad[0]) if len(bad) else None,
               'maximum_expanded_abs_x_m':float(np.max(np.abs(path[:,0]))+radius),
               'maximum_expanded_abs_y_m':float(np.max(np.abs(path[:,1]))+radius),
               'x_safety_margin_m':float(SAFE_X_M-np.max(np.abs(path[:,0]))-radius),
               'y_safety_margin_m':float(SAFE_Y_M-np.max(np.abs(path[:,1]))-radius)}
        try:
            receipt,_ = _qualify_path21(schedule,manifest,initial,path,nominal,
                                        kind='fixed_eval',case_id=case_id,
                                        evidence_basis=row['evidence_basis'])
            row['conditional_template_passed'] = True
            row['conditional_receipt'] = receipt
        except (ValueError,TypeError,KeyError) as error:
            row['conditional_template_passed'] = False
            row['reason'] = str(error)
        eval_rows.append(row)
    _need(total_controls==expected_total,'C21 nominal integration budget differs')
    return {'schema':'d1-c21-finite-template-preflight-v1',
            'train_sources_per_arm':80,'train_rows':rows,
            'fixed_eval_rows':eval_rows,
            'pure_servo_advance_count':total_controls,
            'all_conditional_template_checks_passed':all(
                row['conditional_template_passed'] for row in (*rows,*eval_rows)),
            'actual_future_resets_verified':False,
            'runtime_requirement':'qualify_reset21 with each actual new reset before first control',
            'model_calls':0,'physics_calls':0}
