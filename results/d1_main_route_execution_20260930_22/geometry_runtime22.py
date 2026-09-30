"""Capture one already completed reset's compiled geometry without advancing it.

The measurement-data geom positions are part of the existing reset path. This
module never constructs, forwards, or steps a model. Its caller must compare
the runtime counters before and after capture and qualify the snapshot before
the first control.
"""
from __future__ import annotations

import numpy as np
import hashlib
import json
from pathlib import Path


def _identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def load_sealed_templates22(spec: dict) -> tuple[dict,dict,dict]:
    """Load C21's saved four templates, never derive replacement templates."""
    finite = spec['finite_geometry']
    report_path = Path(finite['prior_preflight_path'])
    if _identity(report_path)!=finite['prior_preflight_identity']:
        raise ValueError('C21 finite preflight identity changed')
    report = json.loads(report_path.read_text())
    folder = Path(finite['expected_templates_directory'])
    if folder!=report_path.parent or report.get('pure_servo_advance_count')!=184800:
        raise ValueError('C21 finite expected template location/budget differs')
    templates,controls = {},{}
    inputs = {str(report_path):_identity(report_path)}
    for terrain in ('flat','bumps','rough','ramp'):
        paths = [folder/f'expected_{terrain}_geometry21.json',
                 folder/f'expected_{terrain}_initial_state21.npz',
                 folder/f'expected_{terrain}_control_reset21.json']
        for path in paths:
            if report['inputs'].get(str(path))!=_identity(path):
                raise ValueError('C21 expected reset archive differs: '+str(path))
            inputs[str(path)] = _identity(path)
        geometry = json.loads(paths[0].read_text())
        with np.load(paths[1],allow_pickle=False) as saved:
            initial = {field:np.array(saved[field],copy=True) for field in
                       ('qpos','qvel','ctrl','qacc_warmstart','observation','time')}
        templates[terrain]=(geometry,initial)
        controls[terrain]=json.loads(paths[2].read_text())
    return templates,controls,inputs


def capture_actual_reset_geometry22(env) -> dict:
    plant = env.unwrapped.plant
    model,live,data = plant.model,plant.data,plant.measurement_data
    if data is live or any(
        not np.array_equal(np.asarray(getattr(data,field)),
                           np.asarray(getattr(live,field)))
        for field in ('qpos','qvel','ctrl')
    ) or float(data.time)!=float(live.time):
        raise RuntimeError('reset measurement and live state differ before geometry capture')
    return {
        'base_world_position_m':data.qpos[:3].copy().tolist(),
        'model_identity':{
            'model_address':int(model._address),
            'data_address':int(data._address),
            'source':'actual_compiled_course_reset_geometry_v1',
            'model_ngeom':int(model.ngeom),
        },
        'robot_collision_geoms':[
            {'geom_id':gid,'body_id':int(model.geom_bodyid[gid]),
             'world_center_m':data.geom_xpos[gid].copy().tolist(),
             'rbound_m':float(model.geom_rbound[gid]),'collision':True}
            for gid in range(model.ngeom)
            if model.geom_bodyid[gid]!=0
            and (model.geom_contype[gid] or model.geom_conaffinity[gid])
        ],
        'terrain_world_geoms':plant.collision_terrain_metadata['world_collision_geoms'],
    }
