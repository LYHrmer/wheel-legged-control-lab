"""Pure admission checks: no MuJoCo, policy, construction, reset or native step."""
from types import SimpleNamespace

import numpy as np
import pytest

from native_guard24 import SingleStepNativeGuard24


def plant():
    model = SimpleNamespace(ngeom=3, _address=11,
        geom_bodyid=np.array([0,0,1]), geom_type=np.array([0,6,5]),
        geom_contype=np.array([1,1,1]), geom_conaffinity=np.array([1,1,1]),
        geom_pos=np.array([[0.,0.,0.],[-3.1,0.,.0075],[0.,0.,0.]]),
        geom_size=np.array([[0.,0.,.1],[.18,.62,.0075],[.1,.1,.1]]),
        geom_quat=np.array([[1.,0.,0.,0.]]*3),opt=SimpleNamespace(timestep=.002))
    rows = [{'geom_id':i,'name':name,'body_id':0,'collision':True,'type':kind,
             'position_m':model.geom_pos[i].tolist(),'size_m':model.geom_size[i].tolist(),
             'quaternion_wxyz':model.geom_quat[i].tolist()}
            for i,name,kind in ((0,'floor','plane'),(1,'terrain_single_15mm_box','box'))]
    return SimpleNamespace(model=model,data=SimpleNamespace(_address=12),
         collision_terrain_metadata={'world_collision_geoms':rows},
         terrain_geom_ids={0,1},physics_steps=5,control_dt=.01)


def guard():
    # Bind is intentionally tested alone, without initializing a runtime/writer.
    value = SingleStepNativeGuard24.__new__(SingleStepNativeGuard24)
    value.plant = None
    value.attempted = 0
    return value


def test_exact_two_native_geoms_admitted_once():
    candidate = plant()
    check = guard()
    check.bind(candidate)
    assert check._terrain == {0:('floor','floor'),1:('terrain_single_15mm_box','step')}
    assert check._identity == (11,12)
    with pytest.raises(RuntimeError,match='only once'):
        check.bind(candidate)


@pytest.mark.parametrize('fault',['height','fake_type','hidden_world','stale_metadata','cadence','fractional_steps','control_dt'])
def test_nonmatching_native_scene_rejected(fault):
    candidate = plant()
    if fault == 'height':
        candidate.model.geom_size[1,2] = .015
        candidate.collision_terrain_metadata['world_collision_geoms'][1]['size_m'][2] = .015
    elif fault == 'fake_type':
        candidate.model.geom_type[1] = 5
    elif fault == 'hidden_world':
        candidate.model.geom_bodyid[2] = 0
    elif fault == 'stale_metadata':
        candidate.model.geom_pos[1,0] += .1
    elif fault == 'cadence':
        candidate.physics_steps = 4
    elif fault == 'fractional_steps':
        candidate.physics_steps = 5.5
    else:
        candidate.control_dt = .02
    with pytest.raises(ValueError):
        guard().bind(candidate)
