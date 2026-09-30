"""Pure analytical checks only; root owns execution, no model construction."""
from copy import deepcopy
import math
import numpy as np
import pytest
from geometry24 import (make_schedule, raw_rows24, nominal_path24,
                        reset_geometry_gate24, SPAWN)
from kinematics24 import reconstruct24, collision_bounds24


def synthetic():
    n=7
    m=dict(body_parentid=[0,0,1,2,2,2,2],
        body_pos=[[0.,0.,0.],[0.,0.,0.],[1.,0.,0.]]+[[0.,0.,0.]]*4,
        body_quat=[[1.,0.,0.,0.]]*n,
        body_ipos=[[0.,0.,0.],[.1,0.,0.],[.1,0.,0.]]+[[0.,0.,0.]]*4,
        body_mass=[0.,2.,1.,0.,0.,0.,0.],body_jntadr=[-1,0,1,-1,-1,-1,-1],
        body_jntnum=[0,1,1,0,0,0,0],jnt_type=[0,3],jnt_qposadr=[0,7],
        jnt_dofadr=[0,6],jnt_axis=[[0.,0.,1.]]*2,jnt_pos=[[0.,0.,0.],[.2,0.,0.]],
        qpos0=[0.]*23,geom_bodyid=[1,2,3,4,5,6],
        geom_pos=[[0.,0.,0.]]*6,geom_quat=[[1.,0.,0.,0.]]*6,
        geom_type=[2,6,5,5,5,5],geom_size=[[.1,0.,0.],[.2,.1,.05]]+[[.087,.02,0.]]*4,
        geom_margin=[.001]*6,geom_contype=[1]*6,geom_conaffinity=[1]*6)
    q=np.zeros(23);q[:3]=[1.,2.,3.];q[3:7]=[math.sqrt(.5),0.,0.,math.sqrt(.5)];q[7]=math.pi/2
    v=np.zeros(22);v[:3]=[.3,.4,.5];v[5]=1.
    return m,q,v


@pytest.mark.parametrize('joint_speed',[2.,-2.])
def test_rotated_freebase_offcenter_hinge_and_all_mass_com(joint_speed):
    m,q,v=synthetic();v[6]=joint_speed
    r=reconstruct24(m,q,v)
    np.testing.assert_allclose(r['body_ipos'][1],[1.,2.1,3.],atol=1e-12)
    np.testing.assert_allclose(r['body_ipos'][2],[1.1,3.2,3.],atol=1e-12)
    np.testing.assert_allclose(r['body_com_velocity'][1],[.2,.4,.5],atol=1e-12)
    child_v=[-.9,.4+.1*(1.+joint_speed),.5]
    np.testing.assert_allclose(r['body_com_velocity'][2],child_v,atol=1e-12)
    np.testing.assert_allclose(r['whole_com_velocity'],
        [(2*.2-.9)/3,(2*.4+child_v[1])/3,.5],atol=1e-12)
    rows=collision_bounds24(m,r,{3:0,4:1,5:2,6:3})
    np.testing.assert_allclose(rows[0]['minimum_world_m'],[.9,1.9,2.9],atol=1e-12)
    np.testing.assert_allclose(rows[1]['minimum_world_m'],[1.,3.1,2.95],atol=1e-12)
    np.testing.assert_allclose(rows[2]['minimum_world_m'],[1.113,3.113,2.98],atol=1e-12)
    assert {r['wheel_index'] for r in rows if r['wheel_index'] is not None}==set(range(4))


def test_whole_com_vertical_rotation_is_not_base_velocity():
    m,q,v=synthetic()
    q[3:7]=[1.,0.,0.,0.];q[7]=0.
    v[:]=0.;v[4]=1.;v[6]=0.
    # Base COM x=.1, child COM x=1.1: world pitch makes vz=-x.
    r=reconstruct24(m,q,v)
    np.testing.assert_allclose(r['whole_com_velocity'],[0.,0.,-(2*.1+1.1)/3],atol=1e-12)
    assert not np.isclose(r['whole_com_velocity'][2],r['body_com_velocity'][1,2])


def admission_fixture():
    manifest=dict(control_dt_s=.01,native_dt_s=.002,world_collision_geoms=[
        dict(geom_id=0,name='floor',body_id=0,collision=True,type='plane',
             position_m=[0.,0.,0.],size_m=[6.,3.,.1],quaternion_wxyz=[1.,0.,0.,0.]),
        dict(geom_id=1,name='terrain_single_15mm_box',body_id=0,collision=True,type='box',
             position_m=[-3.1,0.,.0075],size_m=[.18,.62,.0075],quaternion_wxyz=[1.,0.,0.,0.])])
    e=dict(tick=0,time_s=0.,qpos=[*SPAWN,1.,0.,0.,0.]+[0.]*16,qvel=[0.]*22,
        collision_bounds=[dict(minimum_world_m=[-4.26,-.3,.02],maximum_world_m=[-3.34,.3,.6],margin_m=.001)])
    from geometry24 import FRONT,FAR,BOX_CENTER
    samples=[dict(x_m=x,y_m=0.,height_m=h,geom_name='terrain_single_15mm_box' if h else 'floor')
             for x,h in ((SPAWN[0],0.),(BOX_CENTER[0],.015),(FRONT-.01,0.),(FAR+.01,0.))]
    return e,manifest,samples


def test_fixed_schedule_and_nominal_space():
    s=make_schedule();rows=raw_rows24()
    assert len(rows)==s.control_cap==1200
    assert all(r['forward_velocity_mps']==(.2 if 200<=i<1000 else 0.) for i,r in enumerate(rows))
    path=nominal_path24()
    assert path[1000]==pytest.approx(-2.239)
    assert path[1100]==pytest.approx(-2.2)
    assert reset_geometry_gate24(*admission_fixture())['passed'] is True


@pytest.mark.parametrize('tamper',['height','initial_overlap','wrong_oracle','moving_initial'])
def test_reset_geometry_rejects_misbound_scene(tamper):
    e,m,g=deepcopy(admission_fixture())
    if tamper=='height':m['world_collision_geoms'][1]['size_m'][2]=.005
    elif tamper=='initial_overlap':e['collision_bounds'][0]['maximum_world_m'][0]=-3.27
    elif tamper=='wrong_oracle':g[1]['height_m']=0.
    else:e['qvel'][0]=.01
    with pytest.raises(ValueError):reset_geometry_gate24(e,m,g)


def test_unsupported_joint_or_missing_wheel_is_rejected():
    m,q,v=synthetic();m['jnt_type'][1]=2
    with pytest.raises(ValueError):reconstruct24(m,q,v)
    m,q,v=synthetic();r=reconstruct24(m,q,v)
    with pytest.raises(ValueError):collision_bounds24(m,r,{3:0,4:1,5:2})


def test_reader_import_is_physics_free():
    import read24
    import sys
    assert callable(read24.read_run)
    forbidden={'mujoco','torch','stable_baselines3','gym','gymnasium','glfw'}
    assert not any(name.split('.')[0] in forbidden for name in sys.modules)
