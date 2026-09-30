"""Independent compiled-tree arithmetic: no engine import, solve, or integration."""
from __future__ import annotations
import math
import numpy as np

FIELDS = ('body_parentid', 'body_pos', 'body_quat', 'body_ipos', 'body_mass',
          'body_jntadr', 'body_jntnum', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr',
          'jnt_axis', 'jnt_pos', 'qpos0', 'geom_bodyid', 'geom_pos', 'geom_quat',
          'geom_type', 'geom_size', 'geom_margin', 'geom_contype', 'geom_conaffinity')


def capture_binding24(plant):
    """Read compiled arrays only; called once inside counted worker."""
    from run_world_upright_reference_11 import _binding
    value = _binding(plant)
    value['kinematics24'] = {key: np.asarray(getattr(plant.model, key)).tolist()
                             for key in FIELDS}
    value['dynamics24'] = {key: np.asarray(getattr(plant.model,key)).tolist() for key in
        ('body_inertia','body_iquat','dof_damping','dof_armature','dof_frictionloss',
         'actuator_ctrlrange','actuator_forcerange','geom_friction','geom_condim')}
    value['solver24'] = {key: float(getattr(plant.model.opt,key)) for key in
        ('timestep','integrator','solver','iterations','ls_iterations')}
    return value


def robot_binding_bridge24(actual, reference):
    """Direct old saved fields; dynamics not previously saved use source proof."""
    keys=('base_body_id','base_body_ipos','wheel_index_by_body_id','joint_ids',
          'joint_qpos_addresses','joint_dof_addresses','actuator_ids',
          'actuator_trntype','actuator_trnid','actuator_gear','body_mass','nominal_total_mass_kg')
    checked=[]
    for key in keys:
        # JSON numeric identities preserve int-key maps only after serialization.
        if key=='wheel_index_by_body_id':
            same={int(k):v for k,v in actual[key].items()}=={int(k):v for k,v in reference[key].items()}
        else:
            same=np.array_equal(np.asarray(actual[key]),np.asarray(reference[key]))
        if not same:
            raise ValueError('C24 robot differs from actual saved C18/C22 compiled binding: '+key)
        checked.append(key)
    for key in ('geom_bodyid','geom_type','geom_size','geom_contype','geom_conaffinity'):
        a=np.asarray(actual[key])[np.asarray(actual['geom_bodyid'])>0]
        b=np.asarray(reference[key])[np.asarray(reference['geom_bodyid'])>0]
        if not np.array_equal(a,b):
            raise ValueError('C24 robot collision/visual primitive topology differs: '+key)
        checked.append(key+'[body_id>0]')
    s=actual['solver24']
    if s!={'timestep':.002,'integrator':3.,'solver':2.,'iterations':20.,'ls_iterations':5.}:
        raise ValueError('C24 actual compiled solver options differ from frozen shared builder')
    d=actual['dynamics24']
    dof=np.asarray(actual['joint_dof_addresses'],dtype=int)
    for key,target in (('dof_damping',.1),('dof_armature',.01),('dof_frictionloss',.2)):
        if not np.all(np.asarray(d[key])[dof]==target):
            raise ValueError('C24 actual compiled joint dynamics differs from shared source')
    collision=(np.asarray(actual['geom_bodyid'])>0)&(
        (np.asarray(actual['geom_contype'])!=0)|(np.asarray(actual['geom_conaffinity'])!=0))
    if (not np.all(np.asarray(d['geom_condim'])[collision]==3)
            or not np.all(np.asarray(d['geom_friction'])[collision]==[.9,.005,.0001])):
        raise ValueError('C24 compiled robot contact dynamics differs from shared source')
    return dict(passed=True,direct_saved_compiled_fields=checked,
        source_derived_fields=['body_inertia_from_same_frozen_URDF',
            'solver_and_joint_dynamics_explicit_same_frozen_builder_settings',
            'robot_contact_friction_and_condim_same_source'],
        no_reference_model_constructed=True)


def rotation(q):
    w,x,y,z = np.asarray(q, dtype=float)
    if abs(w*w+x*x+y*y+z*z-1.) > 1e-6:
        raise ValueError('nonunit kinematic quaternion')
    return np.array(((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
                     (2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),
                     (2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y))))


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    if abs(float(axis@axis)-1.) > 1e-10:
        raise ValueError('nonunit compiled hinge axis')
    x,y,z = axis
    cross = np.array(((0.,-z,y),(z,0.,-x),(-y,x,0.)))
    return np.eye(3)*math.cos(angle)+(1-math.cos(angle))*np.outer(axis,axis)+math.sin(angle)*cross


def reconstruct24(saved, qpos, qvel):
    """D1 free base + zero/one hinge per body; all compiled inertial masses."""
    m = {key: np.asarray(saved[key]) for key in FIELDS}
    q,v = np.asarray(qpos,dtype=float), np.asarray(qvel,dtype=float)
    if q.shape != (23,) or v.shape != (22,) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('C24 fixed D1 state dimensions/finite identity differ')
    n = len(m['body_parentid'])
    p, vel, omega = (np.zeros((n,3)) for _ in range(3))
    rot = np.repeat(np.eye(3)[None], n, axis=0)
    free = 0
    for body in range(1,n):
        parent = int(m['body_parentid'][body])
        if not 0 <= parent < body:
            raise ValueError('compiled tree is not topologically ordered')
        p[body] = p[parent] + rot[parent]@m['body_pos'][body]
        rot[body] = rot[parent]@rotation(m['body_quat'][body])
        omega[body] = omega[parent]
        vel[body] = vel[parent]+np.cross(omega[parent],p[body]-p[parent])
        count, first = int(m['body_jntnum'][body]), int(m['body_jntadr'][body])
        if count == 0:
            continue
        if count != 1:
            raise ValueError('C24 only qualified zero/one-joint bodies')
        j, kind = first, int(m['jnt_type'][first])
        qa, da = int(m['jnt_qposadr'][j]), int(m['jnt_dofadr'][j])
        if kind == 0:
            if parent != 0 or qa != 0 or da != 0:
                raise ValueError('free base binding differs')
            free += 1
            p[body], rot[body], vel[body] = q[:3], rotation(q[3:7]), v[:3]
            omega[body] = rot[body]@v[3:6]
        elif kind == 3:
            local_anchor = m['jnt_pos'][j]
            anchor = p[body]+rot[body]@local_anchor
            anchor_v = vel[body]+np.cross(omega[body],rot[body]@local_anchor)
            world_axis = rot[body]@m['jnt_axis'][j]
            rot[body] = rot[body]@axis_rotation(m['jnt_axis'][j],q[qa]-m['qpos0'][qa])
            omega[body] = omega[body]+world_axis*v[da]
            p[body] = anchor-rot[body]@local_anchor
            vel[body] = anchor_v-np.cross(omega[body],rot[body]@local_anchor)
        else:
            raise ValueError('unqualified compiled joint kind')
    if free != 1:
        raise ValueError('C24 requires exactly one compiled free base')
    ipos = p+np.einsum('nij,nj->ni',rot,m['body_ipos'])
    comvel = vel+np.cross(omega,ipos-p)
    mass = m['body_mass'].astype(float)
    if mass[0] != 0 or np.any(mass < 0) or not np.isfinite(mass).all() or mass.sum() <= 0:
        raise ValueError('invalid whole-system mass binding')
    geom_body = m['geom_bodyid'].astype(int)
    geom_pos = p[geom_body]+np.einsum('nij,nj->ni',rot[geom_body],m['geom_pos'])
    geom_rot = np.stack([rot[b]@rotation(qg) for b,qg in zip(geom_body,m['geom_quat'])])
    return dict(body_ipos=ipos, body_com_velocity=comvel, geom_xpos=geom_pos,
        geom_xmat=geom_rot, whole_com_position=np.sum(mass[:,None]*ipos,axis=0)/mass.sum(),
        whole_com_velocity=np.sum(mass[:,None]*comvel,axis=0)/mass.sum())


def collision_bounds24(saved, state, wheel_map):
    rows = []
    for gid, body in enumerate(saved['geom_bodyid']):
        if body == 0 or not (saved['geom_contype'][gid] or saved['geom_conaffinity'][gid]):
            continue
        kind = saved['geom_type'][gid]
        size = np.asarray(saved['geom_size'][gid],dtype=float)
        r = state['geom_xmat'][gid]
        center = state['geom_xpos'][gid]
        if kind == 2:
            extent = np.full(3,size[0])
        elif kind == 3:
            extent = size[0]+size[1]*np.abs(r[:,2])
        elif kind == 4:
            extent = np.sqrt(np.sum((r*size[None,:])**2,axis=1))
        elif kind == 5:
            extent = size[1]*np.abs(r[:,2])+size[0]*np.sqrt(r[:,0]**2+r[:,1]**2)
        elif kind == 6:
            extent = np.abs(r)@size
        else:
            raise ValueError('unsupported actual collision primitive')
        rows.append(dict(geom_id=gid, body_id=body, geom_type=kind,
            wheel_index=wheel_map.get(body), margin_m=saved['geom_margin'][gid],
            minimum_world_m=(center-extent).tolist(), maximum_world_m=(center+extent).tolist()))
    if {r['wheel_index'] for r in rows if r['wheel_index'] is not None} != set(range(4)):
        raise ValueError('actual four wheel collision geometries missing')
    return rows


def endpoint24(plant, binding, tick):
    """Cross-check pure reconstruction against actual synchronized native caches.

    mj_objectVelocity is read-only; calls are explicitly recorded separately
    from native integration. No new model/data, forward, or kinematics call.
    """
    import mujoco
    d, live = plant.measurement_data, plant.data
    if (not np.array_equal(d.qpos,live.qpos) or not np.array_equal(d.qvel,live.qvel)
            or float(d.time) != float(live.time)):
        raise RuntimeError('endpoint cache not synchronized with integrated state')
    result = reconstruct24(binding['kinematics24'],d.qpos,d.qvel)
    for key,actual in (('body_ipos',d.xipos),('geom_xpos',d.geom_xpos),
                       ('geom_xmat',d.geom_xmat.reshape((-1,3,3)))):
        if not np.allclose(result[key],actual,rtol=0,atol=2e-10):
            raise RuntimeError('independent endpoint kinematics differs: '+key)
    masses = np.asarray(binding['kinematics24']['body_mass'])
    velocities = np.zeros((len(masses),3))
    calls = 0
    for body in range(1,len(masses)):
        value = np.empty(6)
        mujoco.mj_objectVelocity(plant.model,d,mujoco.mjtObj.mjOBJ_BODY,body,value,0)
        calls += 1
        velocities[body] = value[3:]
    whole = (masses[:,None]*velocities).sum(axis=0)/masses.sum()
    if not np.allclose(whole,result['whole_com_velocity'],rtol=0,atol=2e-9):
        raise RuntimeError('independent whole-system COM velocity differs from actual native body velocities')
    return dict(tick=tick,time_s=float(d.time),
        cache_source='plant.measurement_data_post_refresh_measurements',
        qpos=np.asarray(d.qpos).tolist(),qvel=np.asarray(d.qvel).tolist(),
        whole_com_position_m=result['whole_com_position'].tolist(),
        whole_com_velocity_mps=whole.tolist(),
        readonly_mj_objectVelocity_calls=calls,
        collision_bounds=collision_bounds24(binding['kinematics24'],result,
            {int(k):v for k,v in binding['wheel_index_by_body_id'].items()}))
