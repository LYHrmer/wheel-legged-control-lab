"""Online C31 task/safety classification from complete saved data only.

This mirrors the frozen C30 physical gates without importing readers whose
import guards intentionally forbid the warm training/physics process. Final
qualification still belongs to the independent cold saved-data reader.
"""
from __future__ import annotations

import math

import numpy as np

from kinematics24 import reconstruct24, rotation, collision_bounds24


# Frozen C27/spec27.json original flat 30 mm gates used by C30 score_skill30.
# Only fields consumed by this online classifier are copied here. The cold
# reader independently reopens the frozen source/spec and recomputes each gate.
GATES31 = {
    'cycle_limit_s':15.,'signed_lateral_m':(.025,.05),
    'final_xy_error_lt_m':.012,'final_yaw_error_lt_rad':.12,
    'finish_margin_ge_m':.005,'retention_ge':.9,'max_rollback_le_m':.005,
    'entry_abs_roll_pitch_le_rad':.12,'entry_origin_speed_lt_mps':.04,
    'entry_angvel_norm_lt_rps':.08,'entry_foot_plane_abs_le_m':.01,
    'tail_controls':100,'tail_mean_abs_body_COM_vx_le_mps':.04,
    'tail_mean_abs_body_COM_vy_le_mps':.04,
    'tail_mean_abs_yawrate_le_rps':.05,'tail_clearance_std_le_m':.02,
    'tail_positive_native_load_fraction_each_wheel_ge':.95,
    'positive_native_load_threshold_N':1e-8,
}


def _state31(qpos, qvel, binding):
    q,v=np.asarray(qpos,dtype=float),np.asarray(qvel,dtype=float)
    r=rotation(q[3:7])
    roll=math.atan2(r[2,1],r[2,2])
    pitch=math.asin(float(-np.clip(r[2,0],-1.,1.)))
    yaw=math.atan2(r[1,0],r[0,0])
    omega=r@v[3:6]
    world_com=v[:3]+np.cross(omega,r@np.asarray(binding['base_body_ipos']))
    return {'rpy':np.array([roll,pitch,yaw]),'origin_velocity':v[:3],
            'world_angular_velocity':omega,'body_com_velocity':r.T@world_com}


def _geometry31(kin,qpos,qvel,wheel_map):
    state=reconstruct24(kin,qpos,qvel)
    points=np.zeros((4,3))
    for body,leg in wheel_map.items():
        gids=[i for i,b in enumerate(kin['geom_bodyid']) if int(b)==body]
        if not gids:
            raise ValueError('C31 compiled wheel body has no geometry')
        gid=gids[0]
        body_r=state['geom_xmat'][gid]@rotation(kin['geom_quat'][gid]).T
        body_p=state['geom_xpos'][gid]-body_r@np.asarray(kin['geom_pos'][gid])
        joint=int(kin['body_jntadr'][body])
        if kin['body_jntnum'][body]!=1 or kin['jnt_type'][joint]!=3:
            raise ValueError('C31 wheel body is not original one-hinge topology')
        axis=body_r@np.asarray(kin['jnt_axis'][joint])
        az=float(np.clip(axis[2],-1.,1.))
        point=body_p-.087*(np.array([0.,0.,1.])-axis*az)/math.sqrt(max(1e-10,1-az*az))
        sign=0. if abs(az)<.002 else np.sign(az)
        points[leg]=point-.020*sign*axis
    minimum=np.full(4,np.inf)
    for row in collision_bounds24(kin,state,wheel_map):
        if row['wheel_index'] is not None:
            j=row['wheel_index']
            minimum[j]=min(minimum[j],row['minimum_world_m'][2])
    if not np.isfinite(minimum).all():
        raise ValueError('C31 whole-wheel collision envelope is incomplete')
    return {'contact_points_m':points,'whole_com_m':state['whole_com_position'],
            'wheel_shape_min_z_m':minimum}


def _margin31(points_xy,com_xy):
    p=np.asarray(points_xy,dtype=float)
    center=p.mean(axis=0)
    p=p[np.argsort(np.arctan2(p[:,1]-center[1],p[:,0]-center[0]))]
    margins=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        edge=b-a
        norm=float(np.linalg.norm(edge))
        if norm>1e-8:
            margins.append(float(np.array([-edge[1],edge[0]])@(com_xy-a)/norm))
    return min(margins,default=-1.)


def _ground31(geometry,x,y):
    rows={int(k):v for k,v in geometry.items()}
    floor=[(gid,r) for gid,r in rows.items() if r['type']=='plane']
    if len(floor)!=1 or floor[0][1]['name']!='floor' or not np.allclose(
            floor[0][1]['position_m'],(0,0,0),atol=0,rtol=0):
        raise ValueError('C31 actual flat floor geometry differs')
    height=0.
    for row in rows.values():
        if row['type']!='box':
            continue
        center=np.asarray(row['position_m'],dtype=float)
        half=np.asarray(row['size_m'],dtype=float)
        r=rotation(row['quaternion_wxyz'])
        dx,dy=x-center[0],y-center[1]
        lower,upper,face=-math.inf,math.inf,None
        for axis in range(3):
            slope=float(r[2,axis])
            offset=float(r[0,axis]*dx+r[1,axis]*dy)
            if abs(slope)<=1e-14:
                if abs(offset)>half[axis]:
                    face=None
                    break
                continue
            a,b=((-half[axis]-offset)/slope+center[2],
                 (half[axis]-offset)/slope+center[2])
            lower=max(lower,min(a,b))
            if max(a,b)<upper:
                upper=max(a,b)
                face=math.copysign(1.,slope)*r[:,axis]
        if face is not None and lower<=upper and upper>height:
            if face[2]<=0:
                raise ValueError('C31 compiled terrain vertical hit is not upward')
            height=upper
    return height


def _native_loads31(native,threshold):
    loads=np.zeros((len(native),4),dtype=bool)
    contacts=np.zeros_like(loads)
    for i,item in enumerate(native):
        for contact in item['contacts']:
            leg=contact['robot_wheel_index']
            if leg is not None and contact['terrain_family'] is not None:
                supported=contact['efc_address']>=0
                contacts[i,leg]|=supported
                loads[i,leg]|=supported and contact['normal_load_on_robot_n']>threshold
    return loads,contacts


def _tail31(rows,states,loads,begin,end,binding,geometry,gates):
    if (begin<0 or end>len(rows) or end-begin!=100 or
            gates['tail_controls']!=100):
        return {'passed':False,'complete_window':False,'window':[begin,end]}
    zero=all(all(row['info'][key][field]==0. for key in
        ('raw_operator_command','consumed_command') for field in
        ('forward_velocity_mps','lateral_velocity_mps','yaw_rate_rps'))
        for row in rows[begin:end])
    velocities=np.asarray([_state31(states['qpos'][i+1],states['qvel'][i+1],
        binding)['body_com_velocity'] for i in range(begin,end)])
    yaw=np.abs(states['qvel'][begin+1:end+1,5])
    clearance=np.asarray([q[2]-_ground31(geometry,q[0],q[1])
        for q in states['qpos'][begin+1:end+1]])
    fractions=loads[5*begin:5*end].mean(axis=0)
    checks={
        'raw_and_servo_zero':zero,
        'body_vx':bool(np.mean(np.abs(velocities[:,0]))<=gates['tail_mean_abs_body_COM_vx_le_mps']),
        'body_vy':bool(np.mean(np.abs(velocities[:,1]))<=gates['tail_mean_abs_body_COM_vy_le_mps']),
        'yaw_rate':bool(yaw.mean()<=gates['tail_mean_abs_yawrate_le_rps']),
        'clearance_std':bool(clearance.std()<=gates['tail_clearance_std_le_m']),
        'positive_loads':bool(np.all(fractions>=gates['tail_positive_native_load_fraction_each_wheel_ge']))}
    return {'passed':all(checks.values()),'complete_window':True,
            'window':[begin,end],'checks':checks,
            'each_wheel_positive_load_fraction':fractions.tolist()}


def score_task31(rows,states,native,*,binding,geometry,gates,direction,
                 global_control_offset,cancel_expected=False,cancel_index=None):
    """Classify success vs controlled metric failure after physical closure."""
    n=len(rows)
    q=np.asarray(states['qpos'],dtype=float)
    v=np.asarray(states['qvel'],dtype=float)
    if (direction not in (-1,1) or q.shape!=(n+1,23) or v.shape!=(n+1,22)
            or len(native)!=5*n or
            [r['native_index'] for r in native]
               != list(range(5*global_control_offset,5*(global_control_offset+n)))
            or [r['control_index'] for r in rows]!=list(range(n))):
        raise ValueError('C31 online task lacks complete actual local/global intervals')
    wheel_map={int(k):int(j) for k,j in binding['wheel_index_by_body_id'].items()}
    kin=binding['kinematics24']
    side=[i for i,row in enumerate(rows) if row['actor']=='traditional_side']
    if not side or side!=list(range(side[0],side[-1]+1)):
        raise ValueError('C31 task has no single contiguous admitted side interval')
    start,end=side[0],side[-1]+1
    first=rows[start]['info']['controller_record']
    last=rows[end-1]['info']['controller_record']
    finish=last['diagnostic_after']
    if ([i for i in side if rows[i]['info']['controller_record']['accepted_start']]!=[start]
            or first['intent_direction']!=direction):
        raise ValueError('C31 actual side start/intent differs')
    initial=_state31(q[start],v[start],binding)
    yaw0=initial['rpy'][2]
    forward=np.array([math.cos(yaw0),math.sin(yaw0)])
    left=np.array([-math.sin(yaw0),math.cos(yaw0)])
    delta=q[:,:2]-q[start,:2]
    signed=direction*(delta@left)
    target=q[start,:2]+direction*.03*left
    before=first['diagnostic_before']
    if (not np.allclose(before['initial_pose'],q[start,:3],atol=2e-9,rtol=0)
            or not np.allclose(before['delta'],
                               np.r_[direction*.03*left,0.],atol=2e-9,rtol=0)
            or abs(math.atan2(math.sin(before['start_yaw_rad']-yaw0),
                               math.cos(before['start_yaw_rad']-yaw0)))>2e-9):
        raise ValueError('C31 side start/goal differs from actual 30 mm pose')
    loads,contacts=_native_loads31(native,gates['positive_native_load_threshold_N'])
    beginning=_geometry31(kin,q[start],v[start],wheel_map)
    ending=_geometry31(kin,q[end],v[end],wheel_map)
    margin=_margin31(ending['contact_points_m'][:,:2],ending['whole_com_m'][:2])
    entry={
        'exact_200_preparation':start==200,
        'roll_pitch':bool(np.max(np.abs(initial['rpy'][:2]))<=gates['entry_abs_roll_pitch_le_rad']),
        'origin_speed':bool(np.linalg.norm(initial['origin_velocity'])<gates['entry_origin_speed_lt_mps']),
        'angular_speed':bool(np.linalg.norm(initial['world_angular_velocity'])<gates['entry_angvel_norm_lt_rps']),
        'foot_plane':bool(np.max(np.abs(beginning['contact_points_m'][:,2]))<=gates['entry_foot_plane_abs_le_m']),
        'four_actual_contacts':bool(start>0 and contacts[5*start-1].all())}
    liftoff,touchdown=[False]*4,[False]*4
    for i in side:
        shaped=_geometry31(kin,q[i+1],v[i+1],wheel_map)
        for wheel in range(4):
            touchdown[wheel]|=bool(liftoff[wheel] and loads[5*i:5*i+5,wheel].any())
            liftoff[wheel]|=bool(shaped['contact_points_m'][wheel,2]>.012
                and shaped['wheel_shape_min_z_m'][wheel]>0.
                and not contacts[5*i+4,wheel])
    leg_dofs=np.asarray(binding['joint_dof_addresses']).reshape(4,4)[:,:3].ravel()
    side_safe=all(abs(r['roll_deg'])<=math.degrees(.32)
        and abs(r['pitch_deg'])<=math.degrees(.32)
        and r['after']['qpos'][2]>=.32 and r['nonwheel_contact_count']==0
        and np.max(np.abs(np.asarray(r['after']['qvel'])[leg_dofs]))<=18.
        for r in native[5*start:5*end])
    rolling_safe=all(abs(r['roll_deg'])<=10 and abs(r['pitch_deg'])<=10
        and r['after']['qpos'][2]-_ground31(geometry,*r['after']['qpos'][:2])>=.28
        and r['nonwheel_contact_count']==0 and abs(r['after']['qpos'][0])<=11
        and abs(r['after']['qpos'][1])<=6
        for i,row in enumerate(rows) if row['actor']!='traditional_side'
        for r in native[5*i:5*i+5])
    complete=bool(finish['done'] and finish['leg'] is None
                  and last['handoff_next_preview'])
    landed=bool(complete and contacts[5*end-1].all() and loads[5*end-1].all()
                and margin>=gates['finish_margin_ge_m'])
    position_error=float(np.linalg.norm(q[end,:2]-target))
    yaw_end=_state31(q[end],v[end],binding)['rpy'][2]
    yaw_error=math.atan2(math.sin(yaw_end-yaw0),math.cos(yaw_end-yaw0))
    cycle_success=bool(finish['failure'] is None and finish['status']['success']
        and gates['signed_lateral_m'][0]<=signed[end]<=gates['signed_lateral_m'][1]
        and position_error<gates['final_xy_error_lt_m']
        and abs(yaw_error)<gates['final_yaw_error_lt_rad']
        and all(liftoff) and all(touchdown)
        and (end-start)*.01<=gates['cycle_limit_s'])
    observation_end=end+400
    exact_retention=complete and observation_end==n
    retention=(float(signed[observation_end]/signed[end])
               if observation_end<=n and signed[end]!=0 else None)
    rollback=(float(signed[end]-signed[end:observation_end+1].min())
              if observation_end<=n else None)
    retained=bool(retention is not None and retention>=gates['retention_ge']
                  and rollback<=gates['max_rollback_le_m'])
    stopped=_tail31(rows,states,loads,observation_end-100,observation_end,
                    binding,geometry,gates)
    zero_commands=all(all(row['info'][key][field]==0. for key in
        ('raw_operator_command','consumed_command') for field in
        ('forward_velocity_mps','lateral_velocity_mps','yaw_rate_rps'))
        for row in rows)
    resumed=bool(exact_retention and all(row['actor']=='B'
        and row['policy_predict_called'] is True
        for row in rows[end:observation_end]))
    mandatory={'entry':all(entry.values()),'side_physical_safety':bool(side_safe),
        'rolling_physical_safety':bool(rolling_safe),
        'no_termination_or_truncation':not any(r['terminated'] or r['truncated'] for r in rows),
        'zero_raw_and_servo_motion':zero_commands,
        'one_side_within_1500':end-start<=1500,
        'safe_landed_handoff':landed,
        'exact_400_retention':bool(exact_retention),
        'tail_stopped_loaded':stopped['passed'],
        'B22_predict_restored':resumed}
    task_metrics={'lateral_goal_and_touchdown':cycle_success,
                  'retention_ratio_and_rollback':retained}
    safe=all(mandatory.values())
    cancel_ok=bool(cancel_expected and cancel_index is not None
        and any(rows[i]['info']['controller_record']['cancel_requested'] for i in side)
        and finish['failure']=='cancelled' and not finish['status']['success']
        and 0<end-cancel_index<=300)
    success=safe and not cancel_expected and all(task_metrics.values())
    return {'schema':'d1-c31-online-task-v1',
            'online_success':bool(success),
            'online_safe_cancel':bool(safe and cancel_ok),
            'online_controlled_failure_eligible':bool(safe and not success and not cancel_expected),
            'safety_complete':bool(safe),'mandatory_gates':mandatory,
            'task_metric_gates':task_metrics,'entry_gates':entry,
            'side_control_window':[start,end],
            'retention_control_window':[end,observation_end],
            'signed_lateral_m':float(signed[end]),
            'final_xy_error_m':position_error,
            'final_yaw_error_rad':yaw_error,
            'finish_support_margin_m':float(margin),
            'side_duration_s':(end-start)*.01,
            'cancel_to_handoff_controls':None if cancel_index is None else end-cancel_index,
            'retention_ratio':retention,'max_rollback_m':rollback,
            'each_wheel_liftoff':liftoff,
            'each_wheel_loaded_touchdown':touchdown,
            'tail_stop':stopped,
            'independent_qualification_pending':True}
