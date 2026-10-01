"""Versioned headless lateral reference skill over the one C27 torque owner.

No model or MjData is made by importing this module. make_side30 constructs
exactly one existing audited teacher controller and upgrades that same object.
The zero arm calls its original methods and never evaluates a new reference.
"""
from __future__ import annotations

import math
import numpy as np

from kinematics24 import collision_bounds24

SKILL_SCHEMA30 = 'd1-c30-lateral-reference-skill-v1'
_ACTIONS = frozenset(('fixed_nonzero_left','fixed_nonzero_right',
                      'zero_left','zero_right','cancel_left','cancel_right'))


def _curve(time_s, duration_s):
    u = min(1.0, max(0.0, float(time_s)/float(duration_s)))
    if u <= 0.0:
        return 0.0, 0.0, 0.0
    if u >= 1.0:
        return 1.0, 0.0, 0.0
    return (10*u**3-15*u**4+6*u**5,
            (30*u*u-60*u**3+30*u**4)/duration_s,
            (60*u-180*u*u+120*u**3)/(duration_s*duration_s))


def make_side30(plant, arm, geometry_binding, control_index_provider):
    """Return one C27-compatible side instance with no second scratch/data."""
    from hybrid_adapter27 import make_fast_side27
    if arm not in _ACTIONS or not callable(control_index_provider):
        raise ValueError('C30 arm and control index provider must be fixed')
    if not isinstance(geometry_binding,dict) or 'kinematics24' not in geometry_binding:
        raise ValueError('C30 needs the actual captured compiled geometry binding')
    wheel_map={int(key):int(value) for key,value in
               geometry_binding['wheel_index_by_body_id'].items()}
    if set(wheel_map.values())!=set(range(4)):
        raise ValueError('C30 actual compiled wheel body map differs')
    side=make_fast_side27(plant,'teacher')

    class SkilledFast30(type(side)):
        def reset(self):
            super().reset()
            self._skill30_events=[]
            self._skill30_latches=0
            self._skill30_cancel_requested=False
            self._skill30_cancel_at=None
            self._skill30_horizontal_started=False
            self._skill30_horizontal_start_control=None
            self._skill30_horizontal_elapsed=0.0
            self._skill30_horizontal_commanded_last=False
            self._skill30_last_gate={}
            self._skill30_foot_velocity=np.zeros(3)
            self._skill30_foot_accel=np.zeros(3)
            self._skill30_shape_support_calls=0
            self._skill30_base_origin_speed_mps=None
            self.last_skill30_record=None
            # The latest completed normal interval survives Fast.start/reset;
            # the exact index check below prevents using a stale episode.

        def set_interval_evidence(self, *, control_index,
                                  contact_free_5_by_wheel,
                                  native_wheel_min_z_5x4):
            index=int(control_index)
            contact=tuple(contact_free_5_by_wheel)
            minima=np.asarray(native_wheel_min_z_5x4,dtype=float)
            old=getattr(self,'_skill30_interval_evidence',None)
            if (index<0 or len(contact)!=4 or any(type(x) is not bool for x in contact)
                    or minima.shape!=(5,4) or not np.isfinite(minima).all()
                    or old is not None and index<=old['control_index']):
                raise ValueError('C30 previous complete 5T interval differs')
            self._skill30_interval_evidence={
                'control_index':index,'contact_free_5_by_wheel':contact,
                'native_wheel_min_z_5x4':minima.copy()}

        def drain_macro_events(self):
            events=tuple(self._skill30_events)
            self._skill30_events=[]
            return events

        def _wheel_shape_min30(self, data):
            self._skill30_shape_support_calls+=1
            state={'geom_xpos':np.asarray(data.geom_xpos),
                   'geom_xmat':np.asarray(data.geom_xmat).reshape((-1,3,3))}
            rows=collision_bounds24(geometry_binding['kinematics24'],state,wheel_map)
            result=np.full(4,np.inf)
            for row in rows:
                leg=row['wheel_index']
                if leg is not None:
                    result[leg]=min(result[leg],row['minimum_world_m'][2])
            if not np.isfinite(result).all():
                raise RuntimeError('compiled whole-wheel primitive is missing')
            return result

        def _prepare_leg(self):
            if self._skill30_cancel_requested:
                self.leg=None
                self.loading_leg=None
                self._enter('abort_hold')
                return
            super()._prepare_leg()  # exact old planning and audited C27 hook
            if self.phase != 'shift' or self.failure is not None:
                # The frozen planner can abort. It has not produced a usable
                # shift, so do not run extra IK or label a macro as latched.
                return
            index=int(control_index_provider())
            if index<0 or self._skill30_latches!=self.index:
                raise RuntimeError('C30 latch must equal this real global control and leg order')
            original_from=np.asarray(self.body_from,dtype=float).copy()
            original_to=np.asarray(self.body_to,dtype=float).copy()
            original_shift=float(self.shift_time)
            plan_com_height=float(self.com_height)
            proposed=(0.0,0.0,0.0)
            projected=proposed
            consumed=proposed
            reason='teacher_exact_zero'
            extra_ik=None
            static_margin=None
            extra_qpos=None
            extra_points=None
            extra_com=None
            if not arm.startswith('zero_'):
                toward=original_from[:2]-original_to[:2]
                distance=float(np.linalg.norm(toward))
                world_delta=(np.zeros(2) if distance==0.0 else
                             min(.008,distance)*toward/distance)
                cy,sy=math.cos(self.yaw),math.sin(self.yaw)
                local=np.array([cy*world_delta[0]+sy*world_delta[1],
                                -sy*world_delta[0]+cy*world_delta[1]])
                proposed=(float(local[0]),float(local[1]),.5)
                local=np.clip(local,-.012,.012)
                projected=(float(local[0]),float(local[1]),.5)
                world_delta=np.array([cy*local[0]-sy*local[1],
                                      sy*local[0]+cy*local[1]])
                candidate=original_to.copy()
                candidate[:2]+=world_delta
                if (np.linalg.norm(candidate[:2]-original_from[:2])>.120
                        or not np.isfinite(candidate).all()):
                    reason='max_shift_rejected_xy'
                else:
                    # One extra old plan IK only, in the already-authorized
                    # start/compute scratch scope. No second model or data;
                    # this one additional plan IK uses the original forwards.
                    from d1_fast_side_step import _edges
                    from wheel_legged_control.d1.model import (
                        JOINT_POSITION_LOW, JOINT_POSITION_HIGH)
                    target,extra_ik,points,com=self._ik(
                        candidate,self.quat,self.feet,None,None,outer=2,inner=6)
                    extra_qpos=np.asarray(self.scratch.qpos).copy().tolist()
                    extra_points=np.asarray(points).copy().tolist()
                    extra_com=np.asarray(com).copy().tolist()
                    stance=np.delete(np.arange(4),self.leg)
                    edges=_edges(points[stance,:2])
                    static_margin=min((normal@(com[:2]-origin)
                                       for origin,normal in edges),default=-1.)
                    joints=np.asarray(target,dtype=float)
                    in_range=(joints.shape==(16,) and
                        np.all(joints>=np.asarray(JOINT_POSITION_LOW)-1e-12) and
                        np.all(joints<=np.asarray(JOINT_POSITION_HIGH)+1e-12))
                    if (not np.isfinite(joints).all() or not in_range
                            or not math.isfinite(extra_ik) or extra_ik>.008
                            or static_margin<.028):
                        reason='extra_plan_IK_or_tripod_margin_rejected_xy'
                    else:
                        self.body_to=candidate
                        self.shift_time=self._move_time(self.body_to-self.body_from)
                        consumed=projected
                        reason='body_xy_projected_and_consumed'
                # λ remains separately available even when XY is rejected.
                consumed=(consumed[0],consumed[1],.5)
            self._skill30_lambda=float(consumed[2])
            self._skill30_horizontal_started=False
            self._skill30_horizontal_start_control=None
            self._skill30_horizontal_elapsed=0.0
            self._skill30_horizontal_commanded_last=False
            self._skill30_foot_velocity[:]=0.0
            self._skill30_foot_accel[:]=0.0
            self._skill30_latches+=1
            self._skill30_events.append({
                'schema':SKILL_SCHEMA30,'control_index':index,
                'latch_control_index':index,'macro_index':self._skill30_latches-1,
                'leg_index':int(self.leg),
                'proposed_action':proposed,'projected_action':projected,
                'consumed_action':consumed,'reason':reason,
                'teacher_body_from_m':original_from.tolist(),
                'teacher_body_to_m':original_to.tolist(),
                'teacher_shift_time_s':original_shift,
                'plan_com_height_m':plan_com_height,
                'consumed_body_from_m':np.asarray(self.body_from).tolist(),
                'consumed_body_to_m':np.asarray(self.body_to).tolist(),
                'consumed_shift_time_s':float(self.shift_time),
                'extra_plan_ik_error_m':extra_ik,
                'extra_plan_candidate_qpos23':extra_qpos,
                'extra_plan_contact_points_m':extra_points,
                'extra_plan_com_m':extra_com,
                'predicted_tripod_static_margin_m':static_margin,
                'initial_body_yaw_rad':float(self.yaw)})

        def cancel(self):
            if (self.status['active'] and self._skill30_lambda>0.0
                    and self.leg is not None and self.phase in ('lift','swing','lower')):
                if not self._skill30_cancel_requested:
                    self._skill30_cancel_requested=True
                    self._skill30_cancel_at=int(control_index_provider())
                    self.fixed_cancelled=True
                return
            return super().cancel()

        def _advance(self):
            if arm.startswith('zero_') or self.phase not in ('lift','swing'):
                self._skill30_base_origin_speed_mps=None
                previous=self.phase
                super()._advance()
                if previous=='lower' and self.phase=='load' and self._skill30_cancel_requested:
                    self.failure='cancelled'
                    self._enter('abort_hold')
                if previous=='load' and self.phase=='recenter' and self._skill30_cancel_requested:
                    self.failure='cancelled'
                    self._enter('abort_hold')
                return
            # Nonzero airborne reference. This reproduces the old fault inputs
            # and checks before replacing only the lift/swing reference clocks.
            from d1_fast_side_step import G, _edges
            self.time+=self.dt
            self.phase_time+=self.dt
            contacts=self._contacts()
            d=self.plant.measurement_data
            points,_=self._geometry(d)
            com=self._com(d)
            stance=np.delete(np.arange(4),self.leg)
            self.com_height=float(np.clip(com[2]-float(np.min(points[:,2])),.10,1.))
            boundary=_edges(points[stance,:2])
            zmp=com[:2]-(self.com_height/G)*self.body_accel[:2]
            self.margin=min((n@(com[:2]-a) for a,n in boundary),default=-1.)
            self.dynamic_margin=min((n@(zmp-a) for a,n in boundary),default=-1.)
            self.lift_peaks[self.leg]=max(self.lift_peaks[self.leg],float(points[self.leg,2]))
            # Preserve the frozen Fast._advance object-velocity query once per
            # compute, including the C27 measured-call/cache audit identity.
            self._skill30_base_origin_speed_mps=float(np.linalg.norm(
                self.plant.base_origin_velocity()))
            if (self.status['active'] and
                    (np.max(np.abs(self.plant.base_rpy[:2]))>.32 or
                     self.plant.base_position[2]<.32)):
                self._abort('body_pose_limit')
            if self.status['active'] and np.max(np.abs(d.qvel[self.vadr[:,:3]]))>18.:
                self._abort('joint_velocity_limit')
            self.missing_support_time=(self.missing_support_time+self.dt
                if not contacts[stance].all() else 0.0)
            if self.missing_support_time>.12:
                self._abort('stance_contact_lost')
            self.margin_fault_time=(self.margin_fault_time+self.dt
                if self.dynamic_margin<self.cfg.margin_fault_m else 0.0)
            if self.margin_fault_time>self.cfg.margin_fault_time_s:
                self._abort('support_margin_lost')
            if self.phase not in ('lift','swing'):
                return
            leg=int(self.leg)
            current_index=int(control_index_provider())
            evidence=getattr(self,'_skill30_interval_evidence',None)
            previous_ok=(evidence is not None and
                evidence['control_index']==current_index-1 and
                evidence['contact_free_5_by_wheel'][leg])
            actual_min=float(self._wheel_shape_min30(d)[leg])
            if self._skill30_horizontal_commanded_last and actual_min<=.012:
                self._abort('horizontal_actual_whole_wheel_clearance_lost')
                return
            if self._skill30_horizontal_commanded_last and evidence is not None:
                if evidence['control_index']!=current_index-1 or np.min(
                        evidence['native_wheel_min_z_5x4'][:,leg])<=.012:
                    self._abort('horizontal_native_whole_wheel_clearance_lost')
                    return
            if self.phase=='lift':
                self.load_weight=0.0
                t=self.phase_time
                height,up_v,up_a=_curve(t,self.cfg.lift_time_s)
                self.foot_offset[2]=self.cfg.lift_height_m*height
                self._skill30_foot_velocity[2]=self.cfg.lift_height_m*up_v
                self._skill30_foot_accel[2]=self.cfg.lift_height_m*up_a
                permitted=(self._skill30_lambda>0.0 and not self._skill30_horizontal_started
                    and t>=(1.0-self._skill30_lambda)*self.cfg.lift_time_s
                    and actual_min>.012 and previous_ok and not contacts[leg] and
                    self.dynamic_margin>self.cfg.margin_gate_m and contacts[stance].all())
                if permitted:
                    self._skill30_horizontal_started=True
                    self._skill30_horizontal_start_control=current_index
                    self._skill30_horizontal_elapsed=0.0
                if (t>=self.cfg.lift_time_s and
                        not self._skill30_horizontal_started and
                        actual_min>.012 and previous_ok and not contacts[leg] and
                        points[leg,2]>self.cfg.lift_clearance_m):
                    # λ=0 is the serial teacher timing; λ>0 may also need to
                    # wait until this first genuinely safe completed interval.
                    self._skill30_horizontal_started=True
                    self._skill30_horizontal_start_control=current_index
                    self._skill30_horizontal_elapsed=0.0
                self._skill30_last_gate={
                    'actual_whole_wheel_min_z_m':actual_min,
                    'previous_interval_index':None if evidence is None else evidence['control_index'],
                    'previous_contact_free_5':bool(previous_ok),
                    'horizontal_start_permitted':bool(permitted),
                    'early_lower_reason':'no_future_reference_shape_certificate'}
                if (t>=self.cfg.lift_time_s and
                        self._skill30_horizontal_started and actual_min>.012
                        and previous_ok and not contacts[leg] and
                        points[leg,2]>self.cfg.lift_clearance_m):
                    self._enter('swing')
                elif t>self.cfg.lift_time_s+self.cfg.liftoff_extra_s:
                    self._abort('liftoff_timeout')
            else:
                self.load_weight=0.0
                self._skill30_foot_velocity[2]=0.0
                self._skill30_foot_accel[2]=0.0
                if not self._skill30_horizontal_started:
                    if actual_min>.012 and previous_ok:
                        self._skill30_horizontal_started=True
                        self._skill30_horizontal_start_control=current_index
                    else:
                        self._abort('swing_started_without_whole_wheel_native_gate')
                        return
                self._skill30_last_gate={
                    'actual_whole_wheel_min_z_m':actual_min,
                    'previous_interval_index':None if evidence is None else evidence['control_index'],
                    'previous_contact_free_5':bool(previous_ok),
                    'horizontal_start_permitted':False,
                    'early_lower_reason':'no_future_reference_shape_certificate'}
                self.foot_offset[2]=self.cfg.lift_height_m
            if self.phase not in ('lift','swing'):
                return
            if self._skill30_horizontal_started:
                # The start control holds zero horizontal reference; the next
                # control advances the same latched quintic clock exactly once.
                if current_index>self._skill30_horizontal_start_control:
                    self._skill30_horizontal_elapsed+=self.dt
                fraction,v,a=_curve(self._skill30_horizontal_elapsed,self.swing_time)
                self.foot_offset[:2]=self.delta[:2]*fraction
                self._skill30_foot_velocity[:2]=self.delta[:2]*v
                self._skill30_foot_accel[:2]=self.delta[:2]*a
                self._skill30_horizontal_commanded_last=True
                if self.phase=='swing' and self._skill30_horizontal_elapsed>=self.swing_time:
                    self.foot_offset[:2]=self.delta[:2]
                    self._enter('lower')

        def compute(self):
            result=super().compute()
            if arm.startswith('zero_'):
                # Diagnostics only: super performed the exact original numeric
                # path, including phase and torque arithmetic.
                reference_min=None
            else:
                reference_min=self._wheel_shape_min30(self.scratch).tolist()
                if (self.phase in ('lift','swing') and
                        self._skill30_horizontal_started and
                        self._skill30_horizontal_elapsed<self.swing_time and
                        self.leg is not None
                        and reference_min[self.leg]<=.012):
                    raise RuntimeError('C30 reference whole-wheel clearance failed before plant step')
            leg=self.leg
            body_velocity=np.zeros(3)
            if self.phase in ('shift','recenter') and self.shift_time>0:
                _,v,_=_curve(self.phase_time,self.shift_time)
                body_velocity=(self.body_to-self.body_from)*v
            foot_position=(np.asarray(self.feet[leg])+self.foot_offset
                           if leg is not None else None)
            foot_velocity=self._skill30_foot_velocity.copy()
            foot_accel=self._skill30_foot_accel.copy()
            if not arm.startswith('zero_') and self.phase=='lower' and leg is not None:
                t=float(self.phase_time)
                top=float(self.cfg.lift_height_m)
                hover=min(self.cfg.lower_hover_m,max(0.0,.25*top))
                if t<=self.cfg.lower_time_s:
                    _,v,a=_curve(t,self.cfg.lower_time_s)
                    foot_velocity[2]=-(top-hover)*v
                    foot_accel[2]=-(top-hover)*a
                elif self.foot_offset[2]>-self.cfg.probe_depth_m:
                    foot_velocity[2]=-self.cfg.probe_speed
                    foot_accel[2]=0.0
                else:
                    foot_velocity[2]=foot_accel[2]=0.0
            elif self.phase not in ('lift','swing'):
                foot_velocity[:]=foot_accel[:]=0.0
            self.last_skill30_record={
                'schema':SKILL_SCHEMA30,'control_index':int(control_index_provider()),
                'phase':str(self.phase),'leg_index':None if leg is None else int(leg),
                'macro_index':self._skill30_latches-1,
                'horizontal_active':bool(self._skill30_horizontal_started and
                    self._skill30_horizontal_elapsed<self.swing_time),
                'horizontal_elapsed_s':float(self._skill30_horizontal_elapsed),
                'horizontal_start_control_index':self._skill30_horizontal_start_control,
                'early_lower_enabled':False,
                'early_lower_reason':'no_future_reference_shape_certificate',
                'lift_elapsed_s':float(self.phase_time) if self.phase=='lift' else None,
                'lower_elapsed_s':float(self.phase_time) if self.phase=='lower' else None,
                'cancel_requested':bool(self._skill30_cancel_requested),
                'cancel_control_index':self._skill30_cancel_at,
                'body_reference_position_m':np.asarray(self.pose).tolist(),
                'body_reference_velocity_mps':np.asarray(body_velocity).tolist(),
                'body_reference_accel_mps2':np.asarray(self.body_accel).tolist(),
                'foot_reference_position_m':None if foot_position is None else
                    np.asarray(foot_position).tolist(),
                'foot_reference_position_source':'feet_plus_foot_offset_before_IK_height_correction',
                'foot_reference_velocity_mps':foot_velocity.tolist(),
                'foot_reference_accel_mps2':foot_accel.tolist(),
                'foot_velocity_mps':foot_velocity.tolist(),
                'foot_accel_mps2':foot_accel.tolist(),
                'foot_offset_m':np.asarray(self.foot_offset).tolist(),
                'reference_qpos23':None if arm.startswith('zero_') else
                    np.asarray(self.scratch.qpos).tolist(),
                'reference_whole_wheel_min_z_m':reference_min,
                'pure_shape_support_calls_cumulative':self._skill30_shape_support_calls,
                'actual_base_origin_speed_mps':self._skill30_base_origin_speed_mps,
                'gate':dict(self._skill30_last_gate)}
            return result

    # Both classes are ordinary Python classes with the same instance layout.
    # No second controller, scratch MjData, plant, or model is constructed.
    side.__class__=SkilledFast30
    side._skill30_arm=arm
    side._skill30_interval_evidence=None
    side.reset()
    return side
