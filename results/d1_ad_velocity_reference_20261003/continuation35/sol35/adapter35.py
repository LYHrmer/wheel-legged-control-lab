"""C35 paired Fast controller; source-only until a separately frozen physical GO.

One existing plant and one Fast scratch. No integration, model load or B22 call.
The pure core requests references; root35 owns five-native support evidence.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import mujoco
import numpy as np

from d1_fast_side_step import (FAST_PROFILE, FastSideStepController,
                               _dense_inertia, _wrap)
from core35 import Core35, Command35, Sensed35
from pair_ik_targets35 import pair_ik_targets35
from wheel_legged_control.d1.model import JOINT_POSITION_HIGH, JOINT_POSITION_LOW


AIR_PHASES35 = ('lift','horizontal','lower','probe','dwell')


def _ramp_peak_speed35(coefficients):
    """Exact polynomial extrema on the entire normalized Hermite segment."""
    derivative = [k*coefficients[k] for k in range(1,6)]
    candidates = [0.,1.]
    for root in np.roots(derivative[::-1]):
        if abs(root.imag) <= 1e-10 and 0. <= root.real <= 1.:
            candidates.append(float(root.real))
    return float(max(abs(np.polyval(coefficients[::-1],u)) for u in candidates))


class PairController35(FastSideStepController):
    """Replace Fast's single-leg phase and compute, retain its deep mechanics."""

    def __init__(self, plant, side_access=None):
        super().__init__(plant, profile=FAST_PROFILE)
        self.access = side_access

    def reset(self):
        """Clear every case-local C35 state as well as Fast's inherited memory."""
        FastSideStepController.reset(self)
        self.core = Core35()
        self.command35 = None
        self.sensed35 = None
        self.started35 = False
        self.entry_z35 = float(self.pose[2])
        self.swing_legs = ()
        self.last_record = None
        self.checked_ramp_coefficients = None
        self.ramp_peak_speed = 0.
        self.previous_horizontal_nonzero = False
        self.previous_horizontal_pair = None

    @property
    def status(self):
        return dict(schema='d1-c35-continuous-side-control-v1',
            phase=self.core.phase, active=self.started35 and not self.done,
            done=bool(self.done), success=bool(self.success),failure=self.failure,
            swing_legs=list(self.swing_legs), pair_index=self.core.pair_index,
            pair_exchanges=self.core.completed_pair_exchanges,
            stop_reason=self.core.stop_reason)

    def start(self, command: Command35, sensed: Sensed35):
        if self.started35:
            return False
        Core35._check(command,sensed)
        if not Core35._valid(command,sensed.control_index):
            return False
        if command.vy_mps == 0. and not command.gait_enabled:
            return False
        if not Core35._all_loaded(sensed) or not sensed.native_safety_ok:
            return False
        if np.max(np.abs(self.plant.base_rpy[:2])) > .12:
            return False
        points,_ = self._geometry(self.plant.measurement_data)
        if np.max(np.abs(points[:,2])) > .01:
            return False
        for a,b in ((0,3),(1,2)):
            relative = points[[a,b],:2]-self.plant.base_position[:2]
            if relative[0,0]*relative[1,0] >= 0 or relative[0,1]*relative[1,1] >= 0:
                return False
        if not np.isfinite(self.plant.data.qpos).all() or not np.isfinite(self.plant.data.qvel).all():
            return False
        FastSideStepController.reset(self)
        self.core = Core35()
        self.entry_z35 = float(self.pose[2])
        self.command35,self.sensed35 = command,sensed
        self.started35 = True
        self.done,self.success,self.failure = False,False,None
        self.swing_legs = ()
        self.last_record = None
        return True

    def set_command(self, command: Command35):
        self.command35 = command

    def set_evidence(self, sensed: Sensed35):
        self.sensed35 = sensed

    def _paired_targets35(self, reference):
        """One scratch solve, masking BOTH swing Z offsets at every outer pass."""
        d = self.scratch
        mujoco.mj_copyData(d,self.model,self.plant.measurement_data)
        pose = np.array((reference.body_xy_m[0],reference.body_xy_m[1],self.entry_z35))
        d.qpos[:3],d.qpos[3:7] = pose,self.quat
        d.qvel[:] = 0.
        requested = np.asarray(reference.feet_world_m,dtype=float)
        anchors = np.asarray(self.core.anchors,dtype=float)
        pair = reference.pair if reference.phase in AIR_PHASES35 else ()
        targets = requested.copy()
        for _ in range(3):
            mujoco.mj_forward(self.model,d)
            _,extent = self._geometry(d)
            targets = pair_ik_targets35(requested,anchors,extent,pair)
            for leg in range(4):
                for _ in range(8):
                    mujoco.mj_forward(self.model,d)
                    error = targets[leg]-d.xpos[self.bodies[leg]]
                    if np.linalg.norm(error) < 1e-6:
                        break
                    mujoco.mj_jacBody(self.model,d,self.jp,self.jr,self.bodies[leg])
                    j = self.jp[:,self.vadr[leg,:3]]
                    step = j.T @ np.linalg.solve(j@j.T+1e-7*np.eye(3),error)
                    if not np.isfinite(step).all():
                        raise RuntimeError('C35 paired IK nonfinite step')
                    q = d.qpos[self.qadr[leg,:3]]+np.clip(step,-.1,.1)
                    d.qpos[self.qadr[leg,:3]] = np.clip(
                        q,JOINT_POSITION_LOW[:3],JOINT_POSITION_HIGH[:3])
                mujoco.mj_forward(self.model,d)
        mujoco.mj_forward(self.model,d)
        error_max = max(float(np.linalg.norm(targets[leg]-d.xpos[self.bodies[leg]]))
                        for leg in range(4))
        return d.qpos[self.plant.qpos_addresses].copy(),error_max,targets.copy()

    def _reference_record35(self, reference):
        row = asdict(reference)
        swing = list(reference.pair) if reference.pair is not None and reference.phase in AIR_PHASES35 else []
        stance = [j for j in range(4) if j not in swing]
        moving_horizontal = (reference.phase == 'horizontal' and
            any(math.hypot(*reference.feet_velocity_world_mps[j][:2]) > 1e-12
                for j in swing))
        prior_horizontal = (self.previous_horizontal_nonzero and
            reference.phase in AIR_PHASES35 and
            reference.pair == self.previous_horizontal_pair)
        row.update(stance_legs=stance,swing_legs=swing,
                   support_mode='pair' if swing else 'all4',
                   support_gate_required=True,
                   horizontal_active=moving_horizontal or prior_horizontal,
                   horizontal_previous_velocity_nonzero=prior_horizontal,
                   horizontal_reference_velocity_nonzero=moving_horizontal,
                   consumed_landing_xy_m=reference.landing_projected_xy_m)
        self.previous_horizontal_nonzero = moving_horizontal
        self.previous_horizontal_pair = reference.pair if moving_horizontal else None
        return row

    def compute(self):
        if not self.started35 or self.command35 is None or self.sensed35 is None:
            raise RuntimeError('C35 compute without declared command/evidence/start')
        if self.done:
            raise RuntimeError('C35 compute after handoff')
        plant,d = self.plant,self.plant.measurement_data
        if not (np.isfinite(plant.data.qpos).all() and np.isfinite(plant.data.qvel).all()):
            raise RuntimeError('C35 nonfinite actual state')
        command,sensed = self.command35,self.sensed35
        phase_before = self.core.phase
        completed_before = self.core.completed_pair_exchanges
        pair_index_before = self.core.pair_index
        anchors_before = self.core.anchors
        reference = self.core.step(command,sensed)
        self.swing_legs = (tuple(reference.pair) if reference.phase in AIR_PHASES35 else ())
        if reference.safe_abort_required:
            self.failure = reference.fault
            raise RuntimeError('C35 physical stop required, torque handoff forbidden: '+str(reference.fault))
        if reference.ramp_coefficients_mps != self.checked_ramp_coefficients:
            self.ramp_peak_speed = _ramp_peak_speed35(reference.ramp_coefficients_mps)
            self.checked_ramp_coefficients = reference.ramp_coefficients_mps
        if self.ramp_peak_speed > .030+1e-12:
            self.failure = 'analytic_reference_speed_gt_030'
            raise RuntimeError(self.failure)
        if abs(self.core.vref) > .030+1e-12:
            self.failure = 'reference_speed_gt_030'
            raise RuntimeError(self.failure)
        com_height = (sensed.whole_com_z_m
                      -min(point[2] for point in sensed.wheel_contact_points_world_m))
        if com_height <= 0 or abs(self.core.aref)*com_height/9.81 > .015+1e-12:
            self.failure = 'reference_acceleration_budget'
            raise RuntimeError(self.failure)
        target,ik_error,ik_world = self._paired_targets35(reference)
        if not np.isfinite(target).all() or ik_error > .008:
            self.failure = 'paired_ik_unreachable'
            raise RuntimeError(self.failure)
        joints = target.reshape(4,4)[:,:3]
        if (np.any(joints < JOINT_POSITION_LOW[:3]-1e-12)
                or np.any(joints > JOINT_POSITION_HIGH[:3]+1e-12)):
            self.failure = 'paired_ik_target_joint_bound'
            raise RuntimeError(self.failure)
        joint_ids = np.asarray(plant.joint_ids).reshape(4,4)[:,:3]
        compiled_limited = np.asarray(self.model.jnt_limited,dtype=bool)[joint_ids]
        compiled_ranges = np.asarray(self.model.jnt_range,dtype=float)[joint_ids]
        if np.any(compiled_limited & ((joints < compiled_ranges[:,:,0]-1e-12)
                                    |(joints > compiled_ranges[:,:,1]+1e-12))):
            self.failure = 'paired_ik_compiled_joint_bound'
            raise RuntimeError(self.failure)
        previous_target = self.previous_target.copy()
        previous_velocity = self.previous_velocity_target.copy()
        target_velocity = np.clip((target-previous_target)/self.dt,-4.,4.)
        target_accel = np.clip((target_velocity-previous_velocity)/self.dt,-60.,60.)
        self.previous_target,self.previous_velocity_target = target.copy(),target_velocity.copy()
        actual_velocity = d.qvel[plant.dof_addresses]
        points,_ = self._geometry(d)
        com = self._com(d)
        linear,angular = plant.base_velocity(local=False)
        desired_pose = np.array((reference.body_xy_m[0],reference.body_xy_m[1],self.entry_z35))
        desired_velocity = np.array((*reference.body_vxy_mps,0.))
        desired_acceleration = np.array((*reference.body_axy_mps2,0.))
        force = self.mass*(np.array((0.,0.,9.81))+desired_acceleration
            +26.*(desired_pose-plant.base_position)+9.*(desired_velocity-linear))
        rpy = plant.base_rpy
        error_rpy = np.array((-rpy[0],-rpy[1],_wrap(self.yaw-rpy[2])))
        cy,sy = np.cos(self.yaw),np.sin(self.yaw)
        rotation = np.array(((cy,-sy,0.),(sy,cy,0.),(0.,0.,1.)))
        moment = 180.*(rotation@error_rpy)-25.*angular
        wrench = np.r_[force,moment]
        weights = np.asarray(reference.force_weights)
        forces = self._allocate(points,com,wrench,weights)
        if self.failure or not np.isfinite(forces).all():
            raise RuntimeError('C35 force allocation failure')
        tau = d.qfrc_bias[plant.dof_addresses].copy()
        jacobians = [None]*4
        for leg in range(4):
            if weights[leg] <= 1e-3:
                continue
            mujoco.mj_jac(self.model,d,self.jp,self.jr,points[leg],self.bodies[leg])
            jacobians[leg] = self.jp[:,plant.dof_addresses].copy()
            tau -= jacobians[leg].T@forces[leg]
        kp,kd = np.full(16,80.),np.full(16,3.)
        for leg in self.swing_legs:
            kp[4*leg:4*leg+3],kd[4*leg:4*leg+3] = 150.,5.
        tau += kp*(target-d.qpos[plant.qpos_addresses]) + kd*(target_velocity-actual_velocity)
        inertia6 = None
        if self.swing_legs:
            dense = _dense_inertia(self.model,self.scratch)
            dofs = self.vadr[list(self.swing_legs),:3].ravel()
            positions = np.array([4*leg+j for leg in self.swing_legs for j in range(3)])
            inertia6 = dense[np.ix_(dofs,dofs)]
            if (self.access is not None and
                    (self.access.last_pair_inertia_6x6 is None or
                     not np.array_equal(inertia6,self.access.last_pair_inertia_6x6))):
                raise RuntimeError('C35 6x6 inertia cache differs from counted fullM')
            feedforward = inertia6@target_accel[positions]
            if not np.isfinite(feedforward).all():
                raise RuntimeError('C35 paired inertia feedforward nonfinite')
            tau[positions] += feedforward
        wheel = np.arange(3,16,4)
        tau[wheel] += (12.*(self.wheel_angles-d.qpos[self.qadr[:,3]])
                       -2.*actual_velocity[wheel])
        tau[wheel] -= (kp[wheel]*(target[wheel]-d.qpos[self.qadr[:,3]])
                       +kd[wheel]*(target_velocity[wheel]-actual_velocity[wheel]))
        if not np.isfinite(tau).all():
            raise RuntimeError('C35 nonfinite torque')
        preclip = tau.copy()
        limit = np.abs(np.asarray(plant.actuator_torque_limit_nm,dtype=float))
        tau = np.clip(tau,-limit,limit)
        reference_row = self._reference_record35(reference)
        reference_row.update(phase_before=phase_before,phase_after=reference.phase,
            pair_index=self.core.pair_index,
            body_pose_world_m=[*reference.body_xy_m,self.entry_z35],
            body_yaw_rad=self.yaw,
            horizontal_common_duration_s=self.core.swing_duration,
            horizontal_origin_world_m=self.core.swing_from,
            horizontal_target_world_m=self.core.swing_to,
            horizontal_span_xy_m=[
                [self.core.swing_to[j][k]-self.core.swing_from[j][k] for k in (0,1)]
                for j in range(4)],
            completed_pair_exchanges=self.core.completed_pair_exchanges,
            stable_stop_controls=self.core.stable_controls)
        calculation = dict(pre_qpos=d.qpos.copy().tolist(),
            pre_qvel=d.qvel.copy().tolist(),target16=target.tolist(),
            target_velocity16=target_velocity.tolist(),target_accel16=target_accel.tolist(),
            previous_target16=previous_target.tolist(),
            previous_velocity_target16=previous_velocity.tolist(),
            ik_world_targets4=ik_world.tolist(),ik_error_m=ik_error,
            contact_points4=points.tolist(),whole_com_world_m=com.tolist(),
            base_linear_velocity_world_mps=np.asarray(linear).tolist(),
            base_angular_velocity_world_rps=np.asarray(angular).tolist(),
            desired_wrench6=wrench.tolist(),allocation_forces4=forces.tolist(),
            jacobians4=[None if x is None else x.tolist() for x in jacobians],
            bias16=d.qfrc_bias[plant.dof_addresses].tolist(),
            wheel_angles4=self.wheel_angles.tolist(),
            kp16=kp.tolist(),kd16=kd.tolist(),
            paired_inertia6=None if inertia6 is None else inertia6.tolist(),
            ramp_peak_speed_mps=self.ramp_peak_speed,
            preclip_torque16=preclip.tolist(),safe_torque16=tau.tolist(),
            torque_limits16=limit.tolist())
        self.last_record = dict(schema='d1-c35-continuous-side-control-v1',
            reference=reference_row,command=asdict(command),sensed=asdict(sensed),
            calculation=calculation,pair_events=(
                [dict(event='pair_completed',pair_index=pair_index_before,
                      pair_legs=list(((0,3),(1,2))[pair_index_before]),
                      control_index=sensed.control_index,
                      anchors_before_world_m=anchors_before,
                      anchors_after_world_m=self.core.anchors)]
                if self.core.completed_pair_exchanges > completed_before else []))
        self.done = bool(reference.ready_for_handoff)
        self.success = self.done
        return tau


__all__ = ['PairController35']
