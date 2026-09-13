"""Experimental slow flat-ground crawl using simulator state and motor torque.

This separate controller is outside the frozen RL experiments. It reads live
MuJoCo state, solves IK on scratch data, and returns torques; only the caller
integrates the plant. No policy or hardware robustness claim is made.
"""
from __future__ import annotations

import mujoco
import numpy as np

from wheel_legged_control.d1.model import JOINT_POSITION_HIGH, JOINT_POSITION_LOW


def _smooth(value):
    t = np.clip(value, 0., 1.)
    return t*t*t*(10.+t*(-15.+6.*t))


def _edges(points):
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))]
    out = []
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        edge = b-a
        if np.linalg.norm(edge) > 1e-8:
            out.append((a, np.array([-edge[1], edge[0]])/np.linalg.norm(edge)))
    return out


class SideStepController:
    """One slow ±3 cm, four-wheel lifting sequence; +1 means initial body left."""

    def __init__(self, plant):
        self.plant, self.model = plant, plant.model
        self.scratch = mujoco.MjData(self.model)
        self.qadr = plant.qpos_addresses.reshape(4, 4)
        self.vadr = plant.dof_addresses.reshape(4, 4)
        self.bodies = list(plant.wheel_body_ids_by_leg)
        self.wheel_joints = plant.joint_ids.reshape(4, 4)[:, 3]
        self.mass = float(self.model.body_mass.sum())
        self.jp, self.jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.dt = plant.control_dt
        self.events = []
        self.reset()

    def reset(self):
        """Discard controller memory; never reset or move the physical plant."""
        d = self.plant.measurement_data
        self.pose = d.qpos[:3].copy()
        self.yaw = float(self.plant.base_rpy[2])
        self.quat = np.array([np.cos(self.yaw/2), 0., 0., np.sin(self.yaw/2)])
        self.feet = d.xpos[self.bodies].copy()
        self.wheel_angles = d.qpos[self.qadr[:, 3]].copy()
        self.phase, self.failure, self.done, self.success = "idle", None, False, False
        self.leg, self.index, self.direction = None, 0, 0
        self.time, self.phase_time, self.contact_dwell = 0., 0., 0.
        self.body_from, self.body_to = self.pose.copy(), self.pose.copy()
        self.initial_pose = self.pose.copy()
        self.delta = np.zeros(3)
        self.foot_offset = np.zeros(3)
        self.previous_target = d.qpos[self.plant.qpos_addresses].copy()
        self.margin, self.ik_error = 0., 0.
        self.missing_support_time = 0.

    @property
    def status(self):
        return {"phase": self.phase, "done": self.done, "success": self.success,
                "failure": self.failure, "active": self.phase not in ("idle", "done", "failed"),
                "active_leg": None if self.leg is None else ("FL", "FR", "RL", "RR")[self.leg],
                "direction": self.direction, "initial_yaw_rad": self.yaw,
                "current_yaw_rad": float(self.plant.base_rpy[2]),
                "support_margin_m": float(self.margin), "ik_error_m": self.ik_error,
                "torque_source": "d1_side_step", "state_source": "simulator_truth"}

    def _contacts(self):
        d = self.plant.measurement_data
        active = np.zeros(4, dtype=bool)
        for c in d.contact:
            a, b = int(c.geom1), int(c.geom2)
            other = b if a in self.plant.terrain_geom_ids else a if b in self.plant.terrain_geom_ids else None
            if other is not None and c.efc_address >= 0:
                body = int(self.model.geom_bodyid[other])
                if body in self.bodies:
                    active[self.bodies.index(body)] = True
        return active

    def start(self, direction, distance_m=.03):
        if isinstance(direction, (bool, np.bool_)) or not isinstance(direction, (int, np.integer)) or direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if not np.isfinite(distance_m) or not .01 <= distance_m <= .04:
            raise ValueError("distance_m must be within [0.01, 0.04]")
        if self.status["active"] or not self._contacts().all():
            return False
        # This prototype assumes the local contact plane is world z=0.
        if np.max(np.abs(self._geometry(self.plant.measurement_data)[0][:, 2])) > .01:
            return False
        if np.max(np.abs(self.plant.base_rpy[:2])) > .12 or np.linalg.norm(self.plant.base_origin_velocity()) > .04:
            return False
        self.reset()
        self.direction = int(direction)
        self.delta = direction*distance_m*np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.])
        self.order = [2, 0, 3, 1] if direction > 0 else [3, 1, 2, 0]
        self._prepare_leg()
        return True

    def _enter(self, phase):
        self.phase, self.phase_time, self.contact_dwell = phase, 0., 0.
        self.events.append({"time": self.time, "phase": phase, "leg": self.leg, "failure": self.failure})

    def _geometry(self, data):
        axes = data.xaxis[self.wheel_joints]
        az = np.clip(axes[:, 2], -1., 1.)
        extent = .087*np.sqrt(np.maximum(1e-10, 1-az*az))+.020*np.abs(az)
        points = data.xpos[self.bodies].copy()
        up = np.tile([0., 0., 1.], (4, 1))
        points -= .087*(up-axes*az[:, None])/np.sqrt(np.maximum(1e-10, 1-az*az))[:, None]
        # Near vertical wheels have a contact line; choose its center.
        signs = np.where(np.abs(az) < .002, 0., np.sign(az))
        points -= .020*signs[:, None]*axes
        return points, extent

    def _com(self, data):
        return (self.model.body_mass[:, None]*data.xipos).sum(axis=0)/self.mass

    def _prepare_leg(self):
        self.leg = self.order[self.index]
        points, _ = self._geometry(self.plant.measurement_data)
        boundary = _edges(np.delete(points[:, :2], self.leg, axis=0))
        offset = self._com(self.plant.measurement_data)-self.plant.base_position
        target = self.initial_pose+self.delta*(self.index/4.)
        xy = target[:2]+offset[:2]
        for _ in range(24):
            for origin, normal in boundary:
                xy += normal*max(0., .040-normal@(xy-origin))
        self.body_from, self.body_to = self.pose.copy(), target.copy()
        self.body_to[:2] = xy-offset[:2]
        self.foot_offset[:] = 0.
        self._enter("shift")

    def cancel(self):
        if self.status["active"]:
            self._abort("cancelled")

    def _abort(self, reason):
        if self.failure is None:
            self.failure = reason
        # A fault can occur after _advance marked completion in this same tick.
        # Revoke that handoff until a subsequent stable contact dwell succeeds.
        self.done, self.success = False, False
        if self.phase not in ("abort_land", "abort_hold"):
            if self.phase in ("shift", "unload"):
                self.leg = None
                self.foot_offset[:] = 0.
            self.land_from = self.foot_offset.copy()
            self._enter("abort_land" if self.leg is not None else "abort_hold")

    def _advance(self):
        self.time += self.dt
        self.phase_time += self.dt
        contacts = self._contacts()
        d = self.plant.measurement_data
        points, _ = self._geometry(d)
        stance = np.arange(4) if self.leg is None else np.delete(np.arange(4), self.leg)
        self.margin = min((n@(self._com(d)[:2]-a) for a, n in _edges(points[stance, :2])), default=-1.)
        if self.status["active"] and (np.max(np.abs(self.plant.base_rpy[:2])) > .32 or self.plant.base_position[2] < .32):
            self._abort("body_pose_limit")
        if self.status["active"] and np.max(np.abs(d.qvel[self.vadr[:, :3]])) > 18.:
            self._abort("joint_velocity_limit")
        if self.phase in ("lift", "swing", "lower"):
            self.missing_support_time = self.missing_support_time+self.dt if not contacts[stance].all() else 0.
            if self.missing_support_time > .12:
                self._abort("stance_contact_lost")
        t = self.phase_time
        if self.phase == "shift":
            # Camber changes the actual support polygon as the body shifts.
            # Correct that measured geometry slowly before permitting liftoff.
            if t >= 3. and self.margin < .030:
                boundary = _edges(points[stance, :2])
                _, inward = min(boundary, key=lambda edge: edge[1]@(self._com(d)[:2]-edge[0]))
                self.body_to[:2] += inward*.006*self.dt
            self.pose = self.body_from+_smooth(t/3.)*(self.body_to-self.body_from)
            ready = self.margin > .024 and np.linalg.norm(self.plant.base_origin_velocity()) < .025 and contacts.all()
            if t >= 3. and ready:
                self._enter("unload")
            elif t > 8.:
                self._abort("support_shift_timeout")
        elif self.phase == "unload":
            if t >= .6:
                self._enter("lift")
        elif self.phase == "lift":
            self.foot_offset[2] = .035*_smooth(t/1.)
            if t >= 1. and points[self.leg, 2] > .012 and not contacts[self.leg]:
                self._enter("swing")
            elif t > 2.5:
                self._abort("liftoff_timeout")
        elif self.phase == "swing":
            self.foot_offset = self.delta*_smooth(t/1.5)+[0., 0., .035]
            if t >= 1.5:
                self._enter("lower")
        elif self.phase in ("lower", "abort_land"):
            if self.phase == "lower":
                self.foot_offset = self.delta+np.array([0., 0., .035-.039*_smooth(t/1.2)])
            else:
                self.foot_offset = self.land_from.copy()
                self.foot_offset[2] = (1-_smooth(t/1.2))*self.land_from[2]-.004*_smooth(t/1.2)
            if t > 1.1 and contacts[self.leg]:
                self.contact_dwell += self.dt
            else:
                self.contact_dwell = 0.
            if self.contact_dwell >= .2:
                self.feet[self.leg, :2] += self.foot_offset[:2]
                self.foot_offset[:] = 0.
                self.leg = None
                self._enter("abort_hold" if self.failure else "load")
            elif t > 4.:
                self._abort("touchdown_timeout")
        elif self.phase == "load" and t >= .8:
            self.index += 1
            if self.index < 4:
                self._prepare_leg()
            else:
                self.body_from, self.body_to = self.pose.copy(), self.initial_pose+self.delta
                self._enter("recenter")
        elif self.phase == "recenter":
            self.pose = self.body_from+_smooth(t/3.)*(self.body_to-self.body_from)
            if t >= 3. and contacts.all() and np.linalg.norm(self.plant.base_origin_velocity()) < .04:
                error = self.plant.base_position-(self.initial_pose+self.delta)
                yaw_error = np.arctan2(np.sin(self.plant.base_rpy[2]-self.yaw), np.cos(self.plant.base_rpy[2]-self.yaw))
                self.success = np.linalg.norm(error[:2]) < .012 and abs(yaw_error) < .12
                self.failure = None if self.success else "final_tracking_error"
                self.done = True
                self._enter("done" if self.success else "failed")
            elif t > 8.:
                self._abort("recenter_timeout")
        elif self.phase == "abort_hold":
            if contacts.all() and np.linalg.norm(self.plant.base_origin_velocity()) < .04:
                self.contact_dwell += self.dt
                if self.contact_dwell > .5:
                    self.done = True
                    self._enter("failed")
            else:
                self.contact_dwell = 0.

    def _targets(self):
        d = self.scratch
        mujoco.mj_copyData(d, self.model, self.plant.measurement_data)
        d.qpos[:3], d.qpos[3:7] = self.pose, self.quat
        d.qvel[:] = 0.
        targets = self.feet.copy()
        if self.leg is not None:
            targets[self.leg] += self.foot_offset
        self.ik_error = 0.
        for _ in range(3):
            mujoco.mj_forward(self.model, d)
            _, extent = self._geometry(d)
            targets[:, 2] = extent-.001
            if self.leg is not None:
                targets[self.leg, 2] += self.foot_offset[2]
            for leg in range(4):
                for _ in range(8):
                    mujoco.mj_forward(self.model, d)
                    error = targets[leg]-d.xpos[self.bodies[leg]]
                    if np.linalg.norm(error) < 1e-6:
                        break
                    mujoco.mj_jacBody(self.model, d, self.jp, self.jr, self.bodies[leg])
                    j = self.jp[:, self.vadr[leg, :3]]
                    dq = j.T@np.linalg.solve(j@j.T+1e-7*np.eye(3), error)
                    d.qpos[self.qadr[leg, :3]] = np.clip(d.qpos[self.qadr[leg, :3]]+np.clip(dq, -.1, .1), JOINT_POSITION_LOW[:3], JOINT_POSITION_HIGH[:3])
                self.ik_error = max(self.ik_error, float(np.linalg.norm(error)))
        return d.qpos[self.plant.qpos_addresses].copy()

    def compute(self):
        if not (np.isfinite(self.plant.data.qpos).all() and np.isfinite(self.plant.data.qvel).all()):
            self.failure, self.phase, self.done, self.success = "nonfinite_state", "abort_hold", False, False
            return np.zeros(16)
        self._advance()
        d = self.plant.measurement_data
        target = self._targets()
        if self.ik_error > .008:
            self._abort("ik_unreachable")
        target_velocity = np.clip((target-self.previous_target)/self.dt, -1.5, 1.5)
        self.previous_target = target.copy()
        velocity = d.qvel[self.plant.dof_addresses]
        points, _ = self._geometry(d)
        com = self._com(d)
        linear, angular = self.plant.base_velocity(local=False)
        force = self.mass*(np.array([0., 0., 9.81])+20*(self.pose-self.plant.base_position)-8*linear)
        rpy = self.plant.base_rpy
        error_rpy = np.array([-rpy[0], -rpy[1], np.arctan2(np.sin(self.yaw-rpy[2]), np.cos(self.yaw-rpy[2]))])
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        heading_rotation = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
        moment = 180*(heading_rotation@error_rpy)-25*angular
        stance = list(range(4))
        if self.leg is not None and self.phase in ("unload", "lift", "swing", "lower", "abort_land"):
            stance.remove(self.leg)
        allocation = np.zeros((6, 3*len(stance)))
        for col, leg in enumerate(stance):
            x, y, z = points[leg]-com
            allocation[:3, 3*col:3*col+3] = np.eye(3)
            allocation[3:, 3*col:3*col+3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        forces = np.linalg.lstsq(allocation, np.r_[force, moment], rcond=None)[0].reshape(-1, 3)
        tau = d.qfrc_bias[self.plant.dof_addresses].copy()
        for leg, f in zip(stance, forces):
            f[2] = np.clip(f[2], 0., .85*self.mass*9.81)
            f[:2] = np.clip(f[:2], -.6*f[2], .6*f[2])
            mujoco.mj_jac(self.model, d, self.jp, self.jr, points[leg], self.bodies[leg])
            tau -= self.jp[:, self.plant.dof_addresses].T@f
        tau += 80*(target-d.qpos[self.plant.qpos_addresses])+3*(target_velocity-velocity)
        # Position brake keeps rolling wheels from defeating fixed stance anchors.
        wheel = np.arange(3, 16, 4)
        tau[wheel] += 12*(self.wheel_angles-d.qpos[self.qadr[:, 3]])-2*velocity[wheel]
        tau[wheel] -= 80*(target[wheel]-d.qpos[self.qadr[:, 3]])+3*(target_velocity[wheel]-velocity[wheel])
        if not np.isfinite(tau).all():
            self._abort("nonfinite_torque")
            tau = np.zeros(16)
        return np.clip(tau, -self.plant.actuator_torque_limit_nm, self.plant.actuator_torque_limit_nm)
