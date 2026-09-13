"""Experimental fast body-relative lateral stepping for the D1 prototype.

Primary implementation: Claude Opus, session 3c2827e5-6927-4524-a0a2-d2a1d21b966c.
Locally adapted and physically validated; original response and failed candidates
are retained separately. This is not a policy or hardware validation result.

This is an experimental controller outside the frozen RL experiments. It reads
live MuJoCo state, plans and solves IK on a private scratch MjData, and returns
motor torque only; the caller alone integrates the plant. No policy, hardware or
robustness claim is made.

Relation to ``scripts/d1_side_step.py``: that module stays untouched and remains
the authoritative conservative fallback. This module subclasses it and replaces
the phase schedule, the support-shift planner and the force allocator. The
public interface (start / compute / cancel / reset / status) is preserved.

Why it is faster (and not merely rescaled): the conservative gait spends ~15 s
of its ~43 s cycle in fixed 3 s body shifts plus a 6 mm/s inward margin creep,
because the support polygon is planned from the *measured* pre-shift geometry
and the camber change during the shift invalidates it. Here the shift target is
found by a fixed point over scratch IK + predicted contact geometry, and the
shift duration is sized from an explicit ZMP acceleration budget. Control dt and
physics dt are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from wheel_legged_control.d1.model import JOINT_POSITION_HIGH, JOINT_POSITION_LOW

if __package__:
    from .d1_side_step import SideStepController, _edges, _smooth
else:
    from d1_side_step import SideStepController, _edges, _smooth

G = 9.81
#: peak of d2/dt2 of the unit quintic smoothstep, used to size move durations.
SMOOTH_PEAK_ACCEL = 5.7735
_SWING_PHASES = ("unload", "lift", "swing", "lower", "abort_land")


def _smooth_ddot(value):
    """Second derivative of ``_smooth`` with respect to its normalised argument."""
    t = np.clip(value, 0., 1.)
    return 60.*t - 180.*t*t + 120.*t*t*t


def _wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _dense_inertia(model, data):
    """Expand inertia with either supported MuJoCo fullM binding signature."""
    matrix = np.zeros((model.nv, model.nv))
    if hasattr(data, "qM"):
        # MuJoCo versions exposing the packed inertia array accept dst, qM.
        mujoco.mj_fullM(model, matrix, data.qM)
    else:
        # Newer versions expose the operation through MjData directly.
        mujoco.mj_fullM(model, data, matrix)
    return matrix


@dataclass(frozen=True)
class GaitProfile:
    """Timing, margin and gain configuration for one lateral stepping cycle.

    Durations are seconds of *simulated* time at the caller's control_dt; none of
    them scales the clock. Distances are metres, angles radians.
    """

    name: str = "fast"

    # --- support planning -------------------------------------------------
    margin_target_m: float = .040       # planned static tripod margin at the shifted pose
    margin_gate_m: float = .020         # dynamic margin required before liftoff
    margin_fault_m: float = -.004       # dynamic margin that counts as a fault while airborne
    margin_fault_time_s: float = .10
    plan_iterations: int = 3            # fixed-point passes over (body xy -> predicted polygon)
    plan_ik_outer: int = 2
    plan_ik_inner: int = 6
    max_shift_m: float = .120           # hard clamp on the planned body displacement

    # --- body motion ------------------------------------------------------
    adaptive_timing: bool = True
    zmp_budget_m: float = .015          # margin spent on commanded body acceleration
    shift_time_bounds: tuple = (.30, 1.40)
    fixed_shift_time_s: float = 3.0     # used when adaptive_timing is False
    shift_settle_speed: float = .060
    shift_extra_time_s: float = 1.10    # grace after the nominal shift before timeout
    shift_correction_rate: float = .012  # fallback inward creep, m/s
    recenter_extra_time_s: float = 2.0
    recenter_settle_speed: float = .035
    final_position_tolerance_m: float = .012
    final_yaw_tolerance_rad: float = .12
    distance_scale: float = 1.0         # optional open-loop calibration of the commanded step

    # --- leg motion -------------------------------------------------------
    unload_time_s: float = .12
    lift_height_m: float = .035
    lift_time_s: float = .26
    lift_clearance_m: float = .012
    adaptive_swing: bool = True
    swing_foot_speed: float = .16       # peak commanded foot speed during the lateral swing
    swing_time_bounds: tuple = (.22, .90)
    fixed_swing_time_s: float = 1.5
    lower_time_s: float = .26
    lower_hover_m: float = .006         # height where the smooth descent hands over to the probe
    probe_speed: float = .030           # constant-velocity touchdown probe, m/s
    probe_depth_m: float = .015
    touchdown_dwell_s: float = .08
    load_time_s: float = .15
    load_settle_speed: float = .080

    # --- timeouts (extra allowance beyond the nominal duration) -----------
    liftoff_extra_s: float = .60
    touchdown_extra_s: float = .90
    load_extra_s: float = 1.20
    abort_hold_dwell_s: float = .30
    abort_hold_speed: float = .040

    # --- gains ------------------------------------------------------------
    body_kp: float = 26.
    body_kd: float = 9.
    attitude_kp: float = 180.
    attitude_kd: float = 25.
    stance_kp: float = 80.
    stance_kd: float = 3.
    swing_kp: float = 150.
    swing_kd: float = 5.
    wheel_brake_kp: float = 12.
    wheel_brake_kd: float = 2.
    target_velocity_clip: float = 4.
    target_accel_clip: float = 60.
    force_regularization: float = 1e-2
    swing_inertia_feedforward: bool = True
    normal_force_fraction: float = .85  # per-leg vertical force cap, fraction of total weight
    friction_fraction: float = .60

    # --- safety / reporting ------------------------------------------------
    finish_margin_m: float = .005
    finish_foot_height_m: float = .006
    ik_error_limit_m: float = .008
    torque_source: str = "d1_fast_side_step"


#: Default. Nominal budget for a 30 mm step: 4 planned shifts (~3.6-4.6 s total),
#: 4 x (0.12 unload + 0.26 lift + ~0.35 swing + 0.26 lower + ~0.10 probe +
#: 0.08 dwell + 0.15 load) ~= 5.3 s, plus ~0.9 s recenter.
FAST_PROFILE = GaitProfile()

#: In-module fallback that reproduces the conservative schedule inside this state
#: machine. It is *not* bit-identical to ``scripts/d1_side_step.py`` (the unload/
#: load waits are ramps and touchdown uses a slow probe); use the original module
#: when an exactly-as-validated gait is required.
CONSERVATIVE_PROFILE = replace(
    FAST_PROFILE,
    name="conservative",
    margin_gate_m=.024,
    adaptive_timing=False,
    fixed_shift_time_s=3.0,
    shift_settle_speed=.025,
    shift_extra_time_s=5.0,
    shift_correction_rate=.006,
    unload_time_s=.60,
    lift_time_s=1.0,
    adaptive_swing=False,
    fixed_swing_time_s=1.5,
    lower_time_s=1.20,
    lower_hover_m=.004,
    probe_speed=.004,
    probe_depth_m=.004,
    touchdown_dwell_s=.20,
    load_time_s=.80,
    load_settle_speed=.040,
    liftoff_extra_s=1.50,
    touchdown_extra_s=2.80,
    load_extra_s=2.00,
    abort_hold_dwell_s=.50,
    body_kp=20.,
    body_kd=8.,
    swing_kp=80.,
    swing_kd=3.,
    target_velocity_clip=1.5,
    swing_inertia_feedforward=False,
    torque_source="d1_fast_side_step",
)

PROFILES = {"fast": FAST_PROFILE, "conservative": CONSERVATIVE_PROFILE}


class FastSideStepController(SideStepController):
    """Experimental four-wheel lifting lateral step; +1 means initial body left.

    Body-relative: the commanded displacement is built from the yaw captured at
    ``start`` and that yaw is regulated for the whole cycle. There is no turn,
    drive or return substitution.
    """

    def __init__(self, plant, profile=FAST_PROFILE):
        if isinstance(profile, str):
            profile = PROFILES[profile]
        # set before super().__init__, which calls self.reset()
        self.cfg = profile
        super().__init__(plant)

    # ------------------------------------------------------------------ state

    def reset(self):
        """Discard controller memory; never reset or move the physical plant."""
        super().reset()
        cfg = self.cfg
        self.body_accel = np.zeros(3)
        self.load_weight = 1.
        self.loading_leg = None
        self.shift_time = float(cfg.fixed_shift_time_s)
        self.swing_time = float(cfg.fixed_swing_time_s)
        self.dynamic_margin = 0.
        self.com_height = .45
        self.margin_fault_time = 0.
        self.land_from = np.zeros(3)
        self.previous_velocity_target = np.zeros(self.plant.dof_addresses.size)
        self.lift_peaks = np.zeros(4)
        self.plan_error = 0.
        self.max_torque_fraction = 0.
        self.start_time = 0.

    @property
    def status(self):
        s = dict(super().status)
        s.update({
            "controller": "d1_fast_side_step",
            "profile": self.cfg.name,
            "torque_source": self.cfg.torque_source,
            "dynamic_margin_m": float(self.dynamic_margin),
            "load_weight": float(self.load_weight),
            "loading_leg": None if self.loading_leg is None else ("FL", "FR", "RL", "RR")[self.loading_leg],
            "elapsed_s": float(self.time - self.start_time),
            "shift_time_s": float(self.shift_time),
            "swing_time_s": float(self.swing_time),
            "lift_peaks_m": [float(v) for v in self.lift_peaks],
            "plan_error_m": float(self.plan_error),
            "max_torque_fraction": float(self.max_torque_fraction),
            "step_index": int(self.index),
        })
        return s

    # ------------------------------------------------------------------ entry

    def start(self, direction, distance_m=.03):
        cfg = self.cfg
        if isinstance(direction, (bool, np.bool_)) or not isinstance(direction, (int, np.integer)) or direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if not np.isfinite(distance_m) or not .01 <= distance_m <= .04:
            raise ValueError("distance_m must be within [0.01, 0.04]")
        if self.status["active"] or not self._contacts().all():
            return False
        if not (np.isfinite(self.plant.data.qpos).all() and np.isfinite(self.plant.data.qvel).all()):
            return False
        # This prototype assumes the local contact plane is world z=0.
        if np.max(np.abs(self._geometry(self.plant.measurement_data)[0][:, 2])) > .01:
            return False
        if np.max(np.abs(self.plant.base_rpy[:2])) > .12 or np.linalg.norm(self.plant.base_origin_velocity()) > .04:
            return False
        self.reset()
        self.direction = int(direction)
        travel = float(distance_m)*float(cfg.distance_scale)
        self.delta = direction*travel*np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.])
        if not np.isfinite(self.delta).all():
            return False
        # Same-side pair first: consecutive tripods share most of their support
        # polygon, so shifts 2 and 4 of the sequence are short.
        self.order = [2, 0, 3, 1] if direction > 0 else [3, 1, 2, 0]
        self.swing_time = self._swing_time()
        self.start_time = self.time
        self._prepare_leg()
        return True

    # ------------------------------------------------------------------ timing

    def _move_time(self, span):
        """Size ideal horizontal COM acceleration to the commanded ZMP budget.

        The nominal upper duration must not shorten a move below that budget.
        Actual stability is still checked from contact and body motion.
        """
        cfg = self.cfg
        if not cfg.adaptive_timing:
            return float(cfg.fixed_shift_time_s)
        distance = float(np.linalg.norm(np.asarray(span, dtype=float)[:2]))
        if not np.isfinite(distance):
            return float(cfg.shift_time_bounds[1])
        budget = max(float(cfg.zmp_budget_m), 1e-4)
        duration = np.sqrt(max(1e-9, SMOOTH_PEAK_ACCEL*distance*self.com_height/(G*budget)))
        return float(max(duration, cfg.shift_time_bounds[0]))

    def _swing_time(self):
        cfg = self.cfg
        if not cfg.adaptive_swing:
            return float(cfg.fixed_swing_time_s)
        distance = float(np.linalg.norm(self.delta[:2]))
        duration = 1.875*distance/max(float(cfg.swing_foot_speed), 1e-3)
        return float(np.clip(duration, cfg.swing_time_bounds[0], cfg.swing_time_bounds[1]))

    # ------------------------------------------------------------ kinematics

    def _ik(self, pose, quat, feet, leg, offset, outer, inner):
        """Solve stance IK on scratch data. Returns (qpos targets, error, points, com).

        Only ``self.scratch`` is written. The live plant data is read-only here.
        """
        d = self.scratch
        mujoco.mj_copyData(d, self.model, self.plant.measurement_data)
        d.qpos[:3] = pose
        d.qpos[3:7] = quat
        d.qvel[:] = 0.
        targets = np.asarray(feet, dtype=float).copy()
        if leg is not None and offset is not None:
            targets[leg] = targets[leg] + offset
        error_max = 0.
        for _ in range(outer):
            mujoco.mj_forward(self.model, d)
            _, extent = self._geometry(d)
            targets[:, 2] = extent - .001
            if leg is not None and offset is not None:
                targets[leg, 2] += offset[2]
            for index in range(4):
                for _ in range(inner):
                    mujoco.mj_forward(self.model, d)
                    error = targets[index] - d.xpos[self.bodies[index]]
                    if np.linalg.norm(error) < 1e-6:
                        break
                    mujoco.mj_jacBody(self.model, d, self.jp, self.jr, self.bodies[index])
                    j = self.jp[:, self.vadr[index, :3]]
                    step = j.T@np.linalg.solve(j@j.T + 1e-7*np.eye(3), error)
                    if not np.isfinite(step).all():
                        break
                    q = d.qpos[self.qadr[index, :3]] + np.clip(step, -.1, .1)
                    d.qpos[self.qadr[index, :3]] = np.clip(q, JOINT_POSITION_LOW[:3], JOINT_POSITION_HIGH[:3])
                mujoco.mj_forward(self.model, d)
                error_max = max(error_max, float(np.linalg.norm(targets[index] - d.xpos[self.bodies[index]])))
        mujoco.mj_forward(self.model, d)
        points, _ = self._geometry(d)
        return d.qpos[self.plant.qpos_addresses].copy(), error_max, points, self._com(d)

    def _targets(self):
        target, error, _, _ = self._ik(self.pose, self.quat, self.feet, self.leg, self.foot_offset,
                                       outer=3, inner=8)
        self.ik_error = error
        return target

    # ------------------------------------------------------------- planning

    def _prepare_leg(self):
        """Plan the body pose that puts the *predicted* tripod margin on target.

        The conservative controller projected into the polygon measured before
        the shift; hip abduction re-cambers the wheels while the body moves, so
        that polygon is wrong at the destination and the error had to be creeped
        out at 6 mm/s. Here the projection is iterated against geometry predicted
        by scratch IK at each candidate body pose.
        """
        cfg = self.cfg
        self.leg = self.order[self.index]
        self.loading_leg = None
        self.load_weight = 1.
        self.lift_peaks[self.leg] = -1.
        stance = np.delete(np.arange(4), self.leg)
        nominal = self.initial_pose + self.delta*(self.index/4.)
        xy = nominal[:2].copy()
        self.plan_error = 0.
        for _ in range(max(1, int(cfg.plan_iterations))):
            pose = nominal.copy()
            pose[:2] = xy
            _, error, points, com = self._ik(pose, self.quat, self.feet, None, None,
                                             outer=cfg.plan_ik_outer, inner=cfg.plan_ik_inner)
            self.plan_error = max(self.plan_error, error)
            boundary = _edges(points[stance, :2])
            if not boundary:
                break
            goal = com[:2].copy()
            for _ in range(24):
                for origin, normal in boundary:
                    goal += normal*max(0., cfg.margin_target_m - normal@(goal - origin))
            correction = goal - com[:2]
            if not np.isfinite(correction).all():
                self._abort("plan_nonfinite")
                return
            xy = xy + correction
            if np.linalg.norm(correction) < 5e-4:
                break
        offset = np.clip(xy - nominal[:2], -cfg.max_shift_m, cfg.max_shift_m)
        self.body_from = self.pose.copy()
        self.body_to = nominal.copy()
        self.body_to[:2] = nominal[:2] + offset
        if not np.isfinite(self.body_to).all():
            self._abort("plan_nonfinite")
            return
        self.shift_time = self._move_time(self.body_to - self.body_from)
        self.foot_offset[:] = 0.
        self.body_accel = np.zeros(3)
        self._enter("shift")

    # -------------------------------------------------------------- failure

    def _abort(self, reason):
        previous_leg, previous_phase = self.leg, self.phase
        super()._abort(reason)
        self.body_accel = np.zeros(3)
        self.margin_fault_time = 0.
        if self.leg is None and previous_leg is not None and previous_phase in ("shift", "unload"):
            # the leg never left the ground; ramp its share back in rather than stepping it
            self.loading_leg = previous_leg
        if self.leg is None and self.loading_leg is None:
            self.load_weight = 1.

    def _finish(self, success, failure):
        """Hand off only with four feet down, no pending swing and a positive margin."""
        contacts = self._contacts()
        points, _ = self._geometry(self.plant.measurement_data)
        airborne = (self.leg is not None
                    or float(np.max(np.abs(self.foot_offset))) > 1e-9
                    or not bool(contacts.all())
                    or float(np.max(points[:, 2])) > self.cfg.finish_foot_height_m)
        if airborne or self.margin < self.cfg.finish_margin_m or self.missing_support_time > 0.:
            self._abort(failure if failure is not None else "unsafe_completion")
            return False
        self.loading_leg = None
        self.load_weight = 1.
        self.success = bool(success)
        self.failure = None if success else failure
        self.done = True
        self._enter("done" if success else "failed")
        return True

    def _weights(self):
        w = np.ones(4)
        if self.leg is not None and self.phase in _SWING_PHASES:
            w[self.leg] = self.load_weight
        elif self.loading_leg is not None:
            w[self.loading_leg] = self.load_weight
        return np.clip(w, 0., 1.)

    # -------------------------------------------------------- state machine

    def _advance(self):
        cfg = self.cfg
        self.time += self.dt
        self.phase_time += self.dt
        contacts = self._contacts()
        d = self.plant.measurement_data
        points, _ = self._geometry(d)
        com = self._com(d)
        stance = np.arange(4) if self.leg is None else np.delete(np.arange(4), self.leg)
        self.com_height = float(np.clip(com[2] - float(np.min(points[:, 2])), .10, 1.))
        boundary = _edges(points[stance, :2])
        zmp = com[:2] - (self.com_height/G)*self.body_accel[:2]
        self.margin = min((n@(com[:2] - a) for a, n in boundary), default=-1.)
        self.dynamic_margin = min((n@(zmp - a) for a, n in boundary), default=-1.)
        if self.leg is not None:
            self.lift_peaks[self.leg] = max(self.lift_peaks[self.leg], float(points[self.leg, 2]))
        speed = float(np.linalg.norm(self.plant.base_origin_velocity()))

        if self.status["active"] and (np.max(np.abs(self.plant.base_rpy[:2])) > .32 or self.plant.base_position[2] < .32):
            self._abort("body_pose_limit")
        if self.status["active"] and np.max(np.abs(d.qvel[self.vadr[:, :3]])) > 18.:
            self._abort("joint_velocity_limit")
        if self.phase in ("lift", "swing", "lower"):
            self.missing_support_time = self.missing_support_time + self.dt if not contacts[stance].all() else 0.
            if self.missing_support_time > .12:
                self._abort("stance_contact_lost")
            self.margin_fault_time = self.margin_fault_time + self.dt if self.dynamic_margin < cfg.margin_fault_m else 0.
            if self.margin_fault_time > cfg.margin_fault_time_s:
                self._abort("support_margin_lost")
        else:
            self.missing_support_time = 0.
            self.margin_fault_time = 0.

        phase, t = self.phase, self.phase_time

        if phase in ("shift", "recenter"):
            duration = max(self.shift_time, 1e-3)
            fraction = min(1., t/duration)
            span = self.body_to - self.body_from
            self.pose = self.body_from + _smooth(fraction)*span
            self.body_accel = span*(_smooth_ddot(fraction)/(duration*duration)) if fraction < 1. else np.zeros(3)

        if phase == "shift":
            ready = (self.dynamic_margin > cfg.margin_gate_m and contacts.all()
                     and speed < cfg.shift_settle_speed)
            if t >= self.shift_time and ready:
                self._enter("unload")
            elif t >= self.shift_time and t <= self.shift_time + cfg.shift_extra_time_s:
                # Guard only: the planner should have landed the margin already.
                if self.margin < cfg.margin_target_m and boundary:
                    _, inward = min(boundary, key=lambda edge: edge[1]@(com[:2] - edge[0]))
                    self.body_to[:2] += inward*cfg.shift_correction_rate*self.dt
            elif t > self.shift_time + cfg.shift_extra_time_s:
                self._abort("support_shift_timeout")

        elif phase == "unload":
            self.body_accel = np.zeros(3)
            self.load_weight = float(1. - _smooth(t/max(cfg.unload_time_s, 1e-6)))
            if t >= cfg.unload_time_s:
                self.load_weight = 0.
                self._enter("lift")

        elif phase == "lift":
            self.load_weight = 0.
            self.foot_offset[2] = cfg.lift_height_m*_smooth(t/max(cfg.lift_time_s, 1e-6))
            if t >= cfg.lift_time_s and points[self.leg, 2] > cfg.lift_clearance_m and not contacts[self.leg]:
                self._enter("swing")
            elif t > cfg.lift_time_s + cfg.liftoff_extra_s:
                self._abort("liftoff_timeout")

        elif phase == "swing":
            self.load_weight = 0.
            self.foot_offset = self.delta*_smooth(t/max(self.swing_time, 1e-6)) + np.array([0., 0., cfg.lift_height_m])
            if t >= self.swing_time:
                self._enter("lower")

        elif phase in ("lower", "abort_land"):
            self.load_weight = 0.
            if phase == "lower":
                lateral, start_z = self.delta[:2], cfg.lift_height_m
            else:
                lateral, start_z = self.land_from[:2], float(self.land_from[2])
            hover = min(cfg.lower_hover_m, max(0., .25*start_z))
            if t <= cfg.lower_time_s:
                height = start_z - (start_z - hover)*_smooth(t/max(cfg.lower_time_s, 1e-6))
            else:
                # Constant-velocity probe: a smoothstep arrives with zero closing
                # speed and then waits to be noticed. This bounds impact speed at
                # probe_speed instead.
                height = max(hover - cfg.probe_speed*(t - cfg.lower_time_s), -cfg.probe_depth_m)
            offset = np.zeros(3)
            offset[:2] = lateral
            offset[2] = height
            self.foot_offset = offset
            if contacts[self.leg] and t > .4*cfg.lower_time_s:
                self.contact_dwell += self.dt
            else:
                self.contact_dwell = 0.
            if self.contact_dwell >= cfg.touchdown_dwell_s:
                self.feet[self.leg, :2] += self.foot_offset[:2]
                self.foot_offset[:] = 0.
                self.loading_leg = self.leg
                self.leg = None
                self.load_weight = 0.
                self._enter("abort_hold" if self.failure else "load")
            elif t > cfg.lower_time_s + cfg.touchdown_extra_s:
                self._abort("touchdown_timeout")

        elif phase == "load":
            self.load_weight = float(_smooth(t/max(cfg.load_time_s, 1e-6)))
            if t >= cfg.load_time_s and contacts.all() and speed < cfg.load_settle_speed:
                self.load_weight = 1.
                self.loading_leg = None
                self.index += 1
                if self.index < 4:
                    self._prepare_leg()
                else:
                    self.body_from = self.pose.copy()
                    self.body_to = self.initial_pose + self.delta
                    self.shift_time = self._move_time(self.body_to - self.body_from)
                    self._enter("recenter")
            elif t > cfg.load_time_s + cfg.load_extra_s:
                self._abort("load_timeout")

        elif phase == "abort_hold":
            self.load_weight = min(1., self.load_weight + self.dt/max(cfg.load_time_s, 1e-3))
            if contacts.all() and speed < cfg.abort_hold_speed:
                self.contact_dwell += self.dt
                if self.contact_dwell > cfg.abort_hold_dwell_s:
                    self._finish(False, self.failure or "aborted")
            else:
                self.contact_dwell = 0.

        if self.phase == "recenter":
            if self.phase_time >= self.shift_time and contacts.all() and speed < cfg.recenter_settle_speed:
                error = self.plant.base_position - (self.initial_pose + self.delta)
                yaw_error = _wrap(self.plant.base_rpy[2] - self.yaw)
                ok = bool(np.linalg.norm(error[:2]) < cfg.final_position_tolerance_m
                          and abs(yaw_error) < cfg.final_yaw_tolerance_rad)
                if ok:
                    self._finish(True, None)
                elif self.phase_time > self.shift_time + cfg.recenter_extra_time_s:
                    self._abort("final_tracking_error")
            elif self.phase_time > self.shift_time + cfg.recenter_extra_time_s:
                self._abort("recenter_timeout")

    # -------------------------------------------------------------- torques

    def _allocate(self, points, com, wrench, weights):
        """Weighted-regularised stance force allocation.

        A leg with weight w carries a Tikhonov penalty proportional to 1/w^2, so
        w -> 0 removes it from the solution continuously. This replaces the
        conservative 0.6 s unload / 0.8 s load dead waits with a force ramp of
        the same physical effect and a tenth of the duration.
        """
        cfg = self.cfg
        allocation = np.zeros((6, 12))
        for leg in range(4):
            x, y, z = points[leg] - com
            allocation[:3, 3*leg:3*leg + 3] = np.eye(3)
            allocation[3:, 3*leg:3*leg + 3] = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]
        penalty = np.repeat(cfg.force_regularization/np.maximum(weights, 1e-3)**2, 3)
        normal = allocation.T@allocation + np.diag(penalty)
        try:
            forces = np.linalg.solve(normal, allocation.T@wrench)
        except np.linalg.LinAlgError:
            forces = np.linalg.lstsq(allocation, wrench, rcond=None)[0]
        if not np.isfinite(forces).all():
            self._abort("nonfinite_allocation")
            forces = np.zeros(12)
        forces = forces.reshape(4, 3)
        cap = cfg.normal_force_fraction*self.mass*G
        for leg in range(4):
            forces[leg, 2] = np.clip(forces[leg, 2], 0., weights[leg]*cap)
            bound = cfg.friction_fraction*forces[leg, 2]
            forces[leg, :2] = np.clip(forces[leg, :2], -bound, bound)
        return forces

    def compute(self):
        cfg = self.cfg
        plant = self.plant
        if not (np.isfinite(plant.data.qpos).all() and np.isfinite(plant.data.qvel).all()):
            self.failure, self.phase = "nonfinite_state", "abort_hold"
            self.done, self.success = False, False
            return np.zeros(self.plant.dof_addresses.size)

        self._advance()
        d = plant.measurement_data
        target = self._targets()
        if not np.isfinite(target).all():
            self._abort("nonfinite_target")
            target = self.previous_target.copy()
        if self.ik_error > cfg.ik_error_limit_m:
            self._abort("ik_unreachable")

        velocity_target = np.clip((target - self.previous_target)/self.dt,
                                  -cfg.target_velocity_clip, cfg.target_velocity_clip)
        accel_target = np.clip((velocity_target - self.previous_velocity_target)/self.dt,
                               -cfg.target_accel_clip, cfg.target_accel_clip)
        self.previous_target = target.copy()
        self.previous_velocity_target = velocity_target.copy()
        velocity = d.qvel[plant.dof_addresses]
        points, _ = self._geometry(d)
        com = self._com(d)
        linear, angular = plant.base_velocity(local=False)

        force = self.mass*(np.array([0., 0., G]) + self.body_accel
                           + cfg.body_kp*(self.pose - plant.base_position) - cfg.body_kd*linear)
        rpy = plant.base_rpy
        error_rpy = np.array([-rpy[0], -rpy[1], _wrap(self.yaw - rpy[2])])
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        heading_rotation = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
        moment = cfg.attitude_kp*(heading_rotation@error_rpy) - cfg.attitude_kd*angular

        weights = self._weights()
        forces = self._allocate(points, com, np.r_[force, moment], weights)
        tau = d.qfrc_bias[plant.dof_addresses].copy()
        for leg in range(4):
            if weights[leg] <= 1e-3:
                continue
            mujoco.mj_jac(self.model, d, self.jp, self.jr, points[leg], self.bodies[leg])
            tau -= self.jp[:, plant.dof_addresses].T@forces[leg]

        size = tau.size
        kp = np.full(size, cfg.stance_kp)
        kd = np.full(size, cfg.stance_kd)
        swinging = self.leg is not None and self.phase in _SWING_PHASES
        if swinging:
            kp[4*self.leg:4*self.leg + 3] = cfg.swing_kp
            kd[4*self.leg:4*self.leg + 3] = cfg.swing_kd
        tau += kp*(target - d.qpos[plant.qpos_addresses]) + kd*(velocity_target - velocity)

        if cfg.swing_inertia_feedforward and swinging and self.phase != "unload":
            inertia = _dense_inertia(self.model, self.scratch)
            dofs = self.vadr[self.leg, :3]
            feedforward = inertia[np.ix_(dofs, dofs)]@accel_target[4*self.leg:4*self.leg + 3]
            if np.isfinite(feedforward).all():
                tau[4*self.leg:4*self.leg + 3] += feedforward

        # Position brake keeps rolling wheels from defeating fixed stance anchors.
        wheel = np.arange(3, size, 4)
        tau[wheel] += cfg.wheel_brake_kp*(self.wheel_angles - d.qpos[self.qadr[:, 3]]) - cfg.wheel_brake_kd*velocity[wheel]
        tau[wheel] -= kp[wheel]*(target[wheel] - d.qpos[self.qadr[:, 3]]) + kd[wheel]*(velocity_target[wheel] - velocity[wheel])

        if not np.isfinite(tau).all():
            self._abort("nonfinite_torque")
            tau = np.zeros(size)
        limit = np.abs(np.asarray(plant.actuator_torque_limit_nm, dtype=float))
        tau = np.clip(tau, -limit, limit)
        if self.status["active"] or self.done:
            self.max_torque_fraction = max(self.max_torque_fraction,
                                           float(np.max(np.abs(tau)/np.maximum(limit, 1e-9))))
        return tau


__all__ = ["CONSERVATIVE_PROFILE", "FAST_PROFILE", "PROFILES", "FastSideStepController", "GaitProfile"]
