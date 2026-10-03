"""Pure C35 held-command, body reference and diagonal-pair phase machine.

No plant, MuJoCo, torque, model, clock or I/O lives here. A Reference35 is a
request, never physical safety proof; the adapter owns every native gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


PAIRS35 = ((0, 3), (1, 2))  # FL+RR, FR+RL; mapping checked by adapter
DT35 = .01
RAMP35 = .25
LANDING_CAP35 = .060
BODY_KP35 = 26.
BODY_KD35 = 9.


def smooth35(u: float) -> float:
    u = min(1., max(0., u))
    return u**3*(10.-15.*u+6.*u*u)


def smooth_derivative35(u: float) -> float:
    u = min(1., max(0., u))
    return 30.*u*u*(1.-u)**2


def smooth_integral35(u: float) -> float:
    u = min(1., max(0., u))
    return 2.5*u**4-3.*u**5+u**6


def hermite_velocity35(v0: float, a0: float, j0: float, v1: float):
    """Quintic velocity: preserve v/a/jerk at start, end at v1/0/0."""
    c0,c1,c2 = v0,a0*RAMP35,j0*RAMP35**2/2.
    d = v1-c0-c1-c2
    e = -c1-2.*c2
    f = -2.*c2
    return (c0,c1,c2,10.*d-4.*e+.5*f,
            -15.*d+7.*e-f,6.*d-3.*e+.5*f)


def velocity_sample35(coefficients, elapsed_s: float):
    """Analytic velocity, acceleration and jerk at elapsed ramp time."""
    u = min(1.,max(0.,elapsed_s/RAMP35))
    c = coefficients
    velocity = sum(c[k]*u**k for k in range(6))
    accel = sum(k*c[k]*u**(k-1) for k in range(1,6))/RAMP35
    jerk = sum(k*(k-1)*c[k]*u**(k-2) for k in range(2,6))/RAMP35**2
    return velocity,accel,jerk


def velocity_integral35(coefficients, elapsed0_s: float, elapsed1_s: float):
    """Exact displacement during a ramp interval, including flat remainder."""
    u0 = min(1.,max(0.,elapsed0_s/RAMP35))
    u1 = min(1.,max(0.,elapsed1_s/RAMP35))
    within = RAMP35*sum(coefficients[k]/(k+1)*(u1**(k+1)-u0**(k+1))
                          for k in range(6))
    beyond = sum(coefficients)
    return within+beyond*max(0.,elapsed1_s-max(elapsed0_s,RAMP35))


def landing_xy35(body_xy, layout_xy, vref_xy, vcom_xy, foot_xy):
    """Return raw/projected landing and radial-cap flag; never hide proposal."""
    raw = tuple(body_xy[j]+layout_xy[j]+.975*vref_xy[j]
                +.08*(vcom_xy[j]-vref_xy[j]) for j in range(2))
    dx, dy = raw[0]-foot_xy[0], raw[1]-foot_xy[1]
    distance = math.hypot(dx, dy)
    if distance <= LANDING_CAP35:
        return raw, raw, False
    factor = LANDING_CAP35/distance
    return raw, (foot_xy[0]+factor*dx, foot_xy[1]+factor*dy), True


def body_force_xy35(mass, reference: Reference35, actual_xy, actual_vxy):
    """Correct velocity-feedback XY force; adapter adds gravity/Z/moments."""
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('mass must be positive and finite')
    return tuple(mass*(reference.body_axy_mps2[j]
        +BODY_KP35*(reference.body_xy_m[j]-actual_xy[j])
        +BODY_KD35*(reference.body_vxy_mps[j]-actual_vxy[j])) for j in range(2))


@dataclass(frozen=True)
class Command35:
    vy_mps: float  # exact allowed raw values: -.025, 0, +.025
    gait_enabled: bool
    issued_control_index: int
    expires_control_index: int
    stop_kind: str  # none, release, cancel, expired, conflict


@dataclass(frozen=True)
class Sensed35:
    control_index: int
    body_xy_m: tuple[float, float]
    body_com_vxy_mps: tuple[float, float]
    body_origin_vxyz_mps: tuple[float, float, float]
    yaw_rad: float
    foot_world_m: tuple[tuple[float, float, float], ...]  # actual wheel BODY centers, FL/FR/RL/RR
    wheel_contact_points_world_m: tuple[tuple[float, float, float], ...]
    actual_whole_wheel_min_z_m: tuple[float, float, float, float]
    previous_five_contact_free: tuple[bool, bool, bool, bool]
    previous_five_pair_loaded: tuple[bool, bool]
    foot_contact: tuple[bool, bool, bool, bool]
    normal_load_n: tuple[float, float, float, float]
    mass_kg: float
    whole_com_z_m: float
    base_z_m: float
    static_support_margin_m: float
    world_angular_speed_rps: float
    native_safety_ok: bool  # adapter's ALL native checks, including pair gate


@dataclass(frozen=True)
class Reference35:
    phase: str
    pair: tuple[int, int] | None
    body_xy_m: tuple[float, float]
    body_vxy_mps: tuple[float, float]
    body_axy_mps2: tuple[float, float]
    body_jerk_xy_mps3: tuple[float, float]
    ramp_coefficients_mps: tuple[float, float, float, float, float, float]
    ramp_elapsed_s: float
    feet_world_m: tuple[tuple[float, float, float], ...]
    feet_velocity_world_mps: tuple[tuple[float, float, float], ...]
    force_weights: tuple[float, float, float, float]
    foot_target_clipped: tuple[bool, bool, bool, bool]
    landing_raw_xy_m: tuple[tuple[float, float] | None, ...]
    landing_projected_xy_m: tuple[tuple[float, float] | None, ...]
    phase_elapsed_s: float
    restore_start_weight: float | None
    stop_reason: str | None
    ready_for_handoff: bool
    safe_abort_required: bool
    fault: str | None


class Core35:
    """One physical control per step; caller supplies fresh measured evidence."""

    def __init__(self):
        self.phase = 'idle'
        self.elapsed = 0.
        self.last_control = None
        self.pair_index = 0
        self.stop_requested = False
        self.stop_reason = None
        self.active_sign = 0
        self.stop_controls = 0
        self.stable_controls = 0
        self.completed_pair_exchanges = 0
        self.yaw = 0.
        self.left = (0., 1.)
        self.body_ref = (0., 0.)
        self.vref = 0.
        self.aref = 0.
        self.jref = 0.
        self.ramp_target = 0.
        self.ramp_elapsed = RAMP35
        self.ramp_coefficients = (0.,)*6
        self.anchors = ((0., 0., 0.),)*4
        self.layout_xy = ((0., 0.),)*4
        self.swing_from = ((0., 0., 0.),)*4
        self.swing_to = ((0., 0., 0.),)*4
        self.swing_duration = .08
        self.dwell = 0.
        self.touchdown_wait = 0.
        self.restore_start_weight = None
        self.clipped = (False,)*4
        self.raw_xy = (None,)*4
        self.projected_xy = (None,)*4
        self.fault = None

    @staticmethod
    def _check(command: Command35, sensed: Sensed35):
        if (type(sensed.control_index) is not int or sensed.control_index < 0
                or type(command.issued_control_index) is not int
                or type(command.expires_control_index) is not int
                or type(command.gait_enabled) is not bool
                or command.vy_mps not in (-.025, 0., .025)
                or command.stop_kind not in ('none','release','cancel',
                                             'conflict','expired')
                or len(sensed.foot_world_m) != 4
                or len(sensed.wheel_contact_points_world_m) != 4
                or len(sensed.actual_whole_wheel_min_z_m) != 4
                or len(sensed.previous_five_contact_free) != 4
                or len(sensed.previous_five_pair_loaded) != 2
                or len(sensed.foot_contact) != 4
                or len(sensed.normal_load_n) != 4):
            raise ValueError('C35 command/evidence shape or value differs')
        values = (*sensed.body_xy_m,*sensed.body_com_vxy_mps,
                  *sensed.body_origin_vxyz_mps,
                  sensed.yaw_rad,sensed.mass_kg,sensed.whole_com_z_m,
                  sensed.base_z_m,sensed.static_support_margin_m,
                  sensed.world_angular_speed_rps,*sensed.normal_load_n,
                  *(v for foot in sensed.foot_world_m for v in foot),
                  *(v for point in sensed.wheel_contact_points_world_m for v in point),
                  *sensed.actual_whole_wheel_min_z_m)
        if not all(math.isfinite(v) for v in values) or sensed.mass_kg <= 0:
            raise ValueError('C35 sensed values nonfinite or mass nonpositive')

    @staticmethod
    def _valid(command: Command35, index: int) -> bool:
        return (command.issued_control_index <= index
                < command.expires_control_index
                and index-command.issued_control_index < 20
                and command.stop_kind not in ('conflict','expired'))

    @staticmethod
    def _all_loaded(sensed: Sensed35) -> bool:
        return (all(sensed.foot_contact[j] and sensed.normal_load_n[j] > 1e-8
                    for j in range(4))
                and sum(sensed.normal_load_n) >= .5*sensed.mass_kg*9.81)

    @staticmethod
    def _stance_loaded(sensed: Sensed35, pair: tuple[int,int]) -> bool:
        stance = tuple(j for j in range(4) if j not in pair)
        return (all(sensed.foot_contact[j] and sensed.normal_load_n[j] > 1e-8
                    for j in stance)
                and sum(sensed.normal_load_n[j] for j in stance)
                    >= .5*sensed.mass_kg*9.81)

    def _enter(self, phase: str):
        self.phase = phase
        self.elapsed = 0.
        if phase != 'dwell':
            self.dwell = 0.
        if phase == 'probe':
            self.touchdown_wait = 0.

    def _begin(self, sensed: Sensed35):
        if not self._all_loaded(sensed):
            raise RuntimeError('C35 start needs four actual loaded wheels')
        self.yaw = sensed.yaw_rad
        self.left = (-math.sin(self.yaw), math.cos(self.yaw))
        self.body_ref = sensed.body_xy_m
        self.anchors = tuple(sensed.foot_world_m)
        self.layout_xy = tuple((foot[0]-self.body_ref[0],
                                foot[1]-self.body_ref[1]) for foot in self.anchors)
        self.pair_index = 0
        self._enter('transfer')

    def _advance_body(self, target_vy: float):
        if target_vy != self.ramp_target:
            self.ramp_coefficients = hermite_velocity35(
                self.vref,self.aref,self.jref,target_vy)
            self.ramp_target = target_vy
            self.ramp_elapsed = 0.
        t0 = self.ramp_elapsed
        t1 = t0+DT35
        distance = velocity_integral35(self.ramp_coefficients,t0,t1)
        self.ramp_elapsed = t1
        if t1 >= RAMP35:
            self.vref,self.aref,self.jref = self.ramp_target,0.,0.
        else:
            self.vref,self.aref,self.jref = velocity_sample35(
                self.ramp_coefficients,t1)
        self.body_ref = (self.body_ref[0]+self.left[0]*distance,
                         self.body_ref[1]+self.left[1]*distance)

    def _reference(self) -> Reference35:
        pair = PAIRS35[self.pair_index] if self.phase not in ('idle','done') else None
        feet = list(self.anchors)
        foot_velocities = [(0.,0.,0.) for _ in range(4)]
        weights = [1.]*4
        if pair is not None and self.phase == 'transfer':
            for leg in pair:
                weights[leg] = 1.-smooth35(self.elapsed/.06)
        if pair is not None and self.phase == 'transfer_restore':
            for leg in pair:
                weights[leg] = (self.restore_start_weight
                    +(1.-self.restore_start_weight)*smooth35(self.elapsed/.06))
        if pair is not None and self.phase in (
                'lift','horizontal','lower','probe','dwell','load'):
            for leg in pair:
                if self.phase == 'lift':
                    foot = self.swing_from[leg]
                    feet[leg] = (foot[0],foot[1],
                                 foot[2]+.035*smooth35(self.elapsed/.12))
                    foot_velocities[leg] = (0.,0.,
                        .035/.12*smooth_derivative35(self.elapsed/.12))
                elif self.phase == 'horizontal':
                    start,target = self.swing_from[leg],self.swing_to[leg]
                    s = smooth35(self.elapsed/self.swing_duration)
                    feet[leg] = (start[0]+(target[0]-start[0])*s,
                                 start[1]+(target[1]-start[1])*s,start[2]+.035)
                    rate = smooth_derivative35(self.elapsed/self.swing_duration)/self.swing_duration
                    foot_velocities[leg] = ((target[0]-start[0])*rate,
                                            (target[1]-start[1])*rate,0.)
                elif self.phase == 'lower':
                    target = self.swing_to[leg]
                    feet[leg] = (target[0],target[1],
                        self.swing_from[leg][2]+.035-.038*smooth35(self.elapsed/.12))
                    foot_velocities[leg] = (0.,0.,
                        -.038/.12*smooth_derivative35(self.elapsed/.12))
                elif self.phase == 'probe':
                    target = self.swing_to[leg]
                    feet[leg] = (target[0],target[1],
                        self.swing_from[leg][2]-.003-min(.012,.03*self.elapsed))
                    foot_velocities[leg] = (0.,0.,-.03 if self.elapsed < .4 else 0.)
                else:  # dwell/load, use actual measured touchdown anchor
                    feet[leg] = self.swing_to[leg]
                weights[leg] = (smooth35(self.elapsed/.06)
                                if self.phase == 'load' else 0.)
        return Reference35(self.phase,pair,self.body_ref,
            (self.left[0]*self.vref,self.left[1]*self.vref),
            (self.left[0]*self.aref,self.left[1]*self.aref),
            (self.left[0]*self.jref,self.left[1]*self.jref),
            self.ramp_coefficients,self.ramp_elapsed,
            tuple(feet),tuple(foot_velocities),tuple(weights),self.clipped,self.raw_xy,
            self.projected_xy,self.elapsed,self.restore_start_weight,
            self.stop_reason,
            self.phase == 'done',self.phase == 'abort_required',self.fault)

    def step(self, command: Command35, sensed: Sensed35) -> Reference35:
        self._check(command,sensed)
        if self.last_control is not None and sensed.control_index != self.last_control+1:
            raise ValueError('C35 control indices must be contiguous')
        self.last_control = sensed.control_index
        if self.phase in ('done','abort_required'):
            return self._reference()
        valid = self._valid(command,sensed.control_index)
        if self.phase == 'idle':
            if valid and (command.vy_mps or command.gait_enabled):
                self._begin(sensed)
                self.active_sign = (1 if command.vy_mps > 0 else
                                    -1 if command.vy_mps < 0 else 0)
            else:
                return self._reference()
        if not sensed.native_safety_ok:
            self.fault = 'native_safety_failed'
            self._enter('abort_required')
            return self._reference()
        if math.hypot(self.body_ref[0]-sensed.body_xy_m[0],
                      self.body_ref[1]-sensed.body_xy_m[1]) > .060:
            self.fault = 'body_reference_tracking_error_gt_60mm'
            self._enter('abort_required')
            return self._reference()
        reason = (command.stop_kind if command.stop_kind in ('conflict','expired')
                  else command.stop_kind if valid else 'expired')
        reversal = (self.active_sign != 0 and command.vy_mps*self.active_sign < 0)
        if (not valid or (command.vy_mps == 0. and not command.gait_enabled)
                or reason in ('release','cancel','conflict','expired') or reversal):
            if not self.stop_requested:
                self.stop_reason = ('reverse_wait' if reversal else
                                    'release' if reason == 'none' else reason)
            self.stop_requested = True
        if self.phase == 'transfer' and self.completed_pair_exchanges >= 16:
            if not self.stop_requested:
                self.stop_reason = 'pair_cap_16'
            self.stop_requested = True
        desired = 0. if self.stop_requested else command.vy_mps
        if self.stop_requested:
            self.stop_controls += 1
            if self.stop_controls > 250:
                self.fault = 'stop_controls_gt_250'
                self._enter('abort_required')
                return self._reference()
        self._advance_body(desired)
        if math.hypot(self.body_ref[0]-sensed.body_xy_m[0],
                      self.body_ref[1]-sensed.body_xy_m[1]) > .060:
            self.fault = 'current_body_reference_tracking_error_gt_60mm'
            self._enter('abort_required')
            return self._reference()
        self.elapsed += DT35
        pair = PAIRS35[self.pair_index]
        if self.phase in ('lift','horizontal','lower','probe','dwell'):
            if not self._stance_loaded(sensed,pair):
                self.fault = 'stance_pair_native_contact_or_load_failed'
                self._enter('abort_required')
                return self._reference()
        if self.phase == 'transfer':
            if self.stop_requested and self._all_loaded(sensed):
                self.restore_start_weight = 1.-smooth35(
                    max(0.,self.elapsed-DT35)/.06)
                self._enter('transfer_restore')
            elif (self.elapsed >= .06 and self._all_loaded(sensed)
                    and self._stance_loaded(sensed,pair)
                    and sensed.previous_five_pair_loaded[self.pair_index]):
                self.swing_from = self.anchors
                self._enter('lift')
            elif self.elapsed > .46:
                self.fault = 'transfer_pair_load_timeout'
                self._enter('abort_required')
        elif self.phase == 'transfer_restore':
            if not self._all_loaded(sensed):
                self.fault = 'transfer_restore_all4_load_failed'
                self._enter('abort_required')
            elif self.elapsed >= .06:
                self._enter('settle')
        elif self.phase == 'lift':
            if (self.elapsed >= .12 and all(
                    sensed.actual_whole_wheel_min_z_m[j] > .012 and
                    sensed.previous_five_contact_free[j] and
                    not sensed.foot_contact[j] for j in pair)):
                targets = list(self.anchors)
                raw_list = [None]*4
                projected_list = [None]*4
                clipped = [False]*4
                vxy = (self.left[0]*self.vref,self.left[1]*self.vref)
                largest = 0.
                for leg in pair:
                    foot = self.swing_from[leg]
                    raw,projected,clipped[leg] = landing_xy35(self.body_ref,
                        self.layout_xy[leg],vxy,sensed.body_com_vxy_mps,foot[:2])
                    raw_list[leg],projected_list[leg] = raw,projected
                    largest = max(largest,math.hypot(projected[0]-foot[0],
                                                      projected[1]-foot[1]))
                    targets[leg] = (*projected,foot[2])
                self.swing_to = tuple(targets)
                self.raw_xy = tuple(raw_list)
                self.projected_xy = tuple(projected_list)
                self.clipped = tuple(clipped)
                self.swing_duration = max(.08,1.875*largest/.16)
                self._enter('horizontal')
            elif self.elapsed > .12+.4:
                self.fault = 'liftoff_clearance_timeout'
                self._enter('abort_required')
        elif self.phase == 'horizontal':
            if not all(sensed.actual_whole_wheel_min_z_m[j] > .012 and
                       sensed.previous_five_contact_free[j] and
                       not sensed.foot_contact[j] for j in pair):
                self.fault = 'horizontal_native_clearance_or_contact_lost'
                self._enter('abort_required')
            elif self.elapsed >= self.swing_duration:
                self._enter('lower')
        elif self.phase == 'lower':
            if self.elapsed >= .12:
                self._enter('probe')
        elif self.phase == 'probe':
            self.touchdown_wait += DT35
            if all(sensed.foot_contact[j] and sensed.normal_load_n[j] > 1e-8
                   for j in pair):
                self.swing_to = tuple((foot[0],foot[1],sensed.foot_world_m[j][2])
                    if j in pair else foot for j,foot in enumerate(self.swing_to))
                self._enter('dwell')
            elif self.elapsed >= .012/.03 or self.touchdown_wait > .6:
                self.fault = 'touchdown_probe_depth_or_time_limit'
                self._enter('abort_required')
        elif self.phase == 'dwell':
            self.touchdown_wait += DT35
            if all(sensed.foot_contact[j] and sensed.normal_load_n[j] > 1e-8
                   for j in pair):
                self.dwell += DT35
                if self.dwell >= .08:
                    self._enter('load')
            else:
                self.dwell = 0.
            if self.touchdown_wait > .6:
                self.fault = 'touchdown_dwell_timeout'
                self._enter('abort_required')
        elif self.phase == 'load':
            if self.elapsed >= .06 and self._all_loaded(sensed):
                self.completed_pair_exchanges += 1
                if self.completed_pair_exchanges > 16:
                    self.fault = 'pair_exchanges_gt_16'
                    self._enter('abort_required')
                    return self._reference()
                updated = list(self.anchors)
                for leg in pair:
                    updated[leg] = sensed.foot_world_m[leg]
                self.anchors = tuple(updated)
                self.clipped = (False,)*4
                self.raw_xy = (None,)*4
                self.projected_xy = (None,)*4
                if self.stop_requested:
                    self._enter('settle')
                else:
                    self.pair_index = 1-self.pair_index
                    self._enter('transfer')
        elif self.phase == 'settle':
            if (self._all_loaded(sensed) and abs(self.vref) <= 1e-12
                    and math.sqrt(sum(v*v for v in sensed.body_origin_vxyz_mps)) <= .01
                    and sensed.world_angular_speed_rps <= .05
                    and sensed.static_support_margin_m >= .005):
                self.stable_controls += 1
                if self.stable_controls >= 50:
                    self._enter('done')
            else:
                self.stable_controls = 0
        return self._reference()
