"""Switch course torque ownership only after a side step has safely landed.

The side controller uses simulator truth on flat ground. It is separate from
the frozen learning experiment and never resets or moves the live plant.
"""

from __future__ import annotations

import numpy as np

from wheel_legged_control.d1.control_loop import D1MotionCommand
from wheel_legged_control.d1.interactive import D1TeleopStatus

if __package__:
    from .d1_course_braking import CourseBrakeReference
    from .d1_side_step import SideStepController
else:
    from d1_course_braking import CourseBrakeReference
    from d1_side_step import SideStepController


class CourseSideStepDrive:
    """Own exactly one torque computation and plant integration per call.

    ``direction`` is +1 for body left, -1 for right, and 0 for release.
    Releasing or reversing during a step requests landing, not an immediate
    controller switch. Call ``reset`` after an explicit external zone reset;
    calling it during flight would discard the landing controller's memory.
    """

    def __init__(self, simulation, *, side_controller_factory=SideStepController,
                 brake_reference_mode="legacy"):
        if brake_reference_mode not in ("legacy", "release_reanchor_experimental"):
            raise ValueError("unsupported course brake reference mode")
        self.simulation = simulation
        self.plant = simulation.plant
        self.teleop = simulation.teleop
        self.side = side_controller_factory(self.plant)
        self.braking = CourseBrakeReference()
        self.brake_reference_mode = brake_reference_mode
        self._active = False
        self._blocked_until_release = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        """Clear adapter memory after the caller has reset the simulation."""
        self.side.reset()
        self.braking.reset()
        self._active = False
        self._blocked_until_release = False

    def _prepare_legacy_handoff(self) -> None:
        # Keep the state source and its time/sequence continuous. These are
        # controller memories only; teleop.reset() would teleport the plant.
        teleop = self.teleop
        teleop.controller.reset()
        teleop.terrain_attitude.reset()
        teleop._last_torque[:] = 0.0
        teleop._previous_residual_action[:] = 0.0
        teleop._stuck_steps = 0
        teleop._boost_steps_remaining = 0
        teleop.controller.longitudinal_force_limit_n = 180.0
        # A side-owned interval cannot be the previous wheel-driving command.
        self.braking.observe_applied(0.0, eligible=False)

    def _side_teleop_status(self, torque) -> D1TeleopStatus:
        """Describe this controller without reusing legacy force diagnostics.

        Forward/yaw are zero tracking requests; lateral displacement and the
        actual measured pose belong to the separate side status/plant record.
        None denotes a diagnostic this controller does not calculate.
        """
        return D1TeleopStatus(
            forward_velocity_mps=0.0,
            yaw_rate_rps=0.0,
            base_height_m=float(self.side.pose[2]),
            jump_phase="ready",
            safety_mode="side_step_abort" if self.side.failure else "side_step",
            traction_mode="not_applicable",
            rl_mode="off",
            vertical_feedforward_force_n=None,
            residual_longitudinal_force_n=0.0,
            residual_vertical_force_n=0.0,
            safety_interventions=self.teleop._safety_interventions,
            completed_jumps=self.teleop._completed_jumps,
            torque_saturation_fraction=float(np.mean(
                np.abs(torque) >= 0.98 * self.plant.actuator_torque_limit_nm
            )),
            state_estimation_mode="simulator_truth",
            state_age_ms=None,
            latency_compensation="not_applicable",
            compensation_status="not_applicable",
            compensation_horizon_ms=None,
            contact_allocation="not_applicable",
            allocation_status="not_applicable",
            allocation_wrench_tracking_status="not_applicable",
            allocation_status_reason="side controller has its own force allocation",
            allocation_solve_ms=None,
            allocation_constraint_violation=None,
            allocation_force_error_norm_n=None,
            allocation_moment_error_norm_nm=None,
        )

    def step(self, requested: D1MotionCommand, direction: int):
        """Return (teleop status, applied 16 torques, side/source status).

        A completion tick still belongs to the side controller. Legacy control
        may resume on the next call. Failure alone never permits handoff.
        Jump requests arriving during a side step are discarded while landing;
        the operator must request a new jump after that landing completes.
        """
        if not isinstance(requested, D1MotionCommand):
            raise TypeError("requested must be D1MotionCommand")
        if (
            isinstance(direction, (bool, np.bool_))
            or not isinstance(direction, (int, np.integer))
            or direction not in (-1, 0, 1)
        ):
            raise ValueError("direction must be -1, 0, or +1")
        if direction == 0:
            self._blocked_until_release = False

        teleop = self.teleop
        teleop.forward_velocity_mps = requested.forward_velocity_mps
        teleop.yaw_rate_rps = requested.yaw_rate_rps
        teleop.base_height_m = requested.clearance_m
        jump_pending = teleop._jump_requested
        if (
            not self.active
            and direction != 0
            and not self._blocked_until_release
            and teleop._jump_step is None
            and not jump_pending
        ):
            # SideStepController also checks four contacts, attitude, linear
            # speed and the actual contact plane. Add a stopped-yaw gate.
            _, angular_velocity = self.plant.base_velocity(local=False)
            if np.linalg.norm(angular_velocity) < 0.08:
                self._active = self.side.start(int(direction))

        side_tick = self.active
        jump_discarded = bool(side_tick and jump_pending)
        brake_eligible = (
            self.brake_reference_mode == "release_reanchor_experimental"
            and not side_tick and direction == 0 and not jump_pending
            and teleop._jump_step is None and not teleop._state.has_fallen()
            and teleop._state.base_position[2] >= 0.28
        )
        braking_info = self.braking.prepare(
            teleop.controller, requested.forward_velocity_mps, eligible=bool(brake_eligible)
        )
        if side_tick:
            if direction != self.side.direction or jump_discarded:
                self.side.cancel()
            # Do not queue a delayed jump while a wheel is still in the air.
            teleop._jump_requested = False
            torque = self.side.compute()
            status = self._side_teleop_status(torque)
        else:
            torque, status = teleop.compute()
            # Record the command actually accepted by teleop guards, not merely
            # the request; jumping and recovery cannot synthesize release edges.
            self.braking.observe_applied(
                status.forward_velocity_mps,
                eligible=bool(brake_eligible and status.jump_phase == "ready"
                              and status.safety_mode != "recovery"),
            )

        # The plant applies this same bound itself; return exactly that vector,
        # not an unclipped request or the previous controller's cached torque.
        torque = np.asarray(torque, dtype=np.float64)
        if torque.shape != (16,) or not np.isfinite(torque).all():
            raise RuntimeError("controller returned invalid motor torque")
        torque = np.clip(torque, -self.plant.actuator_torque_limit_nm,
                         self.plant.actuator_torque_limit_nm)
        self.plant.step(torque)
        teleop.update_state()

        if side_tick and self.side.done:
            self._prepare_legacy_handoff()
            self._active = False
            if self.side.failure is not None:
                self._blocked_until_release = direction != 0

        info = dict(self.side.status)
        info.update(
            active=self.active,
            torque_source=info["torque_source"] if side_tick else "legacy",
            state_source="simulator_truth" if side_tick else teleop.state_mode,
            blocked_until_release=self._blocked_until_release,
            jump_request_discarded=jump_discarded,
            brake_reference_mode=self.brake_reference_mode,
            braking_profile=("legacy_release_reference_edge_v1" if self.brake_reference_mode
                             == "release_reanchor_experimental" else "legacy_retained_reference"),
            **braking_info,
        )
        return status, torque.copy(), info
