"""Real-time keyboard driving and scripted terrain demos for the D1 robot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from time import perf_counter, sleep
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .controllers import D1Command
from .env import D1_RESIDUAL_SCALE, encode_d1_observation
from .hierarchical import D1LQRVMCController, D1MPCVMCController
from .model import JOINT_TORQUE_LIMIT, D1Plant
from .policy import load_compatible_d1_policy
from .state_estimation import (
    D1StateEstimate,
    make_d1_state_source,
    make_default_d1_estimator_impairments,
    prepare_d1_control_state,
)
from .terrain import D1_COURSE_SPAWNS, D1TerrainAttitudeEstimator

_SPAWN_KEYS = {
    "1": "start",
    "2": "rough",
    "3": "ramp",
    "4": "stairs",
    "5": "bumps",
    "6": "jump",
}

KEYBOARD_HELP = """
D1 interactive controls
  W / S       increase / decrease forward speed
  A / D       increase left / right yaw rate
  Q / E       lower / raise body height
  X           stop velocity and yaw commands
  SPACE       request one guarded jump
  1..6        reset at start / rough / ramp / stairs / bumps / jump lane
  R           reset at course start
  C           recenter tracking camera
  P           pause physics
  L           toggle compatible PPO residual (when --policy is supplied)
  H           print this help
""".strip()


@dataclass(frozen=True)
class D1TeleopStatus:
    """Observable teleoperation state for overlays, tests, and logs."""

    forward_velocity_mps: float
    yaw_rate_rps: float
    base_height_m: float
    jump_phase: str
    safety_mode: str
    traction_mode: str
    rl_mode: str
    vertical_feedforward_force_n: float
    residual_longitudinal_force_n: float
    residual_vertical_force_n: float
    safety_interventions: int
    completed_jumps: int
    torque_saturation_fraction: float
    state_estimation_mode: str
    state_age_ms: float
    latency_compensation: str
    compensation_status: str
    compensation_horizon_ms: float


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    return np.asarray(
        (np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)),
        dtype=np.float64,
    )


class D1TeleopController:
    """Convert key events into bounded commands, jump phases, and safety states."""

    _jump_schedule = (
        ("crouch", 25, 0.405, -80.0),
        ("thrust", 15, 0.500, 450.0),
        ("flight", 35, 0.455, 0.0),
        ("landing", 45, 0.435, 60.0),
    )

    def __init__(
        self,
        plant: D1Plant,
        *,
        baseline: str = "lqr",
        residual_policy: Any | None = None,
        state_mode: str = "oracle",
        latency_compensation: str = "none",
        state_delay_steps: int = 0,
        sensor_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        if baseline not in {"lqr", "mpc"}:
            raise ValueError("baseline must be 'lqr' or 'mpc'")
        if state_mode not in {"oracle", "estimated"}:
            raise ValueError("state_mode must be 'oracle' or 'estimated'")
        if latency_compensation not in {"none", "constant_velocity"}:
            raise ValueError("latency_compensation must be 'none' or 'constant_velocity'")
        if state_mode == "oracle" and latency_compensation != "none":
            raise ValueError("latency compensation requires state_mode='estimated'")
        if state_delay_steps < 0 or sensor_noise < 0.0:
            raise ValueError("state delay and sensor noise must be non-negative")
        self.plant = plant
        self.state_mode = state_mode
        self.latency_compensation = latency_compensation
        self.seed = seed
        self.residual_policy = residual_policy
        self.controller = (
            D1LQRVMCController(plant) if baseline == "lqr" else D1MPCVMCController(plant)
        )
        self._last_torque = np.zeros(16, dtype=np.float64)
        self._jump_step: int | None = None
        self._jump_requested = False
        self._safety_interventions = 0
        self._completed_jumps = 0
        self._stuck_steps = 0
        self._boost_steps_remaining = 0
        self._previous_residual_action = np.zeros(2, dtype=np.float64)
        self.rl_enabled = False
        self.terrain_attitude = D1TerrainAttitudeEstimator()
        self.forward_velocity_mps = 0.0
        self.yaw_rate_rps = 0.0
        self.base_height_m = plant.nominal_base_height_m
        impairments = None
        if state_mode == "estimated":
            impairments = make_default_d1_estimator_impairments(
                delay_steps=state_delay_steps,
                noise_scale=sensor_noise,
            )
        self.state_source = make_d1_state_source(
            plant,
            impairments=impairments,
            seed=seed,
        )
        self._state: D1StateEstimate
        self._compensation_status = "disabled"
        self._compensation_horizon_s = 0.0
        self.reset("start")

    def _publish_control_state(self, raw_state: D1StateEstimate) -> None:
        result = prepare_d1_control_state(
            raw_state,
            latency_compensation=self.latency_compensation,
        )
        self._state = result.control_state
        self._compensation_status = result.status
        self._compensation_horizon_s = result.applied_horizon_s

    def reset(self, spawn: str = "start") -> None:
        if spawn not in D1_COURSE_SPAWNS:
            raise ValueError(f"unknown course spawn: {spawn}")
        target = D1_COURSE_SPAWNS[spawn]
        self.plant.reset(
            base_position=np.asarray(target.position_m, dtype=np.float64),
            base_quaternion=_yaw_quaternion(target.yaw_rad),
        )
        self._publish_control_state(self.state_source.reset(seed=self.seed))
        self.controller.reset()
        self.terrain_attitude.reset()
        self._last_torque[:] = 0.0
        self._jump_step = None
        self._jump_requested = False
        self._safety_interventions = 0
        self._completed_jumps = 0
        self._stuck_steps = 0
        self._boost_steps_remaining = 0
        self._previous_residual_action[:] = 0.0
        self.forward_velocity_mps = 0.0
        self.yaw_rate_rps = 0.0
        self.base_height_m = self.plant.nominal_base_height_m

    def handle_key(self, keycode: int) -> str | None:
        """Apply one key event and return a viewer event when needed."""

        try:
            key = chr(keycode).lower()
        except (TypeError, ValueError):
            return None
        if key == "w":
            self.forward_velocity_mps = float(np.clip(self.forward_velocity_mps + 0.10, -0.8, 0.8))
        elif key == "s":
            self.forward_velocity_mps = float(np.clip(self.forward_velocity_mps - 0.10, -0.8, 0.8))
        elif key == "a":
            self.yaw_rate_rps = float(np.clip(self.yaw_rate_rps + 0.10, -0.6, 0.6))
        elif key == "d":
            self.yaw_rate_rps = float(np.clip(self.yaw_rate_rps - 0.10, -0.6, 0.6))
        elif key == "q":
            self.base_height_m = float(np.clip(self.base_height_m - 0.01, 0.40, 0.50))
        elif key == "e":
            self.base_height_m = float(np.clip(self.base_height_m + 0.01, 0.40, 0.50))
        elif key == "x":
            self.forward_velocity_mps = 0.0
            self.yaw_rate_rps = 0.0
        elif key == " ":
            self._jump_requested = True
        elif key in _SPAWN_KEYS:
            return f"reset:{_SPAWN_KEYS[key]}"
        elif key == "r":
            return "reset:start"
        elif key == "p":
            return "pause"
        elif key == "l" and self.residual_policy is not None:
            self.rl_enabled = not self.rl_enabled
            if not self.rl_enabled:
                self._previous_residual_action[:] = 0.0
        elif key == "c":
            return "camera"
        elif key == "h":
            return "help"
        return None

    def _residual_action(
        self,
        command: D1Command,
        state: D1StateEstimate,
        *,
        jump_phase: str,
        safety_mode: str,
        terrain_roll_rad: float,
        terrain_pitch_rad: float,
    ) -> tuple[np.ndarray, str]:
        if self.residual_policy is None:
            return np.zeros(2, dtype=np.float64), "unavailable"
        if not self.rl_enabled:
            return np.zeros(2, dtype=np.float64), "off"
        outside_training_envelope = (
            jump_phase != "ready"
            or safety_mode != "normal"
            or abs(self.yaw_rate_rps) > 0.20
            or abs(terrain_roll_rad) > 0.10
            or abs(terrain_pitch_rad) > 0.10
        )
        if outside_training_envelope:
            self._previous_residual_action[:] = 0.0
            return np.zeros(2, dtype=np.float64), "gated"
        observation = encode_d1_observation(
            state=state,
            command=command,
            baseline_longitudinal_force_n=self.controller.last_longitudinal_force_n,
            previous_applied_action=self._previous_residual_action,
        )
        action, _ = self.residual_policy.predict(observation, deterministic=True)
        normalized = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if normalized.shape != (2,):
            raise ValueError("D1 policy must return an action with shape (2,)")
        self._previous_residual_action = normalized.copy()
        return normalized, "on"

    def _jump_command(self, state: D1StateEstimate) -> tuple[str, float, float]:
        roll, pitch, _ = state.base_rpy
        can_jump = (
            state.wheel_ground_contacts >= 3
            and abs(roll) < 0.18
            and abs(pitch) < 0.18
            and not state.has_fallen()
        )
        if self._jump_step is None and self._jump_requested and can_jump:
            self._jump_step = 0
        self._jump_requested = False
        if self._jump_step is None:
            return "ready", self.base_height_m, 0.0

        offset = self._jump_step
        for phase, duration, height, force in self._jump_schedule:
            if offset < duration:
                self._jump_step += 1
                return phase, height, force
            offset -= duration
        self._jump_step = None
        self._completed_jumps += 1
        return "ready", self.base_height_m, 0.0

    def compute(self) -> tuple[np.ndarray, D1TeleopStatus]:
        state = self._state
        jump_phase, height_command, vertical_force = self._jump_command(state)
        roll, pitch, _ = state.base_rpy
        attitude = max(abs(float(roll)), abs(float(pitch)))
        safety_mode = "normal"
        forward = self.forward_velocity_mps
        yaw_rate = self.yaw_rate_rps
        if state.has_fallen() or state.base_position[2] < 0.28:
            safety_mode = "recovery"
            forward = 0.0
            yaw_rate = 0.0
            height_command = self.plant.nominal_base_height_m
            vertical_force = max(vertical_force, 100.0)
            self._jump_step = None
        elif attitude > 0.30 and jump_phase == "ready":
            safety_mode = "degraded"
            forward = float(np.clip(forward, -0.15, 0.15))
            yaw_rate = float(np.clip(yaw_rate, -0.10, 0.10))
        if safety_mode != "normal":
            self._safety_interventions += 1

        terrain_attitude = self.terrain_attitude.update(state)
        local_velocity = float(state.base_linear_velocity_body[0])
        appears_stuck = (
            abs(forward) > 0.12
            and abs(local_velocity) < 0.08
            and state.wheel_ground_contacts >= 3
            and abs(terrain_attitude.pitch_rad) < 0.08
            and jump_phase == "ready"
        )
        self._stuck_steps = self._stuck_steps + 1 if appears_stuck else 0
        if self._stuck_steps >= 25:
            self._boost_steps_remaining = 50
            self._stuck_steps = 0
        if abs(terrain_attitude.pitch_rad) >= 0.08:
            self._boost_steps_remaining = 0
        traction_mode = "boost" if self._boost_steps_remaining > 0 else "normal"
        self._boost_steps_remaining = max(0, self._boost_steps_remaining - 1)
        self.controller.longitudinal_force_limit_n = 500.0 if traction_mode == "boost" else 180.0
        command = D1Command(
            forward_velocity_mps=forward,
            yaw_rate_rps=yaw_rate,
            base_height_m=height_command,
            roll_rad=terrain_attitude.roll_rad,
            pitch_rad=terrain_attitude.pitch_rad,
        )
        residual_action, rl_mode = self._residual_action(
            command,
            state,
            jump_phase=jump_phase,
            safety_mode=safety_mode,
            terrain_roll_rad=terrain_attitude.roll_rad,
            terrain_pitch_rad=terrain_attitude.pitch_rad,
        )
        residual_force = residual_action * D1_RESIDUAL_SCALE
        torque = self.controller.compute(
            command,
            state,
            residual_force_n=residual_force,
            vertical_feedforward_force_n=vertical_force,
        )
        max_delta = JOINT_TORQUE_LIMIT * self.plant.control_dt / 0.02
        torque = np.clip(torque, self._last_torque - max_delta, self._last_torque + max_delta)
        self._last_torque = torque.copy()
        status = D1TeleopStatus(
            forward_velocity_mps=forward,
            yaw_rate_rps=yaw_rate,
            base_height_m=height_command,
            jump_phase=jump_phase,
            safety_mode=safety_mode,
            traction_mode=traction_mode,
            rl_mode=rl_mode,
            vertical_feedforward_force_n=vertical_force,
            residual_longitudinal_force_n=float(residual_force[0]),
            residual_vertical_force_n=float(residual_force[1]),
            safety_interventions=self._safety_interventions,
            completed_jumps=self._completed_jumps,
            torque_saturation_fraction=float(np.mean(np.abs(torque) >= 0.98 * JOINT_TORQUE_LIMIT)),
            state_estimation_mode=self.state_mode,
            state_age_ms=1e3 * state.age_s,
            latency_compensation=self.latency_compensation,
            compensation_status=self._compensation_status,
            compensation_horizon_ms=1e3 * self._compensation_horizon_s,
        )
        return torque, status

    def update_state(self) -> None:
        """Publish the measurement produced after the latest physics step."""

        self._publish_control_state(self.state_source.read())


class D1InteractiveSimulation:
    """One-step interface shared by the GUI, scripted recorder, and tests."""

    def __init__(
        self,
        *,
        baseline: str = "lqr",
        arena: str = "course",
        residual_policy: Any | None = None,
        state_mode: str = "oracle",
        latency_compensation: str = "none",
        state_delay_steps: int = 0,
        sensor_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.plant = D1Plant(arena=arena)
        self.teleop = D1TeleopController(
            self.plant,
            baseline=baseline,
            residual_policy=residual_policy,
            state_mode=state_mode,
            latency_compensation=latency_compensation,
            state_delay_steps=state_delay_steps,
            sensor_noise=sensor_noise,
            seed=seed,
        )

    def reset(self, spawn: str = "start") -> None:
        self.teleop.reset(spawn)

    def step(self) -> D1TeleopStatus:
        torque, status = self.teleop.compute()
        self.plant.step(torque)
        self.teleop.update_state()
        return status


def _configure_tracking_camera(viewer: object, simulation: D1InteractiveSimulation) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = simulation.plant.base_body_id
    viewer.cam.distance = 2.8
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0


def _render_frame(
    simulation: D1InteractiveSimulation,
    renderer: mujoco.Renderer,
    status: D1TeleopStatus,
) -> Image.Image:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = simulation.plant.base_position
    camera.distance = 2.8
    camera.azimuth = 135.0
    camera.elevation = -18.0
    renderer.update_scene(simulation.plant.data, camera=camera)
    frame = Image.fromarray(renderer.render().copy())
    canvas = Image.new("RGB", (frame.width, frame.height + 34), "white")
    canvas.paste(frame, (0, 34))
    label = (
        f"v={status.forward_velocity_mps:+.2f} m/s  "
        f"yaw={status.yaw_rate_rps:+.2f} rad/s  "
        f"jump={status.jump_phase}  rl={status.rl_mode}  traction={status.traction_mode}  "
        f"safety={status.safety_mode}  state={status.state_estimation_mode} "
        f"age={status.state_age_ms:.0f} ms  comp={status.latency_compensation}"
    )
    ImageDraw.Draw(canvas).text((10, 10), label, fill="black")
    return canvas


def run_scripted_demo(
    zone: str,
    output: Path | None = None,
    *,
    baseline: str = "lqr",
    state_mode: str = "oracle",
    latency_compensation: str = "none",
    state_delay_steps: int = 0,
    sensor_noise: float = 0.0,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Run a deterministic skill demo and optionally save a real-physics GIF."""

    if zone not in D1_COURSE_SPAWNS:
        raise ValueError(f"unknown demo zone: {zone}")
    simulation = D1InteractiveSimulation(
        baseline=baseline,
        state_mode=state_mode,
        latency_compensation=latency_compensation,
        state_delay_steps=state_delay_steps,
        sensor_noise=sensor_noise,
        seed=seed,
    )
    simulation.reset(zone)
    duration_s = {
        "start": 7.0,
        "rough": 8.0,
        "ramp": 13.0,
        "stairs": 13.0,
        "bumps": 14.0,
        "jump": 10.0,
    }[zone]
    target_speed = {
        "start": 0.35,
        "rough": 0.30,
        "ramp": 0.38,
        "stairs": 0.24,
        "bumps": 0.28,
        "jump": 0.28,
    }[zone]
    settle_steps = round(1.0 / simulation.plant.control_dt)
    total_steps = round(duration_s / simulation.plant.control_dt)
    renderer = (
        mujoco.Renderer(simulation.plant.model, height=360, width=640)
        if output is not None
        else None
    )
    frames: list[Image.Image] = []
    maximum_height = float(simulation.plant.base_position[2])
    minimum_contacts = 4
    four_wheel_steps = 0
    maximum_roll = 0.0
    maximum_pitch = 0.0
    saturation_sum = 0.0
    step_times_ms: list[float] = []
    compensation_horizons_ms: list[float] = []
    compensation_applied_steps = 0
    compensation_rejected_steps = 0
    terminated = False
    status: D1TeleopStatus
    for step in range(total_steps):
        if step == settle_steps:
            simulation.teleop.forward_velocity_mps = target_speed
        if zone == "jump" and step == round(4.5 / simulation.plant.control_dt):
            simulation.teleop.handle_key(ord(" "))
        step_start = perf_counter()
        status = simulation.step()
        step_times_ms.append(1e3 * (perf_counter() - step_start))
        maximum_height = max(maximum_height, float(simulation.plant.base_position[2]))
        minimum_contacts = min(minimum_contacts, simulation.plant.wheel_ground_contacts)
        four_wheel_steps += int(simulation.plant.wheel_ground_contacts == 4)
        maximum_roll = max(maximum_roll, abs(float(simulation.plant.base_rpy[0])))
        maximum_pitch = max(maximum_pitch, abs(float(simulation.plant.base_rpy[1])))
        saturation_sum += status.torque_saturation_fraction
        compensation_horizons_ms.append(status.compensation_horizon_ms)
        compensation_applied_steps += int(status.compensation_status == "applied")
        compensation_rejected_steps += int(
            status.compensation_status in {"horizon_exceeded", "kinematic_horizon_exceeded"}
        )
        terminated = terminated or simulation.plant.has_fallen()
        if renderer is not None and step % 5 == 0:
            frames.append(_render_frame(simulation, renderer, status))
    if renderer is not None:
        renderer.close()
    if output is not None and frames:
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
            optimize=True,
        )
    distance = float(
        np.linalg.norm(
            simulation.plant.base_position[:2] - np.asarray(D1_COURSE_SPAWNS[zone].position_m[:2])
        )
    )
    minimum_progress = {
        "start": 1.0,
        "rough": 2.0,
        "ramp": 4.0,
        "stairs": 3.5,
        "bumps": 3.5,
        "jump": 1.7,
    }[zone]
    cleared_hurdles = (
        int(
            sum(
                simulation.plant.base_position[0] >= hurdle_x + 0.30 for hurdle_x in (1.8, 3.0, 4.3)
            )
        )
        if zone == "jump"
        else 0
    )
    completed_required_jump = zone != "jump" or (
        status.completed_jumps >= 1 and cleared_hurdles >= 1
    )
    return {
        "zone": zone,
        "baseline": baseline,
        "state_estimation_mode": state_mode,
        "latency_compensation": latency_compensation,
        "evaluation_seed": seed,
        "success": int(not terminated and distance >= minimum_progress and completed_required_jump),
        "distance_m": distance,
        "maximum_height_m": maximum_height,
        "minimum_wheel_contacts": minimum_contacts,
        "four_wheel_contact_ratio": four_wheel_steps / total_steps,
        "max_abs_roll_deg": float(np.rad2deg(maximum_roll)),
        "max_abs_pitch_deg": float(np.rad2deg(maximum_pitch)),
        "torque_saturation_ratio": saturation_sum / total_steps,
        "step_time_p95_ms": float(np.percentile(step_times_ms, 95)),
        "compensation_applied_ratio": compensation_applied_steps / total_steps,
        "compensation_horizon_p95_ms": float(np.percentile(compensation_horizons_ms, 95)),
        "compensation_rejected_steps": compensation_rejected_steps,
        "completed_jumps": status.completed_jumps,
        "cleared_hurdles": cleared_hurdles,
        "jump_height_gain_m": maximum_height - float(D1_COURSE_SPAWNS[zone].position_m[2]),
        "safety_interventions": status.safety_interventions,
    }


def write_course_audit(
    output: Path,
    *,
    baseline: str = "lqr",
    state_mode: str = "oracle",
    latency_compensation: str = "none",
    state_delay_steps: int = 0,
    sensor_noise: float = 0.0,
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """Evaluate every course zone and write recruiter-readable raw and summary data."""

    records = [
        run_scripted_demo(
            zone,
            baseline=baseline,
            state_mode=state_mode,
            latency_compensation=latency_compensation,
            state_delay_steps=state_delay_steps,
            sensor_noise=sensor_noise,
            seed=seed,
        )
        for zone in D1_COURSE_SPAWNS
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "course_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    lines = [
        f"# D1 interactive course audit ({baseline.upper()})",
        "",
        "Scripted commands in the same MuJoCo course used by the keyboard demo.",
        "",
        (
            "| Zone | Pass | Progress [m] | Roll max [deg] | Pitch max [deg] | "
            "4-wheel contact | Jumps | Hurdles | Compensation applied | Rejected | "
            "Step P95 [ms] |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['zone']} | {record['success']} | {record['distance_m']:.2f} | "
            f"{record['max_abs_roll_deg']:.2f} | {record['max_abs_pitch_deg']:.2f} | "
            f"{record['four_wheel_contact_ratio']:.3f} | "
            f"{record['completed_jumps']} | {record['cleared_hurdles']} | "
            f"{record['compensation_applied_ratio']:.3f} | "
            f"{record['compensation_rejected_steps']} | "
            f"{record['step_time_p95_ms']:.3f} |"
        )
    (output / "course_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def run_viewer(
    baseline: str,
    residual_policy: Any | None = None,
    *,
    state_mode: str = "oracle",
    latency_compensation: str = "none",
    state_delay_steps: int = 0,
    sensor_noise: float = 0.0,
    seed: int = 0,
) -> None:
    import mujoco.viewer

    simulation = D1InteractiveSimulation(
        baseline=baseline,
        residual_policy=residual_policy,
        state_mode=state_mode,
        latency_compensation=latency_compensation,
        state_delay_steps=state_delay_steps,
        sensor_noise=sensor_noise,
        seed=seed,
    )
    events: SimpleQueue[int] = SimpleQueue()
    paused = False
    print(KEYBOARD_HELP)
    with mujoco.viewer.launch_passive(
        simulation.plant.model,
        simulation.plant.data,
        key_callback=events.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        _configure_tracking_camera(viewer, simulation)
        next_status_time = 0.0
        while viewer.is_running():
            start = perf_counter()
            while not events.empty():
                event = simulation.teleop.handle_key(events.get())
                if event and event.startswith("reset:"):
                    simulation.reset(event.split(":", 1)[1])
                    _configure_tracking_camera(viewer, simulation)
                    next_status_time = 0.0
                elif event == "pause":
                    paused = not paused
                elif event == "camera":
                    _configure_tracking_camera(viewer, simulation)
                elif event == "help":
                    print(KEYBOARD_HELP)
            if not paused:
                status = simulation.step()
                if simulation.plant.data.time >= next_status_time:
                    print(
                        f"t={simulation.plant.data.time:5.1f}s  "
                        f"v={status.forward_velocity_mps:+.2f}  "
                        f"yaw={status.yaw_rate_rps:+.2f}  "
                        f"jump={status.jump_phase:<7}  "
                        f"rl={status.rl_mode:<11}  "
                        f"traction={status.traction_mode:<6}  safety={status.safety_mode}  "
                        f"state={status.state_estimation_mode} age={status.state_age_ms:.0f}ms  "
                        f"comp={status.latency_compensation}"
                    )
                    next_status_time += 1.0
            viewer.sync()
            remaining = simulation.plant.control_dt - (perf_counter() - start)
            if remaining > 0.0:
                sleep(remaining)


def render_course_overview(output: Path) -> None:
    """Render a reproducible bird's-eye view of the complete skills course."""

    simulation = D1InteractiveSimulation()
    for _ in range(round(0.8 / simulation.plant.control_dt)):
        simulation.step()
    renderer = mujoco.Renderer(simulation.plant.model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (3.1, 0.7, 0.0)
    camera.distance = 10.5
    camera.azimuth = 128.0
    camera.elevation = -55.0
    renderer.update_scene(simulation.plant.data, camera=camera)
    frame = Image.fromarray(renderer.render().copy())
    renderer.close()

    labels = "rough + ramp  |  stairs  |  wave bumps  |  2/4/6 cm jump lane"
    canvas = Image.new("RGB", (frame.width, frame.height + 42), "white")
    canvas.paste(frame, (0, 42))
    ImageDraw.Draw(canvas).text((14, 14), labels, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("lqr", "mpc"), default="lqr")
    parser.add_argument("--state-mode", choices=("oracle", "estimated"), default="oracle")
    parser.add_argument(
        "--latency-compensation",
        choices=("none", "constant_velocity"),
        default="none",
    )
    parser.add_argument("--state-delay-steps", type=int, default=0)
    parser.add_argument("--sensor-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", type=Path, help="load a compatible PPO residual for L toggle")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo-zone", choices=tuple(D1_COURSE_SPAWNS))
    mode.add_argument("--audit-output", type=Path, help="write metrics for every course zone")
    mode.add_argument("--overview", type=Path, help="render a bird's-eye PNG of the course")
    parser.add_argument("--record", type=Path, help="save scripted demo as GIF")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.state_mode != "estimated" and args.latency_compensation != "none":
        raise SystemExit("latency compensation requires --state-mode estimated")
    if args.record is not None and args.demo_zone is None:
        raise SystemExit("--record requires --demo-zone")
    if args.policy is not None and any(
        option is not None for option in (args.demo_zone, args.audit_output, args.overview)
    ):
        raise SystemExit("--policy is available in the interactive viewer only")
    if args.overview is not None:
        render_course_overview(args.overview)
        print(f"saved course overview to {args.overview}")
        return
    if args.audit_output is not None:
        records = write_course_audit(
            args.audit_output,
            baseline=args.baseline,
            state_mode=args.state_mode,
            latency_compensation=args.latency_compensation,
            state_delay_steps=args.state_delay_steps,
            sensor_noise=args.sensor_noise,
            seed=args.seed,
        )
        for record in records:
            print(record)
        return
    if args.demo_zone is None:
        residual_policy = None
        if args.policy is not None:
            try:
                residual_policy = load_compatible_d1_policy(
                    args.policy,
                    expected_baseline=args.baseline,
                    expected_state_mode=args.state_mode,
                    expected_latency_compensation=args.latency_compensation,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                raise SystemExit(str(error)) from error
        run_viewer(
            args.baseline,
            residual_policy,
            state_mode=args.state_mode,
            latency_compensation=args.latency_compensation,
            state_delay_steps=args.state_delay_steps,
            sensor_noise=args.sensor_noise,
            seed=args.seed,
        )
        return
    metrics = run_scripted_demo(
        args.demo_zone,
        args.record,
        baseline=args.baseline,
        state_mode=args.state_mode,
        latency_compensation=args.latency_compensation,
        state_delay_steps=args.state_delay_steps,
        sensor_noise=args.sensor_noise,
        seed=args.seed,
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
