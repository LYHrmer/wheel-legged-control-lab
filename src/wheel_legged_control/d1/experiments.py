"""Benchmark and render the full-body D1 LQR/MPC/residual-RL stack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import t as student_t

from ..provenance import capture_git_provenance
from .env import D1_RESIDUAL_SCALE, D1ResidualEnv
from .model import JOINT_TORQUE_LIMIT
from .policy import load_compatible_d1_policy

plt.switch_backend("Agg")

D1_SCENARIOS = ("nominal", "push", "mismatch_delay")
D1_STATE_DELAY_SWEEP_STEPS = (0, 1, 2, 3, 5)
D1_DEFAULT_POLICY_PATH = Path("results/d1_residual_ppo/model.zip")
D1_SWEEP_ARTIFACTS = (
    "delay_sweep_episodes.csv",
    "delay_sweep_summary.csv",
    "state_delay_sensitivity.md",
    "state_delay_sensitivity.png",
    "evaluation_config.json",
    "delay_sweep_manifest.json",
)

# The Markdown report emits every item in this registry.  Keep the primary
# survival outcomes first; the remaining values describe the trajectory prefix
# observed before an episode either finishes or falls.
D1_SWEEP_METRICS = {
    "success": ("Success", "ratio"),
    "episode_duration_s": ("Episode duration", "s"),
    "mean_reward": ("Mean reward", "reward/step"),
    "velocity_rmse_mps": ("Velocity RMSE", "m/s"),
    "pitch_rmse_deg": ("Pitch RMSE", "deg"),
    "max_abs_pitch_deg": ("Maximum absolute pitch", "deg"),
    "height_rmse_mm": ("Height RMSE", "mm"),
    "normalized_torque_rms": ("Normalized torque RMS", "ratio"),
    "mean_abs_mechanical_power_w": ("Mean absolute mechanical power", "W"),
    "torque_saturation_ratio": ("Torque saturation", "ratio"),
    "four_wheel_contact_ratio": ("Four-wheel contact", "ratio"),
    "undesired_contact_steps": ("Undesired-contact steps", "steps"),
    "solve_p95_ms": ("Solve-time P95", "ms"),
    "state_age_mean_ms": ("Measured state age mean", "ms"),
    "state_age_p95_ms": ("Measured state age P95", "ms"),
    "state_age_max_ms": ("Measured state age maximum", "ms"),
    "compensation_horizon_p95_ms": ("Compensation horizon P95", "ms"),
    "compensation_applied_ratio": ("Compensation applied", "ratio"),
    "compensation_rejected_ratio": ("Compensation rejected", "ratio"),
    "raw_position_estimation_rmse_m": ("Raw position-estimation RMSE", "m"),
    "control_position_estimation_rmse_m": ("Control position-estimation RMSE", "m"),
    "raw_pitch_estimation_rmse_deg": ("Raw pitch-estimation RMSE", "deg"),
    "control_pitch_estimation_rmse_deg": ("Control pitch-estimation RMSE", "deg"),
    "raw_velocity_estimation_rmse_mps": ("Raw velocity-estimation RMSE", "m/s"),
    "control_velocity_estimation_rmse_mps": ("Control velocity-estimation RMSE", "m/s"),
    "residual_action_rms": ("Residual-action RMS", "ratio"),
    "residual_action_delta_rms": ("Residual-action delta RMS", "ratio"),
    "mean_residual_longitudinal_force_n": ("Mean residual longitudinal force", "N"),
    "mean_residual_vertical_force_n": ("Mean residual vertical force", "N"),
}


@dataclass
class D1Rollout:
    controller: str
    scenario: str
    time_s: np.ndarray
    states: np.ndarray
    forward_velocities_mps: np.ndarray
    torques: np.ndarray
    commands: np.ndarray
    pushes: np.ndarray
    rewards: np.ndarray
    wheel_contacts: np.ndarray
    undesired_contacts: np.ndarray
    solve_times_ms: np.ndarray
    terminated: bool
    frames: list[np.ndarray] | None = None
    evaluation_seed: int = -1
    state_estimation_mode: str = "oracle"
    state_ages_ms: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    domain: dict[str, float] = field(default_factory=dict)
    action_delay_steps: int = 0
    state_delay_steps: int = 0
    sensor_noise_scale: float = 0.0
    state_estimator_seed: int | None = None
    residual_actions: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    latency_compensation: str = "none"
    compensation_horizons_ms: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    compensation_applied_flags: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )
    compensation_rejected_flags: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )
    absolute_mechanical_power_w: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    torque_saturation_fractions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    raw_estimated_states: np.ndarray = field(
        default_factory=lambda: np.empty((0, 6), dtype=np.float64)
    )
    control_states: np.ndarray = field(default_factory=lambda: np.empty((0, 6), dtype=np.float64))
    initial_state_fingerprint: str = ""
    initial_command_fingerprint: str = ""
    push_schedule_fingerprint: str = ""
    control_dt_s: float = 0.01


def d1_scenario_options(name: str) -> dict[str, Any]:
    if name == "randomized":
        return {"scenario": "training", "randomize": True}
    options: dict[str, Any] = {"scenario": name, "randomize": False}
    if name == "mismatch_delay":
        options |= {
            "base_mass_scale": 1.12,
            "damping_scale": 1.25,
            "friction_scale": 0.65,
            "actuator_strength_scale": 0.85,
            "action_delay_steps": 3,
            "state_delay_steps": 3,
            "sensor_noise": 1.0,
        }
    return options


def _render_frame(env: D1ResidualEnv, renderer: mujoco.Renderer) -> np.ndarray:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = env.plant.base_position
    camera.distance = 2.2
    camera.azimuth = 135.0
    camera.elevation = -18.0
    renderer.update_scene(env.plant.data, camera=camera)
    return renderer.render().copy()


def _numeric_fingerprint(values: np.ndarray) -> str:
    """Hash numeric evidence with its dtype and shape, not its display format."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "scipy",
        "mujoco",
        "matplotlib",
        "gymnasium",
        "stable-baselines3",
    ):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def run_d1_rollout(
    baseline: str,
    scenario: str,
    seed: int,
    policy: Any | None = None,
    *,
    capture: bool = False,
    state_mode: str = "oracle",
    latency_compensation: str = "none",
    episode_options: dict[str, Any] | None = None,
) -> D1Rollout:
    env = D1ResidualEnv(
        baseline=baseline,
        randomize=False,
        state_mode=state_mode,
        latency_compensation=latency_compensation,
    )
    options = d1_scenario_options(scenario)
    if episode_options is not None:
        options |= episode_options
    observation, reset_info = env.reset(seed=seed, options=options)
    renderer = mujoco.Renderer(env.plant.model, height=300, width=400) if capture else None
    states: list[np.ndarray] = []
    forward_velocities: list[float] = []
    torques: list[np.ndarray] = []
    commands: list[tuple[float, float]] = []
    pushes: list[float] = []
    rewards: list[float] = []
    contacts: list[int] = []
    undesired: list[int] = []
    solve_times: list[float] = []
    state_ages_ms: list[float] = []
    absolute_mechanical_power_w: list[float] = []
    torque_saturation_fractions: list[float] = []
    residual_actions: list[np.ndarray] = []
    compensation_horizons_ms: list[float] = []
    compensation_applied_flags: list[bool] = []
    compensation_rejected_flags: list[bool] = []
    raw_estimated_states: list[np.ndarray] = []
    control_states: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    state_estimation_mode = str(reset_info.get("state_estimation_mode", state_mode))
    domain = {key: float(value) for key, value in reset_info.get("domain", {}).items()}
    action_delay_steps = int(reset_info.get("action_delay_steps", 0))
    state_delay_steps = int(reset_info.get("state_delay_steps", 0))
    sensor_noise_scale = float(reset_info.get("sensor_noise_scale", 0.0))
    state_estimator_seed = reset_info.get("state_estimator_seed")
    initial_truth_state = np.concatenate(
        (
            np.asarray(env.plant.data.qpos, dtype=np.float64),
            np.asarray(env.plant.data.qvel, dtype=np.float64),
        )
    )
    initial_command = np.asarray(
        (
            reset_info["command_velocity_mps"],
            reset_info["command_height_m"],
        ),
        dtype=np.float64,
    )
    # The environment samples the complete push schedule during reset.  Storing
    # its hash avoids falsely treating an early fall (before the push) as a
    # matched input merely because both recorded force prefixes are all zero.
    push_schedule = np.asarray(
        (env._push_start, env._push_end, env._push_force_n),
        dtype=np.float64,
    )
    terminated = truncated = False

    while not (terminated or truncated):
        if policy is None:
            action = np.zeros(2, dtype=np.float32)
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(info["state"])
        forward_velocities.append(info["forward_velocity_mps"])
        torque_nm = np.asarray(info["torque_nm"], dtype=np.float64)
        joint_velocity = np.asarray(info["joint_velocity"], dtype=np.float64)
        strength_scale = float(info["domain"]["actuator_strength_scale"])
        actuator_limit = JOINT_TORQUE_LIMIT * strength_scale
        effective_command_limit = JOINT_TORQUE_LIMIT * min(1.0, strength_scale)
        applied_torque_nm = np.clip(torque_nm, -actuator_limit, actuator_limit)
        torques.append(torque_nm)
        commands.append((info["command_velocity_mps"], info["command_height_m"]))
        pushes.append(info["push_force_n"])
        rewards.append(reward)
        residual_actions.append(np.asarray(info["residual_action"], dtype=np.float64))
        contacts.append(info["wheel_contacts"])
        undesired.append(info["undesired_contacts"])
        solve_times.append(info["solve_ms"])
        state_ages_ms.append(float(info.get("state_age_ms", 0.0)))
        compensation_horizons_ms.append(float(info.get("latency_compensation_horizon_ms", 0.0)))
        compensation_status = str(info.get("latency_compensation_status", "disabled"))
        compensation_applied_flags.append(compensation_status == "applied")
        compensation_rejected_flags.append(
            compensation_status in {"horizon_exceeded", "kinematic_horizon_exceeded"}
        )
        control_state = np.asarray(
            info.get("control_state", info["estimated_state"]), dtype=np.float64
        )
        raw_estimated_state = np.asarray(
            info.get("raw_estimated_state", control_state), dtype=np.float64
        )
        control_states.append(control_state)
        raw_estimated_states.append(raw_estimated_state)
        absolute_mechanical_power_w.append(
            float(np.sum(np.abs(applied_torque_nm * joint_velocity)))
        )
        torque_saturation_fractions.append(
            float(np.mean(np.abs(torque_nm) >= 0.98 * effective_command_limit))
        )
        state_estimation_mode = str(info.get("state_estimation_mode", state_estimation_mode))
        if not domain:
            domain = {key: float(value) for key, value in info["domain"].items()}
        if renderer is not None and len(states) % 5 == 0:
            frames.append(_render_frame(env, renderer))

    if renderer is not None:
        renderer.close()
    label = (
        "D1 LQR+VMC+PPO"
        if policy is not None
        else ("D1 LQR+VMC" if baseline == "lqr" else "D1 MPC+VMC")
    )
    rollout = D1Rollout(
        controller=label,
        scenario=scenario,
        time_s=np.arange(len(states)) * env.plant.control_dt,
        states=np.asarray(states),
        forward_velocities_mps=np.asarray(forward_velocities),
        torques=np.asarray(torques),
        commands=np.asarray(commands),
        pushes=np.asarray(pushes),
        rewards=np.asarray(rewards),
        wheel_contacts=np.asarray(contacts),
        undesired_contacts=np.asarray(undesired),
        solve_times_ms=np.asarray(solve_times),
        terminated=terminated,
        frames=frames if capture else None,
        evaluation_seed=seed,
        state_estimation_mode=state_estimation_mode,
        state_ages_ms=np.asarray(state_ages_ms),
        domain=domain,
        action_delay_steps=action_delay_steps,
        state_delay_steps=state_delay_steps,
        sensor_noise_scale=sensor_noise_scale,
        state_estimator_seed=(None if state_estimator_seed is None else int(state_estimator_seed)),
        absolute_mechanical_power_w=np.asarray(absolute_mechanical_power_w),
        torque_saturation_fractions=np.asarray(torque_saturation_fractions),
        residual_actions=np.asarray(residual_actions),
        latency_compensation=latency_compensation,
        compensation_horizons_ms=np.asarray(compensation_horizons_ms),
        compensation_applied_flags=np.asarray(compensation_applied_flags, dtype=np.bool_),
        compensation_rejected_flags=np.asarray(compensation_rejected_flags, dtype=np.bool_),
        raw_estimated_states=np.asarray(raw_estimated_states),
        control_states=np.asarray(control_states),
        initial_state_fingerprint=_numeric_fingerprint(initial_truth_state),
        initial_command_fingerprint=_numeric_fingerprint(initial_command),
        push_schedule_fingerprint=_numeric_fingerprint(push_schedule),
        control_dt_s=env.plant.control_dt,
    )
    env.close()
    return rollout


def _estimation_rmse(
    estimates: np.ndarray,
    truth: np.ndarray,
    indices: tuple[int, ...],
    *,
    scale: float = 1.0,
) -> float:
    if estimates.shape != truth.shape or estimates.ndim != 2 or estimates.shape[1] < 6:
        return float("nan")
    error = estimates[:, indices] - truth[:, indices]
    return float(scale * np.sqrt(np.mean(error**2)))


def compute_d1_metrics(rollout: D1Rollout) -> dict[str, float | str | int]:
    velocity_error = rollout.forward_velocities_mps - rollout.commands[:, 0]
    pitch_deg = np.rad2deg(rollout.states[:, 1])
    height_error = rollout.states[:, 2] - rollout.commands[:, 1]
    normalized_torque = rollout.torques / JOINT_TORQUE_LIMIT
    state_age_p95_ms = (
        float(np.percentile(rollout.state_ages_ms, 95)) if rollout.state_ages_ms.size else 0.0
    )
    state_age_mean_ms = float(np.mean(rollout.state_ages_ms)) if rollout.state_ages_ms.size else 0.0
    state_age_max_ms = float(np.max(rollout.state_ages_ms)) if rollout.state_ages_ms.size else 0.0
    mean_abs_mechanical_power_w = (
        float(np.mean(rollout.absolute_mechanical_power_w))
        if rollout.absolute_mechanical_power_w.size
        else 0.0
    )
    torque_saturation_ratio = (
        float(np.mean(rollout.torque_saturation_fractions))
        if rollout.torque_saturation_fractions.size
        else float(np.mean(np.abs(normalized_torque) >= 0.98))
    )
    if rollout.residual_actions.size:
        residual_forces = rollout.residual_actions * D1_RESIDUAL_SCALE
        residual_action_rms = float(np.sqrt(np.mean(rollout.residual_actions**2)))
        mean_residual_longitudinal_force_n = float(np.mean(residual_forces[:, 0]))
        mean_residual_vertical_force_n = float(np.mean(residual_forces[:, 1]))
        residual_action_delta_rms = (
            float(np.sqrt(np.mean(np.diff(rollout.residual_actions, axis=0) ** 2)))
            if len(rollout.residual_actions) > 1
            else 0.0
        )
    else:
        residual_action_rms = 0.0
        mean_residual_longitudinal_force_n = 0.0
        mean_residual_vertical_force_n = 0.0
        residual_action_delta_rms = 0.0
    return {
        "controller": rollout.controller,
        "scenario": rollout.scenario,
        "evaluation_seed": rollout.evaluation_seed,
        "state_estimation_mode": rollout.state_estimation_mode,
        "latency_compensation": rollout.latency_compensation,
        "initial_state_fingerprint": rollout.initial_state_fingerprint,
        "initial_command_fingerprint": rollout.initial_command_fingerprint,
        "push_schedule_fingerprint": rollout.push_schedule_fingerprint,
        "state_age_mean_ms": state_age_mean_ms,
        "state_age_p95_ms": state_age_p95_ms,
        "state_age_max_ms": state_age_max_ms,
        "compensation_horizon_p95_ms": (
            float(np.percentile(rollout.compensation_horizons_ms, 95))
            if rollout.compensation_horizons_ms.size
            else 0.0
        ),
        "compensation_applied_ratio": (
            float(np.mean(rollout.compensation_applied_flags))
            if rollout.compensation_applied_flags.size
            else 0.0
        ),
        "compensation_rejected_ratio": (
            float(np.mean(rollout.compensation_rejected_flags))
            if rollout.compensation_rejected_flags.size
            else 0.0
        ),
        "action_delay_steps": rollout.action_delay_steps,
        "state_delay_steps": rollout.state_delay_steps,
        "sensor_noise_scale": rollout.sensor_noise_scale,
        "state_estimator_seed": (
            "" if rollout.state_estimator_seed is None else rollout.state_estimator_seed
        ),
        "base_mass_scale": rollout.domain.get("base_mass_scale", 1.0),
        "damping_scale": rollout.domain.get("damping_scale", 1.0),
        "friction_scale": rollout.domain.get("friction_scale", 1.0),
        "actuator_strength_scale": rollout.domain.get("actuator_strength_scale", 1.0),
        "success": int(not rollout.terminated),
        "episode_steps": len(rollout.time_s),
        "episode_duration_s": float(len(rollout.time_s) * rollout.control_dt_s),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
        "pitch_rmse_deg": float(np.sqrt(np.mean(pitch_deg**2))),
        "max_abs_pitch_deg": float(np.max(np.abs(pitch_deg))),
        "height_rmse_mm": float(1e3 * np.sqrt(np.mean(height_error**2))),
        "normalized_torque_rms": float(np.sqrt(np.mean(normalized_torque**2))),
        "mean_abs_mechanical_power_w": mean_abs_mechanical_power_w,
        "torque_saturation_ratio": torque_saturation_ratio,
        "residual_action_rms": residual_action_rms,
        "residual_action_delta_rms": residual_action_delta_rms,
        "mean_residual_longitudinal_force_n": mean_residual_longitudinal_force_n,
        "mean_residual_vertical_force_n": mean_residual_vertical_force_n,
        "four_wheel_contact_ratio": float(np.mean(rollout.wheel_contacts == 4)),
        "undesired_contact_steps": int(np.count_nonzero(rollout.undesired_contacts)),
        "mean_reward": float(np.mean(rollout.rewards)),
        "solve_p95_ms": float(np.percentile(rollout.solve_times_ms, 95)),
        "raw_position_estimation_rmse_m": _estimation_rmse(
            rollout.raw_estimated_states, rollout.states, (0, 2)
        ),
        "control_position_estimation_rmse_m": _estimation_rmse(
            rollout.control_states, rollout.states, (0, 2)
        ),
        "raw_pitch_estimation_rmse_deg": _estimation_rmse(
            rollout.raw_estimated_states, rollout.states, (1,), scale=180.0 / np.pi
        ),
        "control_pitch_estimation_rmse_deg": _estimation_rmse(
            rollout.control_states, rollout.states, (1,), scale=180.0 / np.pi
        ),
        "raw_velocity_estimation_rmse_mps": _estimation_rmse(
            rollout.raw_estimated_states, rollout.states, (3, 5)
        ),
        "control_velocity_estimation_rmse_mps": _estimation_rmse(
            rollout.control_states, rollout.states, (3, 5)
        ),
    }


def write_d1_metrics(records: list[dict[str, float | str | int]], output: Path) -> None:
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# D1 full-body control benchmark",
        "",
        (
            "| Controller | Scenario | Seed | State | Compensation | Success | Velocity RMSE [m/s] | "
            "Pitch RMSE [deg] | Height RMSE [mm] | State age P95 [ms] | Power [W] | "
            "Torque saturation | Solve P95 [ms] |"
        ),
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['controller']} | {record['scenario']} | "
            f"{record['evaluation_seed']} | {record['state_estimation_mode']} | "
            f"{record['latency_compensation']} | "
            f"{record['success']} | "
            f"{record['velocity_rmse_mps']:.3f} | {record['pitch_rmse_deg']:.3f} | "
            f"{record['height_rmse_mm']:.2f} | {record['state_age_p95_ms']:.3f} | "
            f"{record['mean_abs_mechanical_power_w']:.2f} | "
            f"{record['torque_saturation_ratio']:.3f} | {record['solve_p95_ms']:.3f} |"
        )
    (output / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_std(records: list[dict[str, float | str | int]], key: str) -> str:
    values = np.asarray([float(record[key]) for record in records])
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return f"{values.mean():.3f} ± {standard_deviation:.3f}"


def _paired_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Return a mean and two-sided t interval; an interval needs at least two pairs."""

    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half_width = float(
        student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return mean, mean - half_width, mean + half_width


def _paired_bootstrap_ci(
    values: np.ndarray,
    *,
    resamples: int = 20_000,
    random_seed: int = 20260830,
) -> tuple[float, float, float]:
    """Return a deterministic percentile CI for a mean paired difference."""

    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.integers(0, len(values), size=(resamples, len(values)))
    sampled_means = values[sampled_indices].mean(axis=1)
    lower, upper = np.quantile(sampled_means, (0.025, 0.975))
    return mean, float(lower), float(upper)


def _format_ci(
    estimate: float,
    lower: float,
    upper: float,
    *,
    signed: bool = False,
) -> str:
    value_format = "+.3f" if signed else ".3f"
    estimate_text = format(estimate, value_format)
    if not np.isfinite((lower, upper)).all():
        return f"{estimate_text} [unavailable]"
    return f"{estimate_text} [{format(lower, value_format)}, {format(upper, value_format)}]"


def _records_by_evaluation_seed(
    records: list[dict[str, float | str | int]], controller: str
) -> dict[int, dict[str, float | str | int]]:
    indexed: dict[int, dict[str, float | str | int]] = {}
    for record in records:
        if record["controller"] != controller:
            continue
        seed = int(record["evaluation_seed"])
        if seed in indexed:
            raise ValueError(f"duplicate evaluation_seed {seed} for controller {controller!r}")
        indexed[seed] = record
    return indexed


def write_d1_randomized_audit(records: list[dict[str, float | str | int]], output: Path) -> None:
    controllers = list(dict.fromkeys(str(record["controller"]) for record in records))
    lines = [
        "# D1 randomized-domain audit",
        "",
        "Matched seeds; values are episode mean ± sample standard deviation.",
        "",
        (
            "| Controller | Episodes | Failures | Mean reward | Velocity RMSE | "
            "Pitch RMSE | Height RMSE |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for controller in controllers:
        selected = [record for record in records if record["controller"] == controller]
        failures = sum(1 - int(record["success"]) for record in selected)
        lines.append(
            f"| {controller} | {len(selected)} | {failures} | "
            f"{_mean_std(selected, 'mean_reward')} | "
            f"{_mean_std(selected, 'velocity_rmse_mps')} | "
            f"{_mean_std(selected, 'pitch_rmse_deg')} | "
            f"{_mean_std(selected, 'height_rmse_mm')} |"
        )

    residual_name = "D1 LQR+VMC+PPO"
    baseline_name = "D1 LQR+VMC"
    if residual_name in controllers and baseline_name in controllers:
        baseline_by_seed = _records_by_evaluation_seed(records, baseline_name)
        residual_by_seed = _records_by_evaluation_seed(records, residual_name)
        if baseline_by_seed.keys() != residual_by_seed.keys():
            missing_residual = sorted(baseline_by_seed.keys() - residual_by_seed.keys())
            missing_baseline = sorted(residual_by_seed.keys() - baseline_by_seed.keys())
            raise ValueError(
                "PPO and LQR evaluation seeds differ: "
                f"missing PPO={missing_residual}, missing LQR={missing_baseline}"
            )
        matched_seeds = sorted(baseline_by_seed)
        condition_keys = (
            "state_estimation_mode",
            "latency_compensation",
            "base_mass_scale",
            "damping_scale",
            "friction_scale",
            "actuator_strength_scale",
            "action_delay_steps",
            "state_delay_steps",
            "sensor_noise_scale",
        )
        for seed in matched_seeds:
            baseline_record = baseline_by_seed[seed]
            residual_record = residual_by_seed[seed]
            for key in condition_keys:
                if (
                    key in baseline_record
                    and key in residual_record
                    and baseline_record[key] != residual_record[key]
                ):
                    raise ValueError(f"evaluation condition {key!r} differs at seed {seed}")
        lines += [
            "",
            "## Paired PPO − LQR differences (95% t confidence interval)",
            "",
            f"Matched evaluation seeds: {len(matched_seeds)}.",
            "",
        ]
        for key, label in (
            ("mean_reward", "Mean reward"),
            ("velocity_rmse_mps", "Velocity RMSE [m/s]"),
            ("pitch_rmse_deg", "Pitch RMSE [deg]"),
            ("height_rmse_mm", "Height RMSE [mm]"),
        ):
            baseline_values = np.asarray(
                [float(baseline_by_seed[seed][key]) for seed in matched_seeds]
            )
            residual_values = np.asarray(
                [float(residual_by_seed[seed][key]) for seed in matched_seeds]
            )
            mean, lower, upper = _paired_ci(residual_values - baseline_values)
            lines.append(f"- {label}: {_format_ci(mean, lower, upper, signed=True)}")
    with (output / "randomized_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "randomized_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_d1_scenario(rollouts: list[D1Rollout], output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)
    for rollout in rollouts:
        axes[0].plot(
            rollout.time_s,
            rollout.forward_velocities_mps,
            label=rollout.controller,
        )
        axes[1].plot(rollout.time_s, np.rad2deg(rollout.states[:, 1]))
        axes[2].plot(rollout.time_s, 1e3 * rollout.states[:, 2])
    axes[0].plot(
        rollouts[0].time_s,
        rollouts[0].commands[:, 0],
        "k--",
        linewidth=1.2,
        label="command",
    )
    axes[2].plot(
        rollouts[0].time_s,
        1e3 * rollouts[0].commands[:, 1],
        "k--",
        linewidth=1.2,
    )
    axes[0].set_ylabel("velocity [m/s]")
    axes[1].set_ylabel("pitch [deg]")
    axes[2].set_ylabel("base height [mm]")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(ncol=3, frameon=False)
    axes[0].set_title(f"D1 {rollouts[0].scenario.replace('_', ' ').title()}")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / f"{rollouts[0].scenario}.png", dpi=180)
    plt.close(figure)


def create_d1_comparison_gif(rollouts: list[D1Rollout], output: Path) -> None:
    if any(rollout.frames is None for rollout in rollouts):
        raise ValueError("all rollouts must contain rendered frames")
    frame_count = min(len(rollout.frames or []) for rollout in rollouts)
    composed: list[Image.Image] = []
    for index in range(frame_count):
        panels: list[Image.Image] = []
        for rollout in rollouts:
            frame = Image.fromarray((rollout.frames or [])[index])
            panel = Image.new("RGB", (frame.width, frame.height + 28), "white")
            panel.paste(frame, (0, 28))
            ImageDraw.Draw(panel).text((8, 7), rollout.controller, fill="black")
            panels.append(panel)
        canvas = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
            "white",
        )
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        composed.append(canvas)
    composed[0].save(
        output / "d1_push_comparison.gif",
        save_all=True,
        append_images=composed[1:],
        duration=50,
        loop=0,
    )


def _wilson_interval(successes: int, episodes: int) -> tuple[float, float, float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    estimate = successes / episodes
    z = 1.959963984540054
    denominator = 1.0 + z**2 / episodes
    center = (estimate + z**2 / (2.0 * episodes)) / denominator
    half_width = (
        z
        * np.sqrt(estimate * (1.0 - estimate) / episodes + z**2 / (4.0 * episodes**2))
        / denominator
    )
    return estimate, center - half_width, center + half_width


def _validate_delay_rollout(rollout: D1Rollout, expected_delay_steps: int) -> None:
    if rollout.state_estimation_mode != "estimated":
        raise ValueError("delay sweep rollout did not use estimated state")
    if rollout.state_delay_steps != expected_delay_steps:
        raise ValueError(
            "delay sweep rollout reported state_delay_steps="
            f"{rollout.state_delay_steps}, expected {expected_delay_steps}"
        )
    if len(rollout.time_s) == 0:
        raise ValueError("delay sweep rollout contains no control steps")
    if rollout.state_ages_ms.shape != rollout.time_s.shape:
        raise ValueError("delay sweep must record one state age per control step")
    if not np.isfinite(rollout.state_ages_ms).all():
        raise ValueError("delay sweep state ages must be finite")
    expected_ages_ms = (
        np.minimum(np.arange(1, len(rollout.time_s) + 1), expected_delay_steps)
        * rollout.control_dt_s
        * 1e3
    )
    if not np.allclose(rollout.state_ages_ms, expected_ages_ms, atol=1e-7, rtol=0.0):
        raise ValueError(
            "measured state-age trace does not match the configured delay "
            f"of {expected_delay_steps} steps"
        )
    for name, value in (
        ("initial state", rollout.initial_state_fingerprint),
        ("initial command", rollout.initial_command_fingerprint),
        ("push schedule", rollout.push_schedule_fingerprint),
    ):
        if not value:
            raise ValueError(f"delay sweep is missing the {name} fingerprint")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_d1_state_delay_sweep(
    *,
    output: Path,
    seed: int,
    episodes: int,
    policy: Any | None,
    latency_compensation: str,
    run_metadata: dict[str, Any] | None = None,
) -> list[dict[str, float | str | int]]:
    """Run the fixed, paired D1 state-delay sensitivity experiment."""

    if episodes <= 0:
        raise ValueError("delay sweep requires at least one episode")
    if latency_compensation not in ("none", "constant_velocity"):
        raise ValueError("unsupported latency compensation mode")

    if run_metadata is None:
        provenance = {
            "source": capture_git_provenance(Path(__file__).resolve().parent),
            "checkpoint": {
                "included": policy is not None,
                "path": None,
                "sha256": None,
            },
            "runtime": _runtime_provenance(),
        }
    else:
        # A JSON round trip gives the manifest and config independent, immutable
        # copies of the exact same serializable run provenance.
        provenance = json.loads(json.dumps(run_metadata))
        provenance.setdefault("runtime", _runtime_provenance())

    output.mkdir(parents=True, exist_ok=True)
    for filename in D1_SWEEP_ARTIFACTS:
        (output / filename).unlink(missing_ok=True)

    checkpoint = provenance.get("checkpoint", {})
    if bool(checkpoint.get("included", policy is not None)) != (policy is not None):
        raise ValueError("checkpoint provenance does not match the supplied policy")

    controllers: list[tuple[str, Any | None]] = [("lqr", None), ("mpc", None)]
    if policy is not None:
        controllers.append(("lqr", policy))
    evaluation_seeds = list(range(seed + 100, seed + 100 + episodes))
    records: list[dict[str, float | str | int]] = []
    for delay_steps in D1_STATE_DELAY_SWEEP_STEPS:
        options = {
            "action_delay_steps": 0,
            "state_delay_steps": delay_steps,
            "sensor_noise": 0.0,
        }
        for evaluation_seed in evaluation_seeds:
            for baseline, residual_policy in controllers:
                rollout = run_d1_rollout(
                    baseline,
                    "randomized",
                    evaluation_seed,
                    residual_policy,
                    state_mode="estimated",
                    latency_compensation=latency_compensation,
                    episode_options=options,
                )
                _validate_delay_rollout(rollout, delay_steps)
                records.append(compute_d1_metrics(rollout))

    controller_names = list(dict.fromkeys(str(record["controller"]) for record in records))
    condition_keys = (
        "state_estimation_mode",
        "latency_compensation",
        "base_mass_scale",
        "damping_scale",
        "friction_scale",
        "actuator_strength_scale",
        "action_delay_steps",
        "sensor_noise_scale",
        "state_estimator_seed",
        "initial_state_fingerprint",
        "initial_command_fingerprint",
        "push_schedule_fingerprint",
    )
    for evaluation_seed in evaluation_seeds:
        reference = next(
            record
            for record in records
            if int(record["evaluation_seed"]) == evaluation_seed
            and int(record["state_delay_steps"]) == 0
        )
        for record in records:
            if int(record["evaluation_seed"]) != evaluation_seed:
                continue
            for key in condition_keys:
                if record[key] != reference[key]:
                    raise ValueError(
                        f"delay sweep condition {key!r} differs at seed {evaluation_seed}"
                    )

    summary: list[dict[str, float | str | int]] = []
    for controller in controller_names:
        baseline_records = [
            record
            for record in records
            if record["controller"] == controller and int(record["state_delay_steps"]) == 0
        ]
        baseline_by_seed = {int(record["evaluation_seed"]): record for record in baseline_records}
        if len(baseline_records) != len(baseline_by_seed):
            raise ValueError(f"duplicate zero-delay records for {controller}")
        if set(baseline_by_seed) != set(evaluation_seeds):
            raise ValueError(f"incomplete zero-delay records for {controller}")
        for delay_steps in D1_STATE_DELAY_SWEEP_STEPS:
            selected_records = [
                record
                for record in records
                if record["controller"] == controller
                and int(record["state_delay_steps"]) == delay_steps
            ]
            selected = {int(record["evaluation_seed"]): record for record in selected_records}
            if len(selected_records) != len(selected):
                raise ValueError(f"duplicate delay={delay_steps} records for {controller}")
            if set(selected) != set(evaluation_seeds):
                raise ValueError(f"incomplete delay={delay_steps} records for {controller}")
            for metric, (_, unit) in D1_SWEEP_METRICS.items():
                values = np.asarray([float(selected[item][metric]) for item in evaluation_seeds])
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite sweep metric {metric!r} for {controller}")
                if metric == "success":
                    estimate, lower, upper = _wilson_interval(int(np.sum(values)), len(values))
                    ci_method = "Wilson score"
                else:
                    estimate, lower, upper = _paired_ci(values)
                    ci_method = "mean t"
                deltas = values - np.asarray(
                    [float(baseline_by_seed[item][metric]) for item in evaluation_seeds]
                )
                if metric == "success":
                    delta, delta_lower, delta_upper = _paired_bootstrap_ci(deltas)
                    delta_ci_method = "paired bootstrap"
                else:
                    delta, delta_lower, delta_upper = _paired_ci(deltas)
                    delta_ci_method = "paired t"
                summary.append(
                    {
                        "controller": controller,
                        "state_delay_ms": 10 * delay_steps,
                        "metric": metric,
                        "unit": unit,
                        "episodes": len(values),
                        "estimate": estimate,
                        "ci95_low": lower,
                        "ci95_high": upper,
                        "delta_vs_0ms": delta,
                        "delta_ci95_low": delta_lower,
                        "delta_ci95_high": delta_upper,
                        "ci_method": ci_method,
                        "delta_ci_method": delta_ci_method,
                    }
                )

    with (output / "delay_sweep_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "delay_sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# D1 state-delay sensitivity",
        "",
        (
            "Fixed delay grid: 0/10/20/30/50 ms. Domain samples, initial state, initial "
            "command, estimator seed, and planned push are paired and checked by evaluation seed."
        ),
        "",
        (
            "Success and episode duration are the primary robustness outcomes. Every other "
            "continuous metric uses only the trajectory prefix observed before truncation or "
            "a fall; a smaller error after an early fall is not evidence of better robustness."
        ),
        "",
        (
            "Absolute success intervals use Wilson scores. Absolute continuous intervals use "
            "mean t intervals. Success differences versus 0 ms use a deterministic paired "
            "bootstrap; other differences use paired t intervals. With fewer than two pairs, "
            "the interval is unavailable and stored as NaN in CSV."
        ),
        "",
        "| Controller | Delay [ms] | Metric | Unit | Estimate [95% CI] | Delta vs 0 ms [95% CI] |",
        "|---|---:|---|---|---:|---:|",
    ]
    for item in summary:
        metric = str(item["metric"])
        metric_label, _ = D1_SWEEP_METRICS[metric]
        lines.append(
            f"| {item['controller']} | {item['state_delay_ms']} | {metric_label} | "
            f"{item['unit']} | "
            f"{_format_ci(float(item['estimate']), float(item['ci95_low']), float(item['ci95_high']))} | "
            f"{_format_ci(float(item['delta_vs_0ms']), float(item['delta_ci95_low']), float(item['delta_ci95_high']), signed=True)} |"
        )
    if episodes < 20:
        lines += ["", "Exploratory run: fewer than 20 paired episodes per condition."]
    (output / "state_delay_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex=True)
    panels = (
        ("success", "success rate"),
        ("episode_duration_s", "episode duration [s]"),
        ("velocity_rmse_mps", "velocity RMSE [m/s]"),
        ("max_abs_pitch_deg", "max pitch [deg]"),
    )
    for controller in controller_names:
        for axis, (metric, label) in zip(axes.flat, panels, strict=True):
            selected_summary = [
                item
                for item in summary
                if item["controller"] == controller and item["metric"] == metric
            ]
            estimates = np.asarray(
                [float(item["estimate"]) for item in selected_summary], dtype=np.float64
            )
            lower = np.asarray(
                [float(item["ci95_low"]) for item in selected_summary], dtype=np.float64
            )
            upper = np.asarray(
                [float(item["ci95_high"]) for item in selected_summary], dtype=np.float64
            )
            errors = np.nan_to_num(np.vstack((estimates - lower, upper - estimates)), nan=0.0)
            axis.errorbar(
                [float(item["state_delay_ms"]) for item in selected_summary],
                estimates,
                yerr=np.maximum(errors, 0.0),
                marker="o",
                capsize=2.5,
                label=controller,
            )
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0, 0].set_ylim(0.0, 1.0)
    for axis in axes[-1]:
        axis.set_xlabel("state delay [ms]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "state_delay_sensitivity.png", dpi=180)
    plt.close(figure)

    protocol = {
        "state_mode": "estimated",
        "latency_compensation": latency_compensation,
        "delay_steps": list(D1_STATE_DELAY_SWEEP_STEPS),
        "delay_ms": [10 * item for item in D1_STATE_DELAY_SWEEP_STEPS],
        "evaluation_seeds": evaluation_seeds,
        "controllers": controller_names,
        "action_delay_steps": 0,
        "sensor_noise_scale": 0.0,
        "episodes_per_condition": episodes,
        "exploratory": episodes < 20,
    }
    run_identity = json.dumps(
        {"protocol": protocol, "provenance": provenance}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    run_id = f"d1-delay-{hashlib.sha256(run_identity).hexdigest()[:16]}"
    evaluation_config = {
        "schema": "d1-state-delay-evaluation-config-v2",
        "run_id": run_id,
        "protocol": protocol,
        "provenance": provenance,
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2), encoding="utf-8"
    )
    artifact_names = D1_SWEEP_ARTIFACTS[:-1]
    artifact_hashes = {filename: _file_sha256(output / filename) for filename in artifact_names}
    manifest = {
        "schema": "d1-state-delay-sweep-v2",
        "run_id": run_id,
        "status": "complete",
        "completed": True,
        "protocol": protocol,
        "provenance": provenance,
        "validation": {
            "paired_input_fingerprints": True,
            "measured_state_age_traces": True,
            "record_count": len(records),
            "summary_row_count": len(summary),
        },
        "artifacts": artifact_hashes,
    }
    (output / "delay_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    policy_options = parser.add_mutually_exclusive_group()
    policy_options.add_argument(
        "--policy",
        type=Path,
        default=None,
        help=f"explicit checkpoint (default auto-detects {D1_DEFAULT_POLICY_PATH})",
    )
    policy_options.add_argument(
        "--no-policy",
        action="store_true",
        help="benchmark only LQR and MPC, even when the default checkpoint exists",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result directory (defaults depend on benchmark or delay-sweep mode)",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--audit-episodes", type=int, default=0)
    parser.add_argument(
        "--state-delay-sweep",
        action="store_true",
        help="run the fixed 0/10/20/30/50 ms paired sensitivity experiment",
    )
    parser.add_argument("--state-mode", choices=("oracle", "estimated"), default="oracle")
    parser.add_argument(
        "--latency-compensation",
        choices=("none", "constant_velocity"),
        default="none",
    )
    parser.add_argument("--gif", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.state_mode != "estimated" and args.latency_compensation != "none":
        raise SystemExit("latency compensation requires --state-mode estimated")
    if args.state_delay_sweep and args.state_mode != "estimated":
        raise SystemExit("--state-delay-sweep requires --state-mode estimated")
    if args.state_delay_sweep and args.audit_episodes <= 0:
        raise SystemExit("--state-delay-sweep requires --audit-episodes")
    if args.state_delay_sweep and args.gif:
        raise SystemExit("--state-delay-sweep cannot be combined with --gif")
    if args.output is None:
        if args.state_delay_sweep:
            suffix = "compensated" if args.latency_compensation == "constant_velocity" else "raw"
            args.output = Path(f"results/d1_state_delay_{suffix}")
        else:
            args.output = Path("results/d1_benchmark")

    if args.no_policy:
        policy_path = None
        policy_selection = "disabled"
    elif args.policy is not None:
        if not args.policy.is_file():
            raise SystemExit(f"policy checkpoint not found: {args.policy}")
        policy_path = args.policy
        policy_selection = "explicit"
    elif D1_DEFAULT_POLICY_PATH.is_file():
        policy_path = D1_DEFAULT_POLICY_PATH
        policy_selection = "default"
    else:
        policy_path = None
        policy_selection = "default checkpoint absent"

    source_provenance = capture_git_provenance(Path(__file__).resolve().parent)
    policy_sha256 = None if policy_path is None else _file_sha256(policy_path)
    policy = None
    if policy_path is not None:
        try:
            policy = load_compatible_d1_policy(
                policy_path,
                expected_state_mode=args.state_mode,
                expected_latency_compensation=args.latency_compensation,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise SystemExit(str(error)) from error
    run_metadata = {
        "source": source_provenance,
        "checkpoint": {
            "included": policy is not None,
            "selection": policy_selection,
            "path": None if policy_path is None else str(policy_path),
            "sha256": policy_sha256,
        },
        "runtime": _runtime_provenance(),
    }

    if args.state_delay_sweep:
        run_d1_state_delay_sweep(
            output=args.output,
            seed=args.seed,
            episodes=args.audit_episodes,
            policy=policy,
            latency_compensation=args.latency_compensation,
            run_metadata=run_metadata,
        )
        print((args.output / "state_delay_sensitivity.md").read_text(encoding="utf-8"))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    for filename in D1_SWEEP_ARTIFACTS:
        if filename != "evaluation_config.json":
            (args.output / filename).unlink(missing_ok=True)
    if args.audit_episodes == 0:
        for filename in ("randomized_audit.csv", "randomized_audit.md"):
            (args.output / filename).unlink(missing_ok=True)
    if not args.gif:
        (args.output / "d1_push_comparison.gif").unlink(missing_ok=True)
    evaluation_config = {
        "seed": args.seed,
        "audit_episodes": args.audit_episodes,
        "state_delay_sweep": args.state_delay_sweep,
        "state_mode": args.state_mode,
        "latency_compensation": args.latency_compensation,
        "policy": None if policy_path is None else str(policy_path),
        "policy_sha256": policy_sha256,
        "provenance": run_metadata,
    } | source_provenance
    (args.output / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2), encoding="utf-8"
    )

    all_rollouts: list[D1Rollout] = []
    for scenario in D1_SCENARIOS:
        capture = bool(args.gif and scenario == "push")
        scenario_rollouts = [
            run_d1_rollout(
                "lqr",
                scenario,
                args.seed,
                capture=capture,
                state_mode=args.state_mode,
                latency_compensation=args.latency_compensation,
            ),
            run_d1_rollout(
                "mpc",
                scenario,
                args.seed,
                capture=capture,
                state_mode=args.state_mode,
                latency_compensation=args.latency_compensation,
            ),
        ]
        if policy is not None:
            scenario_rollouts.append(
                run_d1_rollout(
                    "lqr",
                    scenario,
                    args.seed,
                    policy,
                    capture=capture,
                    state_mode=args.state_mode,
                    latency_compensation=args.latency_compensation,
                )
            )
        all_rollouts.extend(scenario_rollouts)
        plot_d1_scenario(scenario_rollouts, args.output)
        if capture:
            create_d1_comparison_gif(scenario_rollouts, args.output)

    records = [compute_d1_metrics(rollout) for rollout in all_rollouts]
    write_d1_metrics(records, args.output)
    if args.audit_episodes > 0:
        audit_records: list[dict[str, float | str | int]] = []
        for offset in range(args.audit_episodes):
            seed = args.seed + 100 + offset
            rollouts = [
                run_d1_rollout(
                    "lqr",
                    "randomized",
                    seed,
                    state_mode=args.state_mode,
                    latency_compensation=args.latency_compensation,
                ),
                run_d1_rollout(
                    "mpc",
                    "randomized",
                    seed,
                    state_mode=args.state_mode,
                    latency_compensation=args.latency_compensation,
                ),
            ]
            if policy is not None:
                rollouts.append(
                    run_d1_rollout(
                        "lqr",
                        "randomized",
                        seed,
                        policy,
                        state_mode=args.state_mode,
                        latency_compensation=args.latency_compensation,
                    )
                )
            audit_records.extend(compute_d1_metrics(rollout) for rollout in rollouts)
        write_d1_randomized_audit(audit_records, args.output)
    print((args.output / "metrics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
