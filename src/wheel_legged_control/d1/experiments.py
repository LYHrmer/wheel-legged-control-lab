"""Benchmark and render the full-body D1 LQR/MPC/residual-RL stack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, field
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
    absolute_mechanical_power_w: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    torque_saturation_fractions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


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


def run_d1_rollout(
    baseline: str,
    scenario: str,
    seed: int,
    policy: Any | None = None,
    *,
    capture: bool = False,
    state_mode: str = "oracle",
) -> D1Rollout:
    env = D1ResidualEnv(baseline=baseline, randomize=False, state_mode=state_mode)
    observation, reset_info = env.reset(seed=seed, options=d1_scenario_options(scenario))
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
    frames: list[np.ndarray] = []
    state_estimation_mode = str(reset_info.get("state_estimation_mode", state_mode))
    domain = {key: float(value) for key, value in reset_info.get("domain", {}).items()}
    action_delay_steps = int(reset_info.get("action_delay_steps", 0))
    state_delay_steps = int(reset_info.get("state_delay_steps", 0))
    sensor_noise_scale = float(reset_info.get("sensor_noise_scale", 0.0))
    state_estimator_seed = reset_info.get("state_estimator_seed")
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
    )
    env.close()
    return rollout


def compute_d1_metrics(rollout: D1Rollout) -> dict[str, float | str | int]:
    velocity_error = rollout.forward_velocities_mps - rollout.commands[:, 0]
    pitch_deg = np.rad2deg(rollout.states[:, 1])
    height_error = rollout.states[:, 2] - rollout.commands[:, 1]
    normalized_torque = rollout.torques / JOINT_TORQUE_LIMIT
    state_age_p95_ms = (
        float(np.percentile(rollout.state_ages_ms, 95)) if rollout.state_ages_ms.size else 0.0
    )
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
        "state_age_p95_ms": state_age_p95_ms,
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
            "| Controller | Scenario | Seed | State | Success | Velocity RMSE [m/s] | "
            "Pitch RMSE [deg] | Height RMSE [mm] | State age P95 [ms] | Power [W] | "
            "Torque saturation | Solve P95 [ms] |"
        ),
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['controller']} | {record['scenario']} | "
            f"{record['evaluation_seed']} | {record['state_estimation_mode']} | "
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
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half_width = float(
        student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return mean, mean - half_width, mean + half_width


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
            lines.append(f"- {label}: {mean:+.3f} [{lower:+.3f}, {upper:+.3f}]")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("results/d1_residual_ppo/model.zip"))
    parser.add_argument("--output", type=Path, default=Path("results/d1_benchmark"))
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--audit-episodes", type=int, default=0)
    parser.add_argument("--state-mode", choices=("oracle", "estimated"), default="oracle")
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="benchmark only LQR and MPC, even when the default checkpoint exists",
    )
    parser.add_argument("--gif", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_episodes == 0:
        for filename in ("randomized_audit.csv", "randomized_audit.md"):
            (args.output / filename).unlink(missing_ok=True)
    if not args.gif:
        (args.output / "d1_push_comparison.gif").unlink(missing_ok=True)

    policy_path = None if args.no_policy or not args.policy.exists() else args.policy
    evaluation_config = {
        "seed": args.seed,
        "audit_episodes": args.audit_episodes,
        "state_mode": args.state_mode,
        "policy": None if policy_path is None else str(policy_path),
        "policy_sha256": (
            None if policy_path is None else hashlib.sha256(policy_path.read_bytes()).hexdigest()
        ),
    } | capture_git_provenance(Path(__file__).resolve().parent)
    (args.output / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2), encoding="utf-8"
    )
    policy = None
    if policy_path is not None:
        try:
            policy = load_compatible_d1_policy(
                policy_path,
                expected_state_mode=args.state_mode,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise SystemExit(str(error)) from error

    all_rollouts: list[D1Rollout] = []
    for scenario in D1_SCENARIOS:
        capture = bool(args.gif and scenario == "push")
        scenario_rollouts = [
            run_d1_rollout("lqr", scenario, args.seed, capture=capture, state_mode=args.state_mode),
            run_d1_rollout("mpc", scenario, args.seed, capture=capture, state_mode=args.state_mode),
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
                run_d1_rollout("lqr", "randomized", seed, state_mode=args.state_mode),
                run_d1_rollout("mpc", "randomized", seed, state_mode=args.state_mode),
            ]
            if policy is not None:
                rollouts.append(
                    run_d1_rollout(
                        "lqr",
                        "randomized",
                        seed,
                        policy,
                        state_mode=args.state_mode,
                    )
                )
            audit_records.extend(compute_d1_metrics(rollout) for rollout in rollouts)
        write_d1_randomized_audit(audit_records, args.output)
    print((args.output / "metrics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
