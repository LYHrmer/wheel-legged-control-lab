"""Benchmark and render the full-body D1 LQR/MPC/residual-RL stack."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import t as student_t

from .env import D1ResidualEnv
from .model import JOINT_TORQUE_LIMIT

plt.switch_backend("Agg")

D1_SCENARIOS = ("nominal", "push", "mismatch_delay")


@dataclass
class D1Rollout:
    controller: str
    scenario: str
    time_s: np.ndarray
    states: np.ndarray
    torques: np.ndarray
    commands: np.ndarray
    pushes: np.ndarray
    rewards: np.ndarray
    wheel_contacts: np.ndarray
    undesired_contacts: np.ndarray
    solve_times_ms: np.ndarray
    terminated: bool
    frames: list[np.ndarray] | None = None


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
            "delay_steps": 3,
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
) -> D1Rollout:
    env = D1ResidualEnv(baseline=baseline, randomize=False)
    observation, _ = env.reset(seed=seed, options=d1_scenario_options(scenario))
    renderer = (
        mujoco.Renderer(env.plant.model, height=300, width=400) if capture else None
    )
    states: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    commands: list[tuple[float, float]] = []
    pushes: list[float] = []
    rewards: list[float] = []
    contacts: list[int] = []
    undesired: list[int] = []
    solve_times: list[float] = []
    frames: list[np.ndarray] = []
    terminated = truncated = False

    while not (terminated or truncated):
        if policy is None:
            action = np.zeros(2, dtype=np.float32)
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(info["state"])
        torques.append(info["torque_nm"])
        commands.append((info["command_velocity_mps"], info["command_height_m"]))
        pushes.append(info["push_force_n"])
        rewards.append(reward)
        contacts.append(info["wheel_contacts"])
        undesired.append(info["undesired_contacts"])
        solve_times.append(info["solve_ms"])
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
        torques=np.asarray(torques),
        commands=np.asarray(commands),
        pushes=np.asarray(pushes),
        rewards=np.asarray(rewards),
        wheel_contacts=np.asarray(contacts),
        undesired_contacts=np.asarray(undesired),
        solve_times_ms=np.asarray(solve_times),
        terminated=terminated,
        frames=frames if capture else None,
    )
    env.close()
    return rollout


def compute_d1_metrics(rollout: D1Rollout) -> dict[str, float | str | int]:
    velocity_error = rollout.states[:, 3] - rollout.commands[:, 0]
    pitch_deg = np.rad2deg(rollout.states[:, 1])
    height_error = rollout.states[:, 2] - rollout.commands[:, 1]
    normalized_torque = rollout.torques / JOINT_TORQUE_LIMIT
    return {
        "controller": rollout.controller,
        "scenario": rollout.scenario,
        "success": int(not rollout.terminated),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
        "pitch_rmse_deg": float(np.sqrt(np.mean(pitch_deg**2))),
        "max_abs_pitch_deg": float(np.max(np.abs(pitch_deg))),
        "height_rmse_mm": float(1e3 * np.sqrt(np.mean(height_error**2))),
        "normalized_torque_rms": float(np.sqrt(np.mean(normalized_torque**2))),
        "four_wheel_contact_ratio": float(np.mean(rollout.wheel_contacts == 4)),
        "undesired_contact_steps": int(np.count_nonzero(rollout.undesired_contacts)),
        "mean_reward": float(np.mean(rollout.rewards)),
        "solve_p95_ms": float(np.percentile(rollout.solve_times_ms, 95)),
    }


def write_d1_metrics(
    records: list[dict[str, float | str | int]], output: Path
) -> None:
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# D1 full-body control benchmark",
        "",
        (
            "| Controller | Scenario | Success | Velocity RMSE [m/s] | Pitch RMSE [deg] | "
            "Height RMSE [mm] | Torque RMS | 4-wheel contact | Solve P95 [ms] |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['controller']} | {record['scenario']} | {record['success']} | "
            f"{record['velocity_rmse_mps']:.3f} | {record['pitch_rmse_deg']:.3f} | "
            f"{record['height_rmse_mm']:.2f} | {record['normalized_torque_rms']:.3f} | "
            f"{record['four_wheel_contact_ratio']:.3f} | {record['solve_p95_ms']:.3f} |"
        )
    (output / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_std(records: list[dict[str, float | str | int]], key: str) -> str:
    values = np.asarray([float(record[key]) for record in records])
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def _paired_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half_width = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half_width, mean + half_width


def write_d1_randomized_audit(
    records: list[dict[str, float | str | int]], output: Path
) -> None:
    with (output / "randomized_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    controllers = list(dict.fromkeys(str(record["controller"]) for record in records))
    lines = [
        "# D1 randomized-domain audit",
        "",
        "Matched seeds; values are episode mean ± sample standard deviation.",
        "",
        "| Controller | Episodes | Failures | Velocity RMSE | Pitch RMSE | Height RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for controller in controllers:
        selected = [record for record in records if record["controller"] == controller]
        failures = sum(1 - int(record["success"]) for record in selected)
        lines.append(
            f"| {controller} | {len(selected)} | {failures} | "
            f"{_mean_std(selected, 'velocity_rmse_mps')} | "
            f"{_mean_std(selected, 'pitch_rmse_deg')} | "
            f"{_mean_std(selected, 'height_rmse_mm')} |"
        )

    residual_name = "D1 LQR+VMC+PPO"
    baseline_name = "D1 LQR+VMC"
    if residual_name in controllers and baseline_name in controllers:
        lines += ["", "## Paired PPO − LQR differences (95% t confidence interval)", ""]
        for key, label in (
            ("velocity_rmse_mps", "Velocity RMSE [m/s]"),
            ("pitch_rmse_deg", "Pitch RMSE [deg]"),
            ("height_rmse_mm", "Height RMSE [mm]"),
        ):
            baseline_values = np.asarray(
                [float(r[key]) for r in records if r["controller"] == baseline_name]
            )
            residual_values = np.asarray(
                [float(r[key]) for r in records if r["controller"] == residual_name]
            )
            mean, lower, upper = _paired_ci(residual_values - baseline_values)
            lines.append(f"- {label}: {mean:+.3f} [{lower:+.3f}, {upper:+.3f}]")
    (output / "randomized_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_d1_scenario(rollouts: list[D1Rollout], output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)
    for rollout in rollouts:
        axes[0].plot(rollout.time_s, rollout.states[:, 3], label=rollout.controller)
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
    parser.add_argument("--gif", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    policy = None
    if args.policy.exists():
        from stable_baselines3 import PPO

        policy = PPO.load(args.policy, device="cpu")

    all_rollouts: list[D1Rollout] = []
    for scenario in D1_SCENARIOS:
        capture = bool(args.gif and scenario == "push")
        scenario_rollouts = [
            run_d1_rollout("lqr", scenario, args.seed, capture=capture),
            run_d1_rollout("mpc", scenario, args.seed, capture=capture),
        ]
        if policy is not None:
            scenario_rollouts.append(
                run_d1_rollout("lqr", scenario, args.seed, policy, capture=capture)
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
                run_d1_rollout("lqr", "randomized", seed),
                run_d1_rollout("mpc", "randomized", seed),
            ]
            if policy is not None:
                rollouts.append(run_d1_rollout("lqr", "randomized", seed, policy))
            audit_records.extend(compute_d1_metrics(rollout) for rollout in rollouts)
        write_d1_randomized_audit(audit_records, args.output)
    print((args.output / "metrics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
