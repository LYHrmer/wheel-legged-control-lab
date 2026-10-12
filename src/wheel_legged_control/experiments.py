"""Evaluate LQR, linear MPC, and an optional residual PPO policy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from .env import WheelLeggedResidualEnv

plt.switch_backend("Agg")


SCENARIOS = ("nominal", "push", "mismatch_delay")


@dataclass
class Rollout:
    controller: str
    scenario: str
    time_s: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    commands: np.ndarray
    pushes: np.ndarray
    rewards: np.ndarray
    terminated: bool
    solve_times_ms: np.ndarray


def scenario_options(name: str) -> dict[str, Any]:
    if name == "randomized":
        return {"scenario": "training", "randomize": True}
    common: dict[str, Any] = {"scenario": name, "randomize": False}
    if name == "mismatch_delay":
        common |= {
            "mass_scale": 1.25,
            "damping_scale": 1.35,
            "delay_steps": 2,
            "sensor_noise": 1.0,
        }
    return common


def run_rollout(
    baseline: str,
    scenario: str,
    seed: int,
    policy: Any | None = None,
) -> Rollout:
    env = WheelLeggedResidualEnv(baseline=baseline, randomize=False)
    observation, _ = env.reset(seed=seed, options=scenario_options(scenario))
    states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    commands: list[tuple[float, float]] = []
    pushes: list[float] = []
    rewards: list[float] = []
    solve_times: list[float] = []
    terminated = truncated = False

    while not (terminated or truncated):
        if policy is None:
            action = np.zeros(2, dtype=np.float32)
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(info["state"])
        controls.append(info["control"])
        commands.append(
            (info["command_velocity_mps"], info["command_leg_extension_m"])
        )
        pushes.append(info["push_force_n"])
        rewards.append(reward)
        solve_times.append(float(getattr(env.controller, "last_solve_ms", 0.0)))

    label = "LQR + PPO" if policy is not None else ("LQR" if baseline == "lqr" else "MPC")
    rollout = Rollout(
        controller=label,
        scenario=scenario,
        time_s=np.arange(len(states)) * env.plant.control_dt,
        states=np.asarray(states),
        controls=np.asarray(controls),
        commands=np.asarray(commands),
        pushes=np.asarray(pushes),
        rewards=np.asarray(rewards),
        terminated=terminated,
        solve_times_ms=np.asarray(solve_times),
    )
    env.close()
    return rollout


def compute_metrics(rollout: Rollout) -> dict[str, float | str | int]:
    velocity_error = rollout.states[:, 3] - rollout.commands[:, 0]
    height_error = rollout.states[:, 2] - rollout.commands[:, 1]
    pitch_deg = np.rad2deg(rollout.states[:, 1])
    normalized_control = rollout.controls / np.array([80.0, 260.0])
    return {
        "controller": rollout.controller,
        "scenario": rollout.scenario,
        "success": int(not rollout.terminated),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
        "pitch_rmse_deg": float(np.sqrt(np.mean(pitch_deg**2))),
        "max_abs_pitch_deg": float(np.max(np.abs(pitch_deg))),
        "height_rmse_mm": float(1e3 * np.sqrt(np.mean(height_error**2))),
        "normalized_effort_rms": float(np.sqrt(np.mean(normalized_control**2))),
        "mean_reward": float(np.mean(rollout.rewards)),
        "solve_p95_ms": float(np.percentile(rollout.solve_times_ms, 95)),
    }


def write_metrics(records: list[dict[str, float | str | int]], output: Path) -> None:
    fieldnames = list(records[0])
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Wheel-legged control benchmark",
        "",
        (
            "| Controller | Scenario | Success | Velocity RMSE [m/s] | Pitch RMSE [deg] | "
            "Height RMSE [mm] | Effort RMS | Solve P95 [ms] |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['controller']} | {record['scenario']} | {record['success']} | "
            f"{record['velocity_rmse_mps']:.3f} | {record['pitch_rmse_deg']:.3f} | "
            f"{record['height_rmse_mm']:.2f} | {record['normalized_effort_rms']:.3f} | "
            f"{record['solve_p95_ms']:.3f} |"
        )
    (output / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_randomized_audit(
    records: list[dict[str, float | str | int]], output: Path
) -> None:
    fieldnames = list(records[0])
    with (output / "randomized_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    controllers = list(dict.fromkeys(str(record["controller"]) for record in records))
    lines = [
        "# Randomized-domain audit",
        "",
        "Each value is the episode mean ± standard deviation over matched random seeds.",
        "",
        (
            "| Controller | Episodes | Failures | Mean reward | Velocity RMSE [m/s] | "
            "Pitch RMSE [deg] | Height RMSE [mm] |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def mean_std(selected: list[dict[str, float | str | int]], key: str) -> str:
        values = np.asarray([float(record[key]) for record in selected])
        return f"{values.mean():.3f} ± {values.std(ddof=0):.3f}"

    for controller in controllers:
        selected = [record for record in records if record["controller"] == controller]
        failures = sum(1 - int(record["success"]) for record in selected)
        lines.append(
            f"| {controller} | {len(selected)} | {failures} | "
            f"{mean_std(selected, 'mean_reward')} | "
            f"{mean_std(selected, 'velocity_rmse_mps')} | "
            f"{mean_std(selected, 'pitch_rmse_deg')} | "
            f"{mean_std(selected, 'height_rmse_mm')} |"
        )
    (output / "randomized_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scenario(rollouts: list[Rollout], output: Path) -> None:
    scenario = rollouts[0].scenario
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
    axes[2].set_ylabel("leg offset [mm]")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(ncol=4, frameon=False)
    axes[0].set_title(scenario.replace("_", " ").title())
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / f"{scenario}.png", dpi=180)
    plt.close(figure)


def plot_summary(records: list[dict[str, float | str | int]], output: Path) -> None:
    controllers = list(dict.fromkeys(str(record["controller"]) for record in records))
    scenarios = list(dict.fromkeys(str(record["scenario"]) for record in records))
    metrics = (
        ("velocity_rmse_mps", "Velocity RMSE [m/s]"),
        ("pitch_rmse_deg", "Pitch RMSE [deg]"),
        ("height_rmse_mm", "Height RMSE [mm]"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    width = 0.8 / len(controllers)
    x = np.arange(len(scenarios))
    for metric_index, (key, title) in enumerate(metrics):
        for controller_index, controller in enumerate(controllers):
            values = [
                next(
                    float(record[key])
                    for record in records
                    if record["controller"] == controller and record["scenario"] == scenario
                )
                for scenario in scenarios
            ]
            axes[metric_index].bar(
                x + (controller_index - (len(controllers) - 1) / 2) * width,
                values,
                width,
                label=controller,
            )
        axes[metric_index].set_title(title)
        axes[metric_index].set_xticks(x, [name.replace("_", "\n") for name in scenarios])
        axes[metric_index].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "summary.png", dpi=180)
    plt.close(figure)


def create_comparison_gif(rollouts: list[Rollout], output: Path) -> None:
    frames = range(0, min(len(rollout.states) for rollout in rollouts), 2)
    figure, axes = plt.subplots(1, len(rollouts), figsize=(4.0 * len(rollouts), 4.0))
    axes_array = np.atleast_1d(axes)
    wheel_radius = 0.13
    drawings: list[tuple[Any, Any, Any, Any]] = []

    for axis, rollout in zip(axes_array, rollouts):
        axis.set_xlim(-0.65, 0.65)
        axis.set_ylim(-0.03, 0.95)
        axis.set_aspect("equal")
        axis.grid(alpha=0.18)
        axis.axhline(0.0, color="0.25", linewidth=1.5)
        wheel = Circle((0.0, wheel_radius), wheel_radius, color="0.12")
        axis.add_patch(wheel)
        (leg,) = axis.plot([], [], color="#ef9b20", linewidth=7, solid_capstyle="round")
        (body,) = axis.plot([], [], color="#2f6fbd", linewidth=18, solid_capstyle="round")
        text = axis.text(0.03, 0.96, "", transform=axis.transAxes, va="top")
        axis.set_title(rollout.controller)
        axis.set_xticks([])
        drawings.append((wheel, leg, body, text))

    def update(frame_index: int) -> list[Any]:
        artists: list[Any] = []
        for rollout, drawing in zip(rollouts, drawings):
            wheel, leg, body, text = drawing
            state = rollout.states[frame_index]
            pitch = state[1]
            length = 0.48 + state[2]
            pivot = np.array([0.0, wheel_radius])
            direction = np.array([np.sin(pitch), np.cos(pitch)])
            shoulder = pivot + length * direction
            top = shoulder + 0.30 * direction
            leg.set_data((pivot[0], shoulder[0]), (pivot[1], shoulder[1]))
            body.set_data((shoulder[0], top[0]), (shoulder[1], top[1]))
            text.set_text(
                f"t={rollout.time_s[frame_index]:.1f}s\n"
                f"v={state[3]:+.2f} m/s\nθ={np.rad2deg(pitch):+.1f}°"
                + (
                    f"\nPUSH={rollout.pushes[frame_index]:+.0f} N"
                    if rollout.pushes[frame_index] != 0.0
                    else ""
                )
            )
            artists.extend((wheel, leg, body, text))
        return artists

    animation = FuncAnimation(figure, update, frames=frames, interval=40, blit=False)
    animation.save(output / "push_comparison.gif", writer=PillowWriter(fps=25))
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("results/residual_ppo/model.zip"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--audit-episodes", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    policy = None
    if args.policy.exists():
        from stable_baselines3 import PPO

        policy = PPO.load(args.policy, device="cpu")

    all_rollouts: list[Rollout] = []
    for scenario in SCENARIOS:
        scenario_rollouts = [
            run_rollout("lqr", scenario, args.seed),
            run_rollout("mpc", scenario, args.seed),
        ]
        if policy is not None:
            scenario_rollouts.append(run_rollout("lqr", scenario, args.seed, policy))
        all_rollouts.extend(scenario_rollouts)
        plot_scenario(scenario_rollouts, args.output)

    records = [compute_metrics(rollout) for rollout in all_rollouts]
    write_metrics(records, args.output)
    plot_summary(records, args.output)
    if args.gif:
        push_rollouts = [rollout for rollout in all_rollouts if rollout.scenario == "push"]
        create_comparison_gif(push_rollouts, args.output)
    if args.audit_episodes > 0:
        if policy is None:
            raise SystemExit("--audit-episodes requires an existing --policy checkpoint")
        audit_records: list[dict[str, float | str | int]] = []
        for offset in range(args.audit_episodes):
            seed = args.seed + 100 + offset
            audit_records.extend(
                compute_metrics(rollout)
                for rollout in (
                    run_rollout("lqr", "randomized", seed),
                    run_rollout("mpc", "randomized", seed),
                    run_rollout("lqr", "randomized", seed, policy),
                )
            )
        write_randomized_audit(audit_records, args.output)
    print((args.output / "metrics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
