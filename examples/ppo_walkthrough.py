"""Inspect GAE boundaries and PPO clipping without training a network."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from wheel_legged_control.ppo_learning import clipped_surrogate, generalized_advantage_estimate


def run_walkthrough() -> dict:
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.5, 0.6, 0.7])
    next_values = np.array([0.6, 0.7, 0.8])
    no_boundary = np.zeros(3, dtype=bool)
    last_boundary = np.array([False, False, True])
    gamma, gae_lambda = 0.9, 0.8
    timeout_advantage, timeout_return = generalized_advantage_estimate(
        rewards, values, next_values, no_boundary, last_boundary, gamma, gae_lambda
    )
    terminal_advantage, _ = generalized_advantage_estimate(
        rewards, values, next_values, last_boundary, no_boundary, gamma, gae_lambda
    )
    td = rewards + gamma * next_values - values
    print("Synthetic three-transition rollout: gamma=0.9, lambda=0.8")
    print(" t  reward   V(s_t) V(final_t)   TD       GAE(timeout) GAE(terminal)")
    trajectory_rows = []
    for step in range(3):
        row = {
            "step": step,
            "reward": float(rewards[step]),
            "value": float(values[step]),
            "next_value_before_reset": float(next_values[step]),
            "td_timeout": float(td[step]),
            "advantage_timeout": float(timeout_advantage[step]),
            "advantage_terminal": float(terminal_advantage[step]),
            "return_timeout": float(timeout_return[step]),
        }
        trajectory_rows.append(row)
        print(
            f" {step}  {rewards[step]:6.2f}   {values[step]:5.2f}    {next_values[step]:5.2f}"
            f"   {td[step]:6.3f}       {timeout_advantage[step]:6.3f}"
            f"        {terminal_advantage[step]:6.3f}"
        )

    ratios = np.array([1.5, 0.5, 1.5, 0.5])
    advantage = np.array([2.0, 2.0, -2.0, -2.0])
    surrogate = clipped_surrogate(np.zeros(4), np.log(ratios), advantage)
    print("\nPPO objective (maximize minimum; optimizer minimizes its negative mean)")
    print(" ratio     A     ratio*A  clip(ratio)*A   minimum")
    clipping_rows = []
    for index in range(4):
        row = {
            "ratio": float(surrogate.ratio[index]),
            "advantage": float(advantage[index]),
            "unclipped": float(surrogate.unclipped[index]),
            "clipped": float(surrogate.clipped[index]),
            "minimum": float(surrogate.minimum[index]),
        }
        clipping_rows.append(row)
        print(
            f" {ratios[index]:5.2f}  {advantage[index]:5.1f}"
            f"    {surrogate.unclipped[index]:6.2f}"
            f"       {surrogate.clipped[index]:6.2f}     {surrogate.minimum[index]:6.2f}"
        )
    print(f"Actor loss = {surrogate.loss:.3f}; A=-2, ratio=1.5 still uses the unclipped branch.")

    discount_rows = []
    print("\nPhysical time at gamma=0.99 (e-folding time, not a hard horizon)")
    for control_dt in (0.02, 0.01):
        row = {
            "control_dt_s": control_dt,
            "gamma": 0.99,
            "discount_time_constant_s": float(-control_dt / np.log(0.99)),
            "weight_after_one_second": float(0.99 ** (1.0 / control_dt)),
            "rollout_256_steps_s": 256 * control_dt,
        }
        discount_rows.append(row)
        print(
            f" dt={control_dt * 1000:2.0f} ms: tau={row['discount_time_constant_s']:.3f} s,"
            f" weight at 1 s={row['weight_after_one_second']:.3f},"
            f" 256 steps={row['rollout_256_steps_s']:.2f} s"
        )
    equivalent_gamma = float(0.99 ** (0.01 / 0.02))
    print(f"Equal discount time constant at 10 ms needs gamma={equivalent_gamma:.8f}.")
    print("This changes neither the existing training configuration nor any checkpoint.")
    return {
        "schema": "ppo-learning-fixtures-v1",
        "synthetic_only": True,
        "gae_gamma": gamma,
        "gae_lambda": gae_lambda,
        "trajectory": trajectory_rows,
        "clipping": clipping_rows,
        "actor_loss": surrogate.loss,
        "discount_timing": discount_rows,
        "equivalent_gamma_at_10ms": equivalent_gamma,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory for JSON and CSV tables")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists; refusing to overwrite: {args.output}")
    report = run_walkthrough()
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "walkthrough.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        for name in ("trajectory", "clipping", "discount_timing"):
            rows = report[name]
            with (args.output / f"{name}.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Saved synthetic numerical tables to {args.output}")


if __name__ == "__main__":
    main()
