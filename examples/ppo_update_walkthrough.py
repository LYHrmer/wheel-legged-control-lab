"""Print one real SGD update of a tiny PPO actor/critic on synthetic data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from wheel_legged_control.ppo_update import (
    UpdateConfig,
    clipping_fixture,
    one_sgd_update,
    synthetic_rollout,
)
from wheel_legged_control.provenance import capture_git_provenance


def run_walkthrough(seed: int = 7, config: UpdateConfig | None = None) -> dict:
    config = UpdateConfig() if config is None else config
    actor, critic, batch, rollout = synthetic_rollout(seed)
    trace = one_sgd_update(actor, critic, batch, config)
    samples = []
    for step in range(len(batch.actions)):
        row = {
            "step": step,
            "observation_velocity": float(batch.observations[step, 0]),
            "observation_reference": float(batch.observations[step, 1]),
            "action_raw": float(batch.actions[step, 0]),
            "reward": rollout["rewards"][step],
            "next_value": rollout["next_values"][step],
            "terminated": rollout["terminated"][step],
            "truncated": rollout["truncated"][step],
            "old_log_prob": float(batch.old_log_prob[step]),
            "advantage": float(batch.advantages[step]),
            "return_target": float(batch.returns[step]),
        }
        for stage in ("before", "after"):
            for name in ("log_prob", "value", "ratio", "unclipped", "clipped", "minimum"):
                row[f"{name}_{stage}"] = trace[stage][name][step]
            unclipped, clipped = row[f"unclipped_{stage}"], row[f"clipped_{stage}"]
            row[f"branch_{stage}"] = (
                "tie" if unclipped == clipped else "clipped" if clipped < unclipped else "unclipped"
            )
        samples.append(row)
    return {
        "schema": "ppo-single-sgd-update-v1",
        "synthetic_only": True,
        "config": asdict(config),
        "choices": {
            "optimizer": "SGD without momentum or weight decay",
            "device": "cpu",
            "dtype": "float64",
            "advantage_normalization": False,
            "gradient_clipping": False,
            "value_clipping": False,
            "action_squashing": False,
            "actor_critic_shared_parameters": False,
        },
        "rollout": rollout,
        "batch": {field.name: getattr(batch, field.name).tolist() for field in fields(batch)},
        "update": trace,
        "samples": samples,
        "clipping_fixture": clipping_fixture(config.clip_range),
    }


def print_trace(report: dict) -> None:
    print("Synthetic 8-step on-policy batch; CPU float64; exactly one SGD optimizer.step().")
    print(" t    action         A    ratio_before ratio_after    V_before    V_after")
    for row in report["samples"]:
        print(
            f" {row['step']} {row['action_raw']:9.5f} {row['advantage']:9.5f}"
            f"     {row['ratio_before']:9.5f}   {row['ratio_after']:9.5f}"
            f"   {row['value_before']:9.5f}  {row['value_after']:9.5f}"
        )
    trace = report["update"]
    for name in ("actor_loss", "value_loss", "entropy", "total_loss"):
        print(f"{name}: {trace['before'][name]:.9f} -> {trace['after'][name]:.9f}")
    print("\nParameter check: after = before - learning_rate * gradient")
    for row in trace["parameters"]:
        print(
            f" {row['parameter']}[{row['flat_index']}]"
            f" {row['before']:+.9f} - lr * ({row['gradient']:+.9f}) = {row['after']:+.9f}"
        )
    print("\nSeparate clipping fixture (not rollout data):")
    for row in report["clipping_fixture"]:
        print(
            f" ratio={row['ratio']:.1f}, A={row['advantage']:+.1f},"
            f" objective={row['minimum']:+.2f},"
            f" d(objective)/d(log_prob)={row['objective_derivative_wrt_log_prob']:+.2f}"
        )
    print("The initial policy equals the sampling policy: before-update ratios are 1.")
    print("Losses describe this frozen batch, not new-policy return or D1 control performance.")


def save_report(output: Path, report: dict) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_paths = (
        "src/wheel_legged_control/ppo_update.py",
        "src/wheel_legged_control/ppo_learning.py",
        "src/wheel_legged_control/provenance.py",
        "examples/ppo_update_walkthrough.py",
        "pyproject.toml",
    )
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "source_sha256": {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in source_paths
        },
    }
    # Source hashes identify uncommitted code; HEAD alone cannot do that.
    git = capture_git_provenance(repository)
    provenance["git_head"] = git["git_commit"]
    provenance["git_dirty"] = git["git_dirty"]
    provenance["git_worktree_sha256"] = git["git_worktree_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "update.json").open("x", encoding="utf-8") as stream:
        json.dump({**report, "provenance": provenance}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for filename, rows in (
        ("samples.csv", report["samples"]),
        ("parameters.csv", report["update"]["parameters"]),
        ("clipping_fixture.csv", report["clipping_fixture"]),
    ):
        with (output / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "schema": "ppo-update-artifacts-v1",
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
        },
        "excludes": ["manifest.json", "README.md"],
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory for the full numerical trace")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists; refusing to overwrite: {args.output}")
    try:
        config = UpdateConfig(args.learning_rate, args.clip_range, args.vf_coef, args.ent_coef)
        report = run_walkthrough(args.seed, config)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    print_trace(report)
    if args.output is not None:
        save_report(args.output, report)
        print(f"Saved update trace to {args.output}")


if __name__ == "__main__":
    main()
