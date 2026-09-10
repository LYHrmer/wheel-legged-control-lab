"""Independent NumPy checks of saved GAE and diagonal-Gaussian PPO updates.

No model is loaded or trained. The saved first-128 env-major fragment cannot
reconstruct minibatch-normalized optimizer losses or prove policy improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(actual, expected, label, *, atol=2e-5, rtol=2e-5):
    require(np.shape(actual) == np.shape(expected), f"{label}: shape mismatch")
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), f"{label}: numerical mismatch")
    return float(np.max(np.abs(np.asarray(actual) - expected)))


def gae_from_td_sums(rewards, values, starts, last_values, last_dones, gamma, lam):
    """Expanded discounted TD sums; intentionally not the learner's recurrence."""
    steps, envs = rewards.shape
    advantages = np.zeros((steps, envs))
    for env in range(envs):
        for first in range(steps):
            weight = 1.0
            for tick in range(first, steps):
                done = last_dones[env] if tick == steps - 1 else starts[tick + 1, env]
                successor = last_values[env] if tick == steps - 1 else values[tick + 1, env]
                delta = rewards[tick, env] - values[tick, env] + gamma * successor * (1 - done)
                advantages[first, env] += weight * delta
                if done:
                    break
                weight *= gamma * lam
    return advantages


def audit_sample(data, row):
    require(
        all(a.dtype.kind in "fbiu" and np.isfinite(a).all() for a in data.values()),
        "nonfinite or nonnumeric array",
    )
    rewards, values, starts = (
        data[f"gae_{name}"] for name in ("rewards", "values", "episode_starts")
    )
    require(rewards.ndim == 2 and min(rewards.shape) > 0, "invalid rollout shape")
    require(values.shape == starts.shape == rewards.shape, "rollout shape mismatch")
    steps, envs = rewards.shape
    last_values, dones = data["gae_last_values"], data["gae_last_dones"]
    require(last_values.shape == dones.shape == (envs,), "boundary shape mismatch")
    require(np.isin(starts, (0, 1)).all() and np.isin(dones, (0, 1)).all(), "invalid end mask")
    gamma, lam = data["gae_gamma"], data["gae_lambda"]
    require(
        gamma.shape == lam.shape == () and 0 <= gamma <= 1 and 0 <= lam <= 1, "invalid GAE discount"
    )
    n = min(128, steps * envs)
    require(
        type(row["n_audit_samples"]) is int and row["n_audit_samples"] == n, "sample count mismatch"
    )
    raw = data["actions_raw"].astype(float)
    require(raw.ndim == 2 and raw.shape[0] == n and raw.shape[1] > 0, "invalid action shape")
    require(
        data["observations"].ndim == 2 and len(data["observations"]) == n, "invalid observations"
    )
    expanded = gae_from_td_sums(
        rewards.astype(float),
        values.astype(float),
        starts,
        last_values,
        dones,
        float(gamma),
        float(lam),
    ).T.reshape(-1)[:n]
    expected_values = values.T.reshape(-1)[:n]
    errors = {
        "gae": check(data["advantages"], expanded, "GAE"),
        "old_values": check(data["old_values"], expected_values, "value indexing", atol=0, rtol=0),
        "returns": check(data["returns"], expanded + expected_values, "returns"),
    }
    clip = row["clip_range"]
    require(type(clip) in (int, float) and np.isfinite(clip) and 0 < clip < 1, "invalid clip range")
    for label in ("before", "after"):
        mean, std = data[f"action_mean_{label}"], data[f"action_std_{label}"]
        require(
            mean.shape == std.shape == raw.shape and np.all(std > 0), "invalid Gaussian parameters"
        )
        gaussian_log = (
            -0.5 * ((raw - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
        ).sum(axis=1)
        errors[f"gaussian_log_{label}"] = check(
            data[f"log_prob_{label}"], gaussian_log, "Gaussian log probability"
        )
        logratio = data[f"log_prob_{label}"] - data["old_log_prob"].astype(float)
        ratio = np.exp(logratio)
        require(np.isfinite(ratio).all(), "nonfinite likelihood ratio")
        errors[f"ratio_{label}"] = check(
            data[f"ratio_{label}"], ratio, "likelihood ratio", atol=1e-10, rtol=1e-8
        )
        check(
            row[f"approx_reverse_kl_{label}"],
            np.mean(np.expm1(logratio) - logratio),
            "KL",
            atol=1e-10,
            rtol=1e-8,
        )
        check(
            row[f"ratio_clip_fraction_{label}"],
            np.mean(np.abs(ratio - 1) > clip),
            "clip fraction",
            atol=0,
            rtol=0,
        )
    check(data["ratio_before"], np.ones(n), "pre-update ratio")
    hashes = (row["param_sha256_before"], row["param_sha256_after"])
    require(
        all(
            isinstance(h, str) and len(h) == 64 and set(h) <= set("0123456789abcdef")
            for h in hashes
        ),
        "invalid parameter hash",
    )
    require(
        row["parameters_changed"] is (hashes[0] != hashes[1]), "parameter change label mismatch"
    )
    examples = []
    for positive in (True, False):
        indices = np.flatnonzero(data["advantages"] > 0 if positive else data["advantages"] < 0)
        if len(indices):
            i = int(indices[0])
            advantage, ratio = float(data["advantages"][i]), float(data["ratio_after"][i])
            unclipped, clipped = (
                advantage * ratio,
                advantage * float(np.clip(ratio, 1 - clip, 1 + clip)),
            )
            examples.append(
                {
                    "sample": i,
                    "advantage": advantage,
                    "ratio": ratio,
                    "unclipped": unclipped,
                    "clipped": clipped,
                    "surrogate_min": min(unclipped, clipped),
                }
            )
    return {
        "max_absolute_errors": errors,
        "raw_advantage_examples_not_optimizer_loss": examples,
        "parameters_changed": row["parameters_changed"],
        "samples": n,
    }


def audit_directory(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    require(rows, "no update records")
    checked = []
    previous = None
    for index, row in enumerate(rows):
        name = f"sample_{index:06d}.npz"
        require(row["audit_index"] == index and row["npz"] == name, "update sequence mismatch")
        path = directory / name
        require(not path.is_symlink(), "symlink sample")
        if previous is not None:
            require(row["param_sha256_before"] == previous, "parameter hash chain mismatch")
        with np.load(path, allow_pickle=False) as archive:
            result = audit_sample({name: archive[name] for name in archive.files}, row)
        checked.append(
            {
                "audit_index": index,
                "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **result,
            }
        )
        previous = row["param_sha256_after"]
    require(
        {p.name for p in directory.glob("sample_*.npz")} == {r["npz"] for r in rows},
        "unlisted sample file",
    )
    return {
        "directory": str(directory),
        "updates_checked": len(checked),
        "updates": checked,
        "scope": "GAE and recorded Gaussian likelihood arithmetic; no model deserialization or optimizer replay",
        "limitations": "first 128 env-major samples are not an unbiased rollout sample; rewards include timeout bootstrap; illustrative raw-advantage clipping is not minibatch-normalized training loss",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    report = {
        "schema": "d1-ppo-math-audit-v1",
        "runs": [audit_directory(path) for path in args.directories],
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"checked {sum(r['updates_checked'] for r in report['runs'])} recorded updates")


if __name__ == "__main__":
    main()
