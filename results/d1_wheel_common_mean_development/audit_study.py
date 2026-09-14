"""Read-only audit of completed wheel-common-mean studies; never step physics.

Example: python -B audit_study.py --repo REPO --input STUDY --output NEW.json
Repeat --input to audit several study roots. Output must be new and outside them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def parameter_sha(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Audit:
    def __init__(self, repo):
        self.repo = repo
        self.checks = 0
        self.failures = []
        self.models = {}

    def check(self, label, condition, detail=None):
        self.checks += 1
        if not bool(condition):
            self.failures.append({"check": label, "detail": detail})

    def close(self, label, actual, expected, atol=1e-9, rtol=1e-8):
        a, b = np.asarray(actual), np.asarray(expected)
        okay = (
            a.shape == b.shape
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol)
        )
        error = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else None
        self.check(label, okay, {"maximum_absolute_error": error})

    def load_model(self, path):
        from stable_baselines3 import PPO

        key = str(Path(path).resolve())
        if key not in self.models:
            self.models[key] = PPO.load(key, device="cpu")
        return self.models[key]

    def manifests(self, root):
        manifests = sorted(root.rglob("manifest.json"))
        self.check(f"{root.name}: root manifest exists", root / "manifest.json" in manifests)
        entries = 0
        for path in manifests:
            manifest = read(path)
            actual = {
                str(p.relative_to(path.parent))
                for p in path.parent.rglob("*")
                if p.is_file() and p != path
            }
            self.check(
                f"{path}: complete file inventory",
                actual == set(manifest),
                {
                    "unlisted": sorted(actual - set(manifest)),
                    "missing": sorted(set(manifest) - actual),
                },
            )
            for rel, expected in manifest.items():
                item = path.parent / rel
                self.check(
                    f"{item}: manifest size and SHA256",
                    item.is_file()
                    and item.stat().st_size == expected["bytes"]
                    and sha(item) == expected["sha256"],
                )
                entries += 1
        return {"manifest_count": len(manifests), "manifest_entries_checked": entries}

    def train(self, root, protocol, global_summary):
        from dataclasses import asdict

        import torch

        from scripts.d1_course_curriculum import scaled_training_terrain

        results, pairs = [], {}
        for seed in protocol["seeds"]:
            pair_hashes = []
            for variant, limit in protocol["variants"].items():
                folder = root / f"seed{seed}" / variant
                label = str(folder)
                summary, initial = (
                    read(folder / "summary.json"),
                    read(folder / "initial_policy.json"),
                )
                metadata, episodes = (
                    read(folder / "model.metadata.json"),
                    read(folder / "episodes.json"),
                )
                model = self.load_model(folder / "model.zip")
                steps, hp = (
                    protocol["training_steps_per_checkpoint"],
                    summary["ppo_actual_hyperparameters"],
                )
                identity = {
                    "seed": seed,
                    "variant": variant,
                    "wheel_common_mean_limit": limit,
                    "actual_transitions": steps,
                    "pipeline_smoke_only": protocol["pipeline_smoke_only"],
                }
                for key, expected in identity.items():
                    self.check(
                        f"{label}: identity {key}",
                        summary[key] == expected and metadata["extra"][key] == expected,
                    )
                self.check(
                    f"{label}: model metadata SHA",
                    sha(folder / "model.zip") == metadata["model_sha256"],
                )
                self.check(
                    f"{label}: saved policy source SHA",
                    metadata["extra"]["policy_module_sha256"]
                    == protocol["source_sha256"]["scripts/d1_wheel_common_mean_policy.py"],
                )
                self.check(
                    f"{label}: loaded policy identity",
                    type(model.policy).__module__ + "." + type(model.policy).__name__
                    == protocol["policy_class"]
                    and model.policy.wheel_common_mean_limit == limit
                    and model.seed == seed,
                )
                self.check(
                    f"{label}: actual checkpoint timesteps",
                    model.num_timesteps == steps == summary["ppo_num_timesteps"],
                )
                expected_epochs = steps // (model.n_steps * model.n_envs) * model.n_epochs
                self.check(
                    f"{label}: optimization epochs",
                    model._n_updates == expected_epochs == summary["ppo_optimization_epochs"],
                )
                self.check(
                    f"{label}: rollout count",
                    summary["rollout_train_calls"] == steps // model.n_steps,
                )
                for key in (
                    "n_steps",
                    "batch_size",
                    "n_epochs",
                    "gamma",
                    "gae_lambda",
                    "learning_rate",
                    "ent_coef",
                    "vf_coef",
                    "max_grad_norm",
                    "normalize_advantage",
                    "use_sde",
                    "sde_sample_freq",
                    "target_kl",
                ):
                    self.check(
                        f"{label}: saved hyperparameter {key}", getattr(model, key) == hp[key]
                    )
                for key, expected in protocol["ppo_settings"].items():
                    if key != "policy_kwargs":
                        self.check(f"{label}: protocol hyperparameter {key}", hp[key] == expected)
                self.check(
                    f"{label}: policy kwargs",
                    model.policy_kwargs
                    == hp["policy_kwargs"]
                    == {
                        **protocol["ppo_settings"]["policy_kwargs"],
                        "wheel_common_mean_limit": limit,
                    },
                )
                self.check(
                    f"{label}: final parameter SHA",
                    parameter_sha(model.policy) == summary["final_policy_parameter_sha256"],
                )
                self.check(
                    f"{label}: final parameters finite",
                    all(torch.isfinite(p).all() for p in model.policy.parameters()),
                )
                optimizer_steps = sorted(
                    {float(v["step"]) for v in model.policy.optimizer.state.values()}
                )
                expected_optimizer_steps = expected_epochs * int(
                    np.ceil(model.n_steps * model.n_envs / model.batch_size)
                )
                self.check(
                    f"{label}: optimizer step counters",
                    optimizer_steps == [float(expected_optimizer_steps)],
                )

                # Independent reconstruction of initial weights requires no environment.
                torch.manual_seed(seed)
                reconstructed = type(model.policy)(
                    model.observation_space,
                    model.action_space,
                    model.lr_schedule,
                    **model.policy_kwargs,
                )
                initial_hash = parameter_sha(reconstructed)
                self.check(
                    f"{label}: regenerated initialization SHA",
                    initial_hash
                    == initial["initialization_state_dict_sha256"]
                    == summary["initialization_state_dict_sha256"]
                    == metadata["extra"]["initialization_state_dict_sha256"],
                )
                self.check(
                    f"{label}: initialization at zero transitions", initial["num_timesteps"] == 0
                )
                self.check(
                    f"{label}: initial hyperparameters", initial["ppo_actual_hyperparameters"] == hp
                )
                pair_hashes.append(initial_hash)
                with (folder / "progress.csv").open() as stream:
                    progress = list(csv.DictReader(stream))
                self.check(
                    f"{label}: CSV final transition count",
                    max(
                        float(r["time/total_timesteps"])
                        for r in progress
                        if r.get("time/total_timesteps")
                    )
                    == steps,
                )
                self.check(
                    f"{label}: CSV final optimization epoch",
                    max(float(r["train/n_updates"]) for r in progress if r.get("train/n_updates"))
                    == model._n_updates,
                )
                by_level, counts, nonflat, reasons = Counter(), Counter(), Counter(), Counter()
                _, terrain_seq, episode_seq = np.random.SeedSequence(seed).spawn(3)
                terrain_rng, episode_rng = (
                    np.random.default_rng(terrain_seq),
                    np.random.default_rng(episode_seq),
                )
                end = 0
                for index, episode in enumerate(episodes):
                    count, level = episode["transitions"], episode["level"]
                    self.check(
                        f"{label}: episode {index} transition continuity",
                        episode["episode_index"] == index
                        and episode["start_total_transitions"] == end
                        and episode["end_total_transitions"] == end + count
                        and count > 0,
                    )
                    self.check(
                        f"{label}: episode {index} curriculum level",
                        level == min(3, (4 * end) // steps),
                    )
                    terrain_index = int(terrain_rng.integers(0, 4))
                    episode_seed = int(episode_rng.integers(0, 2**31 - 1))
                    self.check(
                        f"{label}: episode {index} reset RNG",
                        episode["terrain_index"] == terrain_index
                        and episode["episode_seed"] == episode_seed,
                    )
                    self.check(
                        f"{label}: episode {index} actual terrain",
                        episode["terrain_parameters"]
                        == asdict(scaled_training_terrain(level, terrain_index)),
                    )
                    n = (episode["last_terrain_exposure"] or {}).get("nonflat_steps", 0)
                    self.check(f"{label}: episode {index} exposure bounds", 0 <= n <= count)
                    end += count
                    by_level[str(level)] += count
                    counts[str(level)] += 1
                    nonflat[str(level)] += n
                    reasons[str(episode["terminal_reason"])] += 1
                self.check(f"{label}: physical transition conservation", end == steps)
                for field, counter in (
                    ("transitions_by_level", by_level),
                    ("episodes_by_level", counts),
                    ("geometric_nonflat_steps_by_level", nonflat),
                ):
                    self.check(f"{label}: {field}", dict(counter) == summary[field])
                self.check(f"{label}: aggregate summary row", summary in global_summary["results"])
                results.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "model_sha256": sha(folder / "model.zip"),
                        "actual_transitions": end,
                        "ppo_num_timesteps": model.num_timesteps,
                        "optimization_epochs": model._n_updates,
                        "optimizer_steps": optimizer_steps,
                        "initial_parameter_sha256": initial_hash,
                        "final_parameter_sha256": parameter_sha(model.policy),
                        "episodes": len(episodes),
                        "transitions_by_level": dict(by_level),
                        "geometric_nonflat_steps_by_level": dict(nonflat),
                        "termination_counts": dict(reasons),
                    }
                )
            pairs[str(seed)] = len(set(pair_hashes)) == 1
            self.check(f"seed {seed}: paired initial weights equal", pairs[str(seed)])
        self.check(
            f"{root.name}: total training budget",
            sum(r["actual_transitions"] for r in results) == protocol["training_total_steps"],
        )
        return {
            "training": results,
            "paired_initialization": pairs,
            "exposure_scope": "Reaggregated episode records; no per-transition training states were stored, so terrain-contact exposure cannot be independently reconstructed.",
        }

    def evaluation(self, root, protocol, global_summary):
        import torch

        cases = {c["name"]: c for c in protocol["evaluation_protocol"]["cases"]}
        checkpoints = {}
        for path, expected in protocol["checkpoint_sha256"].items():
            self.check(
                f"{path}: evaluation source checkpoint SHA",
                Path(path).is_file() and sha(path) == expected,
            )
            if path.endswith("model.zip"):
                file = Path(path)
                metadata = read(file.with_name("model.metadata.json"))
                extra = metadata["extra"]
                self.check(f"{path}: metadata model linkage", metadata["model_sha256"] == expected)
                self.check(
                    f"{path}: formal checkpoint identity",
                    extra["actual_transitions"] == 16384 and extra["pipeline_smoke_only"] is False,
                )
                checkpoints[f"seed{extra['seed']}/{extra['variant']}"] = file
        expected_labels = {"zero"} | {
            f"seed{seed}/{variant}"
            for seed in protocol["seeds"]
            for variant in protocol["variants"]
        }
        self.check(
            f"{root.name}: complete checkpoint set", set(checkpoints) == expected_labels - {"zero"}
        )
        results = []
        for name in protocol["selected_case_names"]:
            case, initial_reference = cases[name], None
            self.check(f"{name}: case selected split", case["split"] == protocol["selected_split"])
            for label in sorted(expected_labels):
                folder = root / name / label
                summary, initial = read(folder / "summary.json"), read(folder / "initial.json")
                with (folder / "trace.jsonl").open() as stream:
                    rows = [json.loads(line) for line in stream]
                with np.load(folder / "states.npz", allow_pickle=False) as archive:
                    states = {key: archive[key] for key in archive.files}
                n = len(rows)
                tag = f"{name}/{label}"
                self.check(
                    f"{tag}: no evaluation script failure",
                    "error" not in summary and not (folder / "failure.json").exists(),
                )
                self.check(
                    f"{tag}: nonempty complete state arrays",
                    n > 0
                    and set(states) == {"observation", "qpos", "qvel", "time_s"}
                    and all(len(v) == n + 1 for v in states.values()),
                )
                for key, array in states.items():
                    self.check(f"{tag}: {key} finite", np.isfinite(array).all())
                for key in ("observation", "qpos", "qvel"):
                    self.check(
                        f"{tag}: initial {key} SHA",
                        array_sha(states[key][0]) == initial[f"{key}_sha256"],
                    )
                if initial_reference is None:
                    initial_reference = initial
                self.check(
                    f"{tag}: common initial state and metadata", initial == initial_reference
                )
                meta = initial["episode_metadata"]
                self.check(
                    f"{tag}: case terrain and duration",
                    meta["terrain"] == case["terrain"]
                    and meta["duration_s"] == case["episode_seconds"],
                )
                self.check(
                    f"{tag}: oracle residual task",
                    meta["provider"]["kind"] == "oracle"
                    and meta["action_mode"] == "independent8"
                    and meta["baseline"] == "wheel_leg"
                    and meta["command_source"] == "external_callback",
                )
                times = states["time_s"]
                self.close(
                    f"{tag}: control time increments", np.diff(times), np.full(n, 0.01), atol=1e-10
                )
                self.close(f"{tag}: initial simulation time", times[0], 0.0)
                self.close(
                    f"{tag}: transition sequence",
                    [r["transition"] for r in rows],
                    np.arange(1, n + 1),
                    atol=0,
                    rtol=0,
                )
                metrics = {
                    key: np.array([r["metrics"][key] for r in rows]) for key in rows[0]["metrics"]
                }
                self.close(f"{tag}: metric times", metrics["time_s"], times[1:])
                self.close(
                    f"{tag}: metric positions",
                    np.column_stack([metrics[k] for k in ("x_m", "y_m", "z_m")]),
                    states["qpos"][1:, :3],
                )
                spec = case["command"]
                ramp = np.clip(
                    (times[:-1] - spec["settle_seconds"]) / spec["ramp_seconds"], 0.0, 1.0
                )
                command = np.column_stack(
                    [
                        ramp * spec["target_forward_mps"],
                        ramp * spec["target_yaw_rps"],
                        np.full(n, spec["height_m"]),
                    ]
                )
                self.close(
                    f"{tag}: every executed command",
                    [
                        [
                            r["command"][k]
                            for k in ("forward_velocity_mps", "yaw_rate_rps", "clearance_m")
                        ]
                        for r in rows
                    ],
                    command,
                )
                encoded_command = np.column_stack(
                    [command[:, 0] / 0.6, command[:, 1] / 0.5, (command[:, 2] - 0.455) / 0.08]
                )
                self.close(
                    f"{tag}: policy observed current command",
                    states["observation"][:-1, 38:41],
                    np.clip(encoded_command, -5, 5).astype(np.float32),
                    atol=2e-7,
                )
                self.close(
                    f"{tag}: initial previous action", states["observation"][0, 74:82], np.zeros(8)
                )
                w, x, y, z = states["qpos"][:, 3:7].T
                rolls = np.arctan2(2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
                pitches = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
                yaws = np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
                self.close(f"{tag}: yaw from recorded quaternion", metrics["yaw_rad"], yaws[1:])
                gravity = -np.column_stack(
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
                )
                self.close(
                    f"{tag}: observed prestep projected gravity",
                    states["observation"][:-1, 6:9],
                    gravity[:-1].astype(np.float32),
                    atol=2e-7,
                )
                endpoint_valid = np.array(
                    [
                        r.get("terminal_observation_source") != "last_executable_decision"
                        for r in rows
                    ]
                )
                post_observations = states["observation"][1:][endpoint_valid]
                self.close(
                    f"{tag}: endpoint observed body velocity",
                    post_observations[:, 0],
                    np.clip(metrics["velocity_mps"][endpoint_valid], -5, 5).astype(np.float32),
                    atol=2e-7,
                )
                self.close(
                    f"{tag}: endpoint observed yaw rate",
                    post_observations[:, 5],
                    np.clip(metrics["yaw_rate_rps"][endpoint_valid] / 4, -5, 5).astype(np.float32),
                    atol=2e-7,
                )
                self.close(
                    f"{tag}: yaw rate from saved generalized velocity",
                    metrics["yaw_rate_rps"],
                    states["qvel"][1:, 5],
                )
                self.close(
                    f"{tag}: endpoint angular velocity observation",
                    post_observations[:, 3:6],
                    np.clip(states["qvel"][1:, 3:6][endpoint_valid] / 4, -5, 5).astype(np.float32),
                    atol=2e-7,
                )
                self.close(
                    f"{tag}: endpoint observed height error",
                    post_observations[:, 9],
                    np.clip(metrics["height_error_m"][endpoint_valid] / 0.08, -5, 5).astype(
                        np.float32
                    ),
                    atol=2e-7,
                )
                self.close(
                    f"{tag}: roll error from quaternion and observed terrain target",
                    metrics["roll_error_rad"][endpoint_valid],
                    rolls[1:][endpoint_valid] - post_observations[:, 41] * 0.3,
                    atol=2e-7,
                )
                self.close(
                    f"{tag}: pitch error from quaternion and observed terrain target",
                    metrics["pitch_error_rad"][endpoint_valid],
                    pitches[1:][endpoint_valid] - post_observations[:, 42] * 0.3,
                    atol=2e-7,
                )
                self.close(
                    f"{tag}: velocity error",
                    metrics["velocity_error_mps"],
                    metrics["velocity_mps"] - command[:, 0],
                )
                self.close(
                    f"{tag}: yaw rate error",
                    metrics["yaw_rate_error_rps"],
                    metrics["yaw_rate_rps"] - command[:, 1],
                )
                self.close(
                    f"{tag}: clearance",
                    metrics["clearance_m"],
                    states["qpos"][1:, 2] - metrics["ground_height_m"],
                )
                self.close(
                    f"{tag}: height error",
                    metrics["height_error_m"],
                    metrics["clearance_m"] - command[:, 2],
                )
                arrays = {
                    key: np.array([r[key] for r in rows])
                    for key in (
                        "raw_net_mean",
                        "transformed_gaussian_mean",
                        "gaussian_std",
                        "box_clipped_action",
                        "policy_action",
                        "raw_action",
                        "applied_action",
                    )
                }
                raw, mean = arrays["raw_net_mean"], arrays["transformed_gaussian_mean"]
                expected_mean = raw.copy()
                limit = None if label == "zero" else protocol["variants"][label.split("/")[1]]
                if limit is not None:
                    common = raw[:, 4:].mean(1, keepdims=True)
                    expected_mean[:, 4:] = raw[:, 4:] - common + limit * np.tanh(common / limit)
                self.close(f"{tag}: independent mean transform", mean, expected_mean, atol=3e-7)
                self.close(
                    f"{tag}: native Box clip",
                    arrays["box_clipped_action"],
                    np.clip(mean, -1, 1),
                    atol=2e-6,
                )
                for key in ("policy_action", "raw_action", "applied_action"):
                    self.close(
                        f"{tag}: physical action {key}",
                        arrays[key],
                        arrays["box_clipped_action"],
                        atol=2e-6,
                    )
                self.close(
                    f"{tag}: endpoint previous action",
                    post_observations[:, 74:82],
                    arrays["applied_action"][endpoint_valid].astype(np.float32),
                    atol=2e-7,
                )
                clipping = np.mean((mean < -1) | (mean > 1), axis=1)
                self.close(
                    f"{tag}: clipping fraction",
                    [r["deterministic_box_clipped_fraction"] for r in rows],
                    clipping,
                )
                common_stats = {}
                for field, array_key in (
                    ("raw_net_wheel_common_mean", "raw_net_mean"),
                    ("transformed_gaussian_wheel_common_mean", "transformed_gaussian_mean"),
                    ("box_clipped_wheel_common_mean", "box_clipped_action"),
                    ("executed_wheel_common_mean", "applied_action"),
                ):
                    values = arrays[array_key][:, 4:].mean(1)
                    self.close(f"{tag}: {field}", [r[field] for r in rows], values, atol=2e-7)
                    common_stats[field] = {
                        "mean": float(values.mean()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "max_abs": float(np.max(np.abs(values))),
                    }
                    for stat, value in common_stats[field].items():
                        self.close(
                            f"{tag}: {field} summary {stat}",
                            summary["wheel_common_mean_statistics"][field][stat],
                            value,
                            atol=2e-7,
                        )
                if label == "zero":
                    self.close(
                        f"{tag}: zero policy arrays",
                        np.stack(list(arrays.values())),
                        np.zeros((7, n, 8)),
                    )
                    checkpoint_sha = None
                else:
                    model = self.load_model(checkpoints[label])
                    policy = model.policy
                    self.check(
                        f"{tag}: loaded formal policy",
                        model.num_timesteps == 16384 and policy.wheel_common_mean_limit == limit,
                    )
                    with torch.no_grad():
                        tensor = torch.as_tensor(states["observation"][:-1])
                        features = policy.extract_features(tensor)
                        latent = policy.mlp_extractor.forward_actor(
                            features if policy.share_features_extractor else features[0]
                        )
                        predicted_raw = policy.action_net(latent).cpu().numpy()
                        dist = policy.get_distribution(tensor).distribution
                        predicted_mean = dist.mean.cpu().numpy()
                        predicted_std = dist.stddev.cpu().numpy()
                    self.close(
                        f"{tag}: saved observations checkpoint raw head",
                        raw,
                        predicted_raw,
                        atol=2e-6,
                    )
                    self.close(
                        f"{tag}: saved observations checkpoint mean",
                        mean,
                        predicted_mean,
                        atol=2e-6,
                    )
                    self.close(
                        f"{tag}: saved checkpoint std",
                        arrays["gaussian_std"],
                        predicted_std,
                        atol=2e-6,
                    )
                    checkpoint_sha = sha(checkpoints[label])
                result = self.eval_metrics(tag, rows, states, metrics, arrays, summary, meta, case)
                result.update(
                    case=name,
                    policy=label,
                    source_checkpoint_sha256=checkpoint_sha,
                    wheel_common_mean_statistics=common_stats,
                )
                self.check(
                    f"{tag}: aggregate summary row",
                    {"case": name, **summary} in global_summary["results"],
                )
                results.append(result)
        self.check(
            f"{root.name}: expected evaluation count",
            len(results)
            == len(protocol["selected_case_names"]) * len(expected_labels)
            == len(global_summary["results"]),
        )
        return {
            "evaluations": results,
            "evaluation_limits": "No physical episodes were rerun. Ground contact and actuator-substep truth are not stored; their saved aggregate metrics can be checked for consistency but not independently regenerated.",
        }

    def eval_metrics(self, tag, rows, states, metrics, arrays, summary, meta, case):
        n = len(rows)
        result = {
            "actual_transitions": n,
            "integrated_duration_s": float(states["time_s"][-1]),
            "initial_x_m": float(states["qpos"][0, 0]),
            "final_x_m": float(states["qpos"][-1, 0]),
            "signed_forward_progress_m": float(states["qpos"][-1, 0] - states["qpos"][0, 0]),
        }
        for prefix, field, unit in (
            ("velocity", "velocity_error_mps", "mps"),
            ("yaw_rate", "yaw_rate_error_rps", "rps"),
            ("height", "height_error_m", "m"),
            ("roll", "roll_error_rad", "rad"),
            ("pitch", "pitch_error_rad", "rad"),
        ):
            result[f"{prefix}_rmse_{unit}"] = float(np.sqrt(np.mean(metrics[field] ** 2)))
        terms = {
            key: np.array([r["reward_terms"][key] for r in rows]) for key in rows[0]["reward_terms"]
        }
        reward = np.array([r["reward"] for r in rows])
        self.close(
            f"{tag}: reward equals contribution sum", reward, np.sum(list(terms.values()), axis=0)
        )
        dt = np.diff(states["time_s"])
        c = meta["reward"]
        delta = (
            arrays["applied_action"] - np.vstack([np.zeros((1, 8)), arrays["applied_action"][:-1]])
        ) * (0.01 / dt[:, None])
        recomputed = {
            "tracking_velocity": dt
            * c["velocity_weight"]
            * np.exp(-((metrics["velocity_error_mps"] / c["velocity_sigma_mps"]) ** 2)),
            "tracking_yaw": dt
            * c["yaw_weight"]
            * np.exp(-((metrics["yaw_rate_error_rps"] / c["yaw_sigma_rps"]) ** 2)),
            "height": -dt
            * c["height_weight"]
            * (metrics["height_error_m"] / c["height_scale_m"]) ** 2,
            "attitude": -dt
            * c["attitude_weight"]
            * (metrics["roll_error_rad"] ** 2 + metrics["pitch_error_rad"] ** 2)
            / c["attitude_scale_rad"] ** 2,
            "mechanical_power": -dt
            * c["power_weight"]
            * metrics["mechanical_power_w"]
            / c["mechanical_power_scale_w"],
            "action_change": -dt
            * c["action_change_weight"]
            * np.mean((delta / c["action_change_scale"]) ** 2, axis=1),
            "termination": -c["termination_cost"]
            * np.array([r["terminated"] for r in rows], dtype=float),
        }
        for key, values in recomputed.items():
            self.close(f"{tag}: independent reward {key}", terms[key], values)
        result.update(
            cumulative_return=float(reward.sum()),
            mean_torque_input_clipped_fraction=float(
                metrics["torque_input_clipped_fraction"].mean()
            ),
            mean_deterministic_box_clipped_fraction=float(
                np.mean(
                    (arrays["transformed_gaussian_mean"] < -1)
                    | (arrays["transformed_gaussian_mean"] > 1)
                )
            ),
            minimum_clearance_m=float(metrics["clearance_m"].min()),
            maximum_undesired_ground_contacts=int(metrics["undesired_ground_contacts"].max()),
        )
        for key, value in result.items():
            self.close(f"{tag}: summary {key}", summary[key], value)
        for key, values in terms.items():
            self.close(
                f"{tag}: cumulative {key}",
                summary["cumulative_reward_terms"][key],
                float(values.sum()),
            )
        endings = np.array([r["terminated"] or r["truncated"] for r in rows])
        self.check(f"{tag}: first terminal ends recording", not endings[:-1].any() and endings[-1])
        last = rows[-1]
        for key in ("terminated", "truncated", "terminal_reason"):
            self.check(f"{tag}: summary {key}", summary[key] == last[key])
            result[key] = last[key]
        max_steps = round(case["episode_seconds"] / 0.01)
        self.check(f"{tag}: duration within case budget", n <= max_steps)
        self.check(
            f"{tag}: truncation budget",
            not last["truncated"]
            or (
                not last["terminated"]
                and last["terminal_reason"] == "time_limit"
                and n == max_steps
            ),
        )
        if last["terminal_reason"] == "map_boundary":
            self.check(
                f"{tag}: boundary termination supported by position",
                np.any(np.abs(states["qpos"][-1, :2]) >= np.array(meta["safe_half_size_m"])),
            )
        if last["terminal_reason"] == "fall_or_body_contact":
            w, x, y, z = states["qpos"][-1, 3:7]
            roll = np.arctan2(2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
            pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
            self.check(
                f"{tag}: fall/contact termination supported by recorded state",
                metrics["clearance_m"][-1] < 0.22
                or max(abs(roll), abs(pitch)) > 0.85
                or metrics["undesired_ground_contacts"][-1] > 0,
            )
        exposure = [r["terrain_exposure"] for r in rows]
        flags = np.array([e["nonflat_now"] for e in exposure], dtype=bool)
        distances = np.linalg.norm(np.diff(states["qpos"][:, :2], axis=0), axis=1)
        self.close(
            f"{tag}: geometric nonflat counts",
            [e["nonflat_steps"] for e in exposure],
            np.cumsum(flags),
        )
        self.close(
            f"{tag}: geometric nonflat fractions",
            [e["nonflat_fraction"] for e in exposure],
            np.cumsum(flags) / np.arange(1, n + 1),
        )
        self.close(f"{tag}: path lengths", [e["path_m"] for e in exposure], np.cumsum(distances))
        self.close(
            f"{tag}: nonflat path lengths",
            [e["nonflat_path_m"] for e in exposure],
            np.cumsum(distances * flags),
        )
        cells, cumulative_cells = set(), []
        for point, nonflat in zip(states["qpos"][1:, :2], flags):
            if nonflat:
                cells.add(tuple(np.floor(point / 0.25).astype(int)))
            cumulative_cells.append(len(cells))
        self.close(
            f"{tag}: unique geometric nonflat cells",
            [e["unique_nonflat_0p25m_cells"] for e in exposure],
            cumulative_cells,
        )
        self.check(
            f"{tag}: final exposure summary", summary["geometric_nonflat_exposure"] == exposure[-1]
        )
        result["geometric_nonflat_steps"] = int(flags.sum())
        return result

    def study(self, root):
        before_checks, before_failures = self.checks, len(self.failures)
        protocol, summary = read(root / "protocol.json"), read(root / "summary.json")
        report = {"input": str(root), "mode": protocol["mode"], **self.manifests(root)}
        for rel, expected in protocol["source_sha256"].items():
            self.check(
                f"{root.name}: source {rel}",
                (self.repo / rel).is_file() and sha(self.repo / rel) == expected,
            )
        self.check(
            f"{root.name}: completed study",
            summary["all_runs_completed"] is True and not (root / "failure.json").exists(),
        )
        report["source_hashes_checked"] = len(protocol["source_sha256"])
        if protocol["mode"] in ("smoke", "train"):
            report.update(self.train(root, protocol, summary))
        elif protocol["mode"] == "evaluate":
            report.update(self.evaluation(root, protocol, summary))
        else:
            self.check(f"{root.name}: recognized mode", False)
        report.update(checks=self.checks - before_checks, failures=self.failures[before_failures:])
        report["passed"] = not report["failures"]
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, action="append")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    roots = [p.resolve() for p in args.input]
    if output.exists() or any(output.is_relative_to(p) for p in roots):
        parser.error("--output must be new and outside every input study")
    sys.path[:0] = [str(repo / ".local-deps"), str(repo / "src"), str(repo)]
    import torch

    torch.set_num_threads(1)
    audit = Audit(repo)
    studies = []
    for root in roots:
        try:
            studies.append(audit.study(root))
        except Exception as error:  # noqa: BLE001 -- Preserve audit failures in the report.
            audit.check(
                f"{root}: audit could not complete", False, f"{type(error).__name__}: {error}"
            )
            studies.append(
                {"input": str(root), "passed": False, "error": f"{type(error).__name__}: {error}"}
            )
    result = {
        "schema": "d1-wheel-common-mean-readonly-audit-v1",
        "audit_script_sha256": sha(__file__),
        "repo": str(repo),
        "physical_steps_executed": 0,
        "passed": not audit.failures,
        "checks": audit.checks,
        "failures": audit.failures,
        "studies": studies,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": result["passed"],
                "checks": audit.checks,
                "failures": len(audit.failures),
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
