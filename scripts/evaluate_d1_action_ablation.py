"""Evaluate every fixed-budget action candidate on the predeclared new holdout.

Run only after all nine models finish. The zero-residual controller is run once
per case, not copied into three allegedly independent baseline replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from wheel_legged_control.d1.ppo_action_policies import BoundedMeanActorCriticPolicy
from wheel_legged_control.d1.residual_action_env import D1LongitudinalResidualEnv
from wheel_legged_control.d1.terrain_tracking_env import D1TerrainTrackingEnv

if __package__:
    from . import run_d1_action_ablation as study
else:
    import run_d1_action_ablation as study

shared = study.shared
FINAL_BUDGET = 131072
FINAL_BUDGETS = (32768, 65536, 131072)
FINAL_EPOCHS = 1024
TRAINING_SEEDS = (9000, 10000, 11000)


def validate_ppo_settings(metadata, model):
    """Check the loaded algorithm, not only its self-reported metadata."""
    if metadata.get("ppo") != shared.PPO_SETTINGS:
        raise ValueError("PPO metadata differs from the frozen training settings")
    for field, expected in shared.PPO_SETTINGS.items():
        actual = getattr(model, field)
        if field == "clip_range":
            if any(actual(progress) != expected for progress in (0.0, 0.5, 1.0)):
                raise ValueError("loaded PPO clip_range differs from the frozen settings")
        elif actual != expected:
            raise ValueError(f"loaded PPO {field} differs from the frozen settings")
    if model.policy.net_arch != shared.PPO_SETTINGS["policy_kwargs"]["net_arch"]:
        raise ValueError("loaded PPO policy architecture differs from the frozen settings")
    learning_rate = shared.PPO_SETTINGS["learning_rate"]
    if any(model.lr_schedule(progress) != learning_rate for progress in (0.0, 0.5, 1.0)):
        raise ValueError("loaded PPO learning-rate schedule differs from the frozen settings")
    if any(group["lr"] != learning_rate for group in model.policy.optimizer.param_groups):
        raise ValueError("loaded PPO optimizer learning rate differs from the frozen settings")
    if model.n_envs != 4:
        raise ValueError("loaded PPO vectorization differs from the four-worker protocol")
    # These are the SB3 defaults used by the frozen trainer, not extra ablations.
    for field, expected in {
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "clip_range_vf": None,
        "target_kl": None,
    }.items():
        if getattr(model, field) != expected:
            raise ValueError(f"loaded PPO {field} differs from the frozen trainer defaults")


def validate_policy(metadata, model, model_path):
    """Reject shape-compatible but semantically different checkpoints."""
    variant = metadata.get("variant")
    if variant not in study.VARIANTS:
        raise ValueError("unknown action variant")
    environment = D1LongitudinalResidualEnv if variant == "fx_only" else D1TerrainTrackingEnv
    for field in ("observation_schema", "action_schema", "reward_schema", "control_schema"):
        if metadata.get(field) != getattr(environment, field):
            raise ValueError(f"incompatible {field}")
    dimension = 1 if variant == "fx_only" else 2
    if not isinstance(model.observation_space, spaces.Box) or not isinstance(
        model.action_space, spaces.Box
    ):
        raise TypeError("observations and actions must use Box spaces")
    if model.observation_space.shape != (44,) or model.action_space.shape != (dimension,):
        raise ValueError("incompatible observation/action shape")
    if not (np.all(model.action_space.low == -1) and np.all(model.action_space.high == 1)):
        raise ValueError("actions must use the normalized Box")
    scale = [11.25] if dimension == 1 else [11.25, 20.0]
    mapping = "[ax,0]" if dimension == 1 else "[ax,az]"
    if metadata.get("residual_force_scale_n") != scale:
        raise ValueError("incompatible force scale")
    if metadata.get("physical_action_mapping") != mapping:
        raise ValueError("incompatible physical mapping")
    expected_type = BoundedMeanActorCriticPolicy if variant == "bounded_mean" else ActorCriticPolicy
    if type(model.policy) is not expected_type:
        raise ValueError("policy class does not match the declared parameterization")
    if (
        model.use_sde
        or model.policy.use_sde
        or type(model.policy.action_dist) is not DiagGaussianDistribution
        or model.policy.squash_output
    ):
        raise ValueError("only the registered raw Gaussian distribution is eligible; no gSDE")
    distribution = (
        "Normal(tanh(logits), std), samples still clipped"
        if variant == "bounded_mean"
        else "Normal(logits, std), samples clipped"
    )
    if metadata.get("distribution") != distribution:
        raise ValueError("incompatible declared action distribution")
    seed = metadata.get("training_seed")
    if type(seed) is not int or seed not in TRAINING_SEEDS or model.seed != seed:
        raise ValueError("loaded model seed differs from the predeclared training identity")
    if metadata.get("actual_timesteps") != FINAL_BUDGET or model.num_timesteps != FINAL_BUDGET:
        raise ValueError("only the final fixed-budget checkpoint is eligible")
    if metadata.get("ppo_epochs_completed") != FINAL_EPOCHS or model._n_updates != FINAL_EPOCHS:
        raise ValueError("incomplete PPO updates")
    validate_ppo_settings(metadata, model)
    if metadata.get("training_reward_scale") != 0.01:
        raise ValueError("incompatible training reward scale")
    if metadata.get("model_sha256") != shared._sha256(model_path):
        raise ValueError("model SHA256 mismatch")
    if metadata.get("checkpoint_selection") != "final fixed budget; no holdout used":
        raise ValueError("checkpoint selection differs from the final fixed-budget protocol")
    if any(not torch.isfinite(value).all() for value in model.policy.state_dict().values()):
        raise ValueError("policy contains nonfinite parameters")


def expected_initializations(seed):
    """Recreate initialization hashes without running a training or holdout step."""
    with study.preserve_training_rng():
        common = PPO(
            "MlpPolicy",
            study.InitializationSpaceEnv(),
            **shared.PPO_SETTINGS,
            seed=seed,
            device="cpu",
        )
        state = common.policy.state_dict()
        digest = study.parameter_sha256(state)
        projected = {
            key: value[:1].clone()
            if key in {"action_net.weight", "action_net.bias", "log_std"}
            else value
            for key, value in state.items()
        }
        return {
            variant: {
                "common_two_action_parameter_sha256": digest,
                "initialized_parameter_sha256": study.parameter_sha256(projected)
                if variant == "fx_only"
                else digest,
                "projection": "copy all shared tensors; first action-head row and log_std only"
                if variant == "fx_only"
                else "identity",
            }
            for variant in study.VARIANTS
        }


def read_training_inputs(roots):
    """Validate completed runs, frozen protocols and artifact hashes before any case."""
    inputs, identities, source = [], set(), None
    model_hashes, initializations = set(), {}
    for root in roots:
        root = root.resolve()
        protocol = json.loads((root / "protocol.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if summary.get("source_unchanged_during_run") is not True:
            raise ValueError("training sources changed during a run")
        if protocol.get("budgets") != list(FINAL_BUDGETS) or protocol.get("envs") != 4:
            raise ValueError("training budget or vectorization differs from the protocol")
        if protocol["variants"] != list(study.VARIANTS):
            raise ValueError("each seed must include every predeclared variant")
        seeds = protocol.get("seeds", [])
        if (
            not seeds
            or any(type(seed) is not int or seed not in TRAINING_SEEDS for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("protocol seeds must be unique predeclared training seeds")
        expected_protocol = {
            "ppo": shared.PPO_SETTINGS,
            "training_reward_scale": 0.01,
            "training_mode": "mixed",
            "episode_seconds": 4.0,
            "development_cases": shared.evaluation_cases("development", "tracking-v2"),
            "intermediate_development_indices": list(study.INTERMEDIATE_CASES),
            "development_rng_isolation": "restore Python, numpy and torch CPU streams after checkpoint load and evaluation",
        }
        for field, expected in expected_protocol.items():
            if protocol.get(field) != expected:
                raise ValueError(f"training protocol {field} changed")
        if protocol["new_holdout_cases"] != study.frozen_holdout_cases():
            raise ValueError("holdout cases do not match the frozen training protocol")
        if protocol["quality_criteria"] != shared.TRACKING_QUALITY_CRITERIA:
            raise ValueError("quality thresholds changed")
        if source is None:
            source = protocol["source_sha256"]
        elif source != protocol["source_sha256"]:
            raise ValueError("training sources differ across seeds")
        if not isinstance(source, dict) or not source:
            raise ValueError("training source hashes are missing")
        files = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path != root / "manifest.json"
        }
        if not isinstance(manifest, dict) or set(manifest) != files:
            raise ValueError("training artifact manifest must cover every file except itself")
        for relative, digest in manifest.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or shared._sha256(path) != digest:
                raise ValueError(f"training artifact mismatch: {relative}")
        with tarfile.open(root / "runtime_source.tar.gz") as snapshot:
            members = snapshot.getmembers()
            if (
                len(members) != len(source)
                or {member.name for member in members} != set(source)
                or any(not member.isfile() for member in members)
            ):
                raise ValueError("training source snapshot must exactly cover the declared sources")
            for relative, digest in source.items():
                if hashlib.sha256(snapshot.extractfile(relative).read()).hexdigest() != digest:
                    raise ValueError(f"training source snapshot mismatch: {relative}")
        records = summary.get("models", [])
        expected_identities = {(variant, seed) for variant in study.VARIANTS for seed in seeds}
        if summary.get("training_runs") != len(expected_identities) or len(records) != len(
            expected_identities
        ):
            raise ValueError("summary training_runs does not cover the protocol candidates")
        root_identities = set()
        for record in records:
            variant, seed = record["variant"], record["training_seed"]
            identity = (variant, seed)
            if identity in identities:
                raise ValueError("duplicate candidate does not count as another seed")
            if identity not in expected_identities:
                raise ValueError(
                    "summary candidate identity differs from protocol seeds and variants"
                )
            identities.add(identity)
            root_identities.add(identity)
            directory = root / f"{variant}_seed_{seed}"
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata != record or metadata["source_sha256"] != source:
                raise ValueError("metadata differs from the frozen completed summary")
            digest = metadata.get("model_sha256")
            if digest != shared._sha256(directory / "model.zip"):
                raise ValueError("model artifact differs from its metadata hash")
            if digest in model_hashes:
                raise ValueError("copied model is not an independent training candidate")
            model_hashes.add(digest)
            if seed not in initializations:
                initializations[seed] = expected_initializations(seed)
            if metadata.get("initialization") != initializations[seed][variant]:
                raise ValueError("initialization differs from the paired common-seed protocol")
            inputs.append((directory, metadata))
        if root_identities != expected_identities:
            raise ValueError("summary is missing a predeclared training candidate")
    if identities != {(variant, seed) for variant in study.VARIANTS for seed in TRAINING_SEEDS}:
        raise ValueError("all nine predeclared models must finish before the holdout is opened")
    for relative, digest in source.items():
        path = (study.ROOT / relative).resolve()
        if not path.is_relative_to(study.ROOT) or shared._sha256(path) != digest:
            raise ValueError(f"current evaluation runtime differs from training: {relative}")
    return sorted(inputs, key=lambda item: (item[1]["training_seed"], item[1]["variant"])), source


def summarize_controllers(rows):
    """Case averages per training seed; no pseudo-replicated baseline uncertainty."""
    groups = {}
    for row in rows:
        groups.setdefault((row["variant"], row["training_seed"]), []).append(row)
    expected_controllers = {
        (variant, seed) for variant in study.VARIANTS for seed in TRAINING_SEEDS
    } | {("zero_residual", None)}
    if set(groups) != expected_controllers:
        raise ValueError("controller cohort must contain nine candidates and one unseeded baseline")
    baseline_cases = {row["case_id"] for row in groups[("zero_residual", None)]}
    if len(baseline_cases) != 24:
        raise ValueError("baseline must contain the 24 distinct holdout cases")
    for cases in groups.values():
        if len(cases) != 24 or {row["case_id"] for row in cases} != baseline_cases:
            raise ValueError("each controller must contain the same 24 cases exactly once")
    summary = []
    for (variant, seed), cases in groups.items():
        summary.append(
            {
                "variant": variant,
                "training_seed": seed,
                "cases": len(cases),
                "completed": sum(x["completed"] for x in cases),
                "quality_successes": sum(x["quality_success"] for x in cases),
                **{
                    f"mean_{key}": float(np.mean([x[key] for x in cases]))
                    for key in (
                        "velocity_rmse_mps",
                        "tail_velocity_rmse_mps",
                        "clearance_rmse_m",
                        "episode_return",
                    )
                },
            }
        )
    paired = []
    for seed in TRAINING_SEEDS:
        reference = next(
            x for x in summary if x["variant"] == "full_gaussian" and x["training_seed"] == seed
        )
        for variant in ("bounded_mean", "fx_only"):
            candidate = next(
                x for x in summary if x["variant"] == variant and x["training_seed"] == seed
            )
            paired.append(
                {
                    "variant": variant,
                    "training_seed": seed,
                    "comparison": "candidate minus same-seed full_gaussian",
                    **{
                        f"delta_{key}": candidate[key] - reference[key]
                        for key in reference
                        if key.startswith("mean_") or key == "quality_successes"
                    },
                }
            )
    return {"controllers": summary, "paired_seed_differences": paired}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must not exist")
    torch.set_num_threads(1)
    inputs, sources = read_training_inputs(args.training_roots)
    # Validate every model before starting any holdout simulation.
    parameters, candidates = set(), []
    for directory, metadata in inputs:
        model = PPO.load(directory / "model.zip", device="cpu")
        validate_policy(metadata, model, directory / "model.zip")
        identity = (metadata["variant"], study.parameter_sha256(model.policy.state_dict()))
        if identity in parameters:
            raise ValueError("copied policy parameters do not count as independent training seeds")
        parameters.add(identity)
        candidates.append((directory, metadata, model))
    sources = {
        **sources,
        str(Path(__file__).resolve().relative_to(study.ROOT)): shared._sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    study.snapshot_sources(args.output, sources)
    cases = study.frozen_holdout_cases()
    shared._write_json(
        args.output / "protocol.json",
        {
            "scope": "all fixed-budget candidates; no checkpoint selection from this holdout",
            "cases": cases,
            "quality_criteria": shared.TRACKING_QUALITY_CRITERIA,
            "source_sha256": sources,
            "dependency_versions": shared._dependency_versions(),
            "baseline_replication": "one episode per case; shared deterministic baseline, not three replicates",
            "inputs": [
                {
                    "directory": str(p),
                    "model_sha256": m["model_sha256"],
                    "metadata_sha256": shared._sha256(p / "metadata.json"),
                }
                for p, m in inputs
            ],
        },
    )
    metrics = [
        {"training_seed": None, **x}
        for x in study.evaluate(None, "zero_residual", cases, args.output / "zero_residual")
    ]
    for directory, metadata, model in candidates:
        variant, seed = metadata["variant"], metadata["training_seed"]
        scores = study.evaluate(model, variant, cases, args.output / f"{variant}_seed_{seed}")
        metrics.extend({"training_seed": seed, **x} for x in scores)
        print(
            f"{variant} seed={seed} holdout quality={sum(x['quality_success'] for x in scores)}/{len(scores)}",
            flush=True,
        )
    shared._write_csv(args.output / "metrics.csv", metrics, tuple(metrics[0]))
    unchanged = all(shared._sha256(study.ROOT / key) == digest for key, digest in sources.items())
    shared._write_json(
        args.output / "summary.json",
        {
            "episodes": len(metrics),
            "source_unchanged_during_evaluation": unchanged,
            **summarize_controllers(metrics),
        },
    )
    shared._write_json(
        args.output / "manifest.json",
        {
            str(p.relative_to(args.output)): shared._sha256(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )
    if not unchanged:
        raise SystemExit("evaluation runtime changed during holdout")


if __name__ == "__main__":
    main()
