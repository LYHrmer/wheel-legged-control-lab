"""Predeclared joint stress: 30 ms sensing, 2x noise SD, 0.6x friction.

This is not a terrain holdout or an attribution experiment. It reuses the
existing development evaluator and its termination, quality gates and logs.
Load only trusted local SB3 checkpoints; their deserialization executes Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_d1_locomotion_experiment as experiment
from wheel_legged_control.d1.actuator_channel import ActuatorChannelConfig
from wheel_legged_control.d1.locomotion_checkpoint import load_locomotion_policy
from wheel_legged_control.d1.locomotion_env import D1LocomotionRandomization
from wheel_legged_control.d1.model import JOINT_TORQUE_LIMIT

SCHEMA = "d1-joint-unseen-stress-v1"
TRAINING_SEEDS = (27000, 28000, 29000)
EPISODE_SEEDS = (17, 29)
DURATION_S = 60.0
FINAL_STEPS = 32768
GAINS = {
    "wheel_kp": 0.55,
    "wheel_ki": 1.5,
    "yaw_feedback_gain": 4.0,
    "leg_feedback_scale": 1.0,
    "attitude_feedback_scale": 0.25,
}
NOISE_STD_FIELDS = (
    "gyro_std_rad_s",
    "accelerometer_std_m_s2",
    "encoder_position_std_rad",
    "encoder_velocity_std_rad_s",
)
NOISE_FACTOR = 2.0
DELAY_STEPS = 3
FRICTION_SCALE = 0.6
TRAINING_RANDOMIZATION = D1LocomotionRandomization(
    measurement_delay_steps=(0, 2),
    actuator_delay_steps=(0, 3),
    actuator_time_constant_s=(0.0, 0.006),
    actuator_gain=(0.95, 1.05),
)
INPUT_NAMES = (
    "checkpoint.zip",
    "checkpoint.json",
    "training.json",
    "protocol.json",
    "source.json",
    "source_consistency.json",
    "source.tar.gz",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    """Normalize tuples to the exact JSON representation in existing sidecars."""
    return json.loads(json.dumps(value, allow_nan=False))


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def stress_noise():
    return replace(
        experiment.NOISE,
        **{name: getattr(experiment.NOISE, name) * NOISE_FACTOR for name in NOISE_STD_FIELDS},
    )


def stress_randomization():
    return D1LocomotionRandomization(friction_scale=(FRICTION_SCALE, FRICTION_SCALE))


def ideal_actuator():
    # Ideal response still retains the production joint torque limits, in Nm.
    return ActuatorChannelConfig(torque_limit_nm=tuple(JOINT_TORQUE_LIMIT))


def stress_kwargs(kwargs):
    """The only runtime adapter: public constructor parameters, no private state."""
    provider = kwargs["provider_config"]
    if (
        provider.kind != "imu_encoder_fusion"
        or provider.sensor_delay_steps != DELAY_STEPS
        or provider.sensor_noise != experiment.NOISE
    ):
        raise ValueError("stress adapter requires the declared fusion/base noise/30 ms source")
    if "randomization" in kwargs or "actuator_config" in kwargs:
        raise ValueError("stress adapter cannot silently replace another dynamics override")
    return {
        **kwargs,
        "provider_config": replace(provider, sensor_noise=stress_noise()),
        "randomization": stress_randomization(),
        "actuator_config": ideal_actuator(),
    }


@contextmanager
def stress_adapter():
    original = experiment.D1LocomotionEnv

    def factory(**kwargs):
        return original(**stress_kwargs(kwargs))

    with patch.object(experiment, "D1LocomotionEnv", factory):
        yield


def settings(output, record):
    argv = [
        "evaluate",
        "--output",
        str(output),
        "--source",
        "imu_encoder_fusion",
        "--duration",
        str(DURATION_S),
        "--split",
        "development",
        "--history",
        "4",
        "--measurement-delay",
        str(DELAY_STEPS),
    ]
    for name, value in GAINS.items():
        argv += ["--" + name.replace("_", "-"), str(value)]
    if record["training_seed"] is not None:
        directory = Path(record["directory"])
        argv += [
            "--policy",
            str(directory / "checkpoint.zip"),
            "--metadata",
            str(directory / "checkpoint.json"),
        ]
    return experiment.parser().parse_args(argv)


def inspect_checkpoints(study):
    """Reject missing/partial/wrong experiments before simulation or output creation."""
    records = [{"controller": "zero_residual", "training_seed": None}]
    common_sources = None
    for seed in TRAINING_SEEDS:
        directory = Path(study) / f"h4_dr_seed{seed}"
        if (directory / "failure.json").exists():
            raise ValueError(f"training has a failure marker: {directory}")
        if any(not (directory / name).is_file() for name in INPUT_NAMES):
            raise ValueError(
                f"all final checkpoint and source artifacts must be ready: {directory}"
            )
        hashes = {name: sha256(directory / name) for name in INPUT_NAMES}
        protocol = read_json(directory / "protocol.json")
        args = protocol["arguments"]
        expected = {
            "mode": "train",
            "baseline": "wheel_leg",
            "source": "imu_encoder_fusion",
            "history": 4,
            "delay_randomization": True,
            "steps": FINAL_STEPS,
            "seed": seed,
            "workers": 4,
            "duration": DURATION_S,
            "measurement_delay": 0,
        }
        if any(type(args.get(k)) is not type(v) or args[k] != v for k, v in expected.items()):
            raise ValueError(f"not the declared final h4+DR training run: {directory}")
        training_args = argparse.Namespace(**args)
        if asdict(experiment.wheel_control_config(training_args)) != GAINS:
            raise ValueError(f"training controller gains differ: {directory}")
        if protocol["sensor_noise"] != json_value(asdict(experiment.NOISE)):
            raise ValueError(f"training base sensor noise differs: {directory}")
        metadata = read_json(directory / "checkpoint.json")
        training = read_json(directory / "training.json")
        if (
            metadata["extra"] != {"training_seed": seed, "num_timesteps": FINAL_STEPS}
            or metadata["history_length"] != 4
            or metadata["controller_parameters"] != GAINS
            or training["num_timesteps"] != FINAL_STEPS
            or metadata["model_sha256"] != hashes["checkpoint.zip"]
            or training["checkpoint_sha256"] != hashes["checkpoint.zip"]
        ):
            raise ValueError(
                f"final checkpoint/sidecar/training hash or budget differs: {directory}"
            )
        if metadata["recorded_episode"]["randomization"] != json_value(
            asdict(TRAINING_RANDOMIZATION)
        ):
            raise ValueError(f"training randomization differs: {directory}")
        consistency = read_json(directory / "source_consistency.json")
        if consistency != {"unchanged": True, "changed": []}:
            raise ValueError(f"training source consistency is not complete: {directory}")
        source = read_json(directory / "source.json")
        if source["archive_sha256"] != hashes["source.tar.gz"]:
            raise ValueError(f"training source archive hash differs: {directory}")
        sources = source["sha256"]
        required = {"scripts/run_d1_locomotion_experiment.py", "pyproject.toml"}
        required.update(
            str(p.relative_to(experiment.ROOT)) for p in (experiment.ROOT / "src").rglob("*.py")
        )
        if not required.issubset(sources):
            raise ValueError("training source record omits runtime Python sources")
        for name, digest in sources.items():
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or not path.parts
                or path.parts[0] not in ("src", "scripts", "pyproject.toml")
            ):
                raise ValueError("unsafe training source path")
            if common_sources is None and sha256(experiment.ROOT / name) != digest:
                raise ValueError(f"runtime differs from frozen training source: {name}")
        if common_sources is not None and sources != common_sources:
            raise ValueError("three training runs do not share the same frozen sources")
        common_sources = sources
        records.append(
            {
                "controller": f"h4_dr_seed{seed}",
                "training_seed": seed,
                "directory": str(directory.resolve()),
                "input_sha256": hashes,
            }
        )
    return records


def check_inputs_unchanged(records):
    for record in records[1:]:
        for name, digest in record["input_sha256"].items():
            if sha256(Path(record["directory"]) / name) != digest:
                raise ValueError(f"predeclared input changed: {record['controller']}/{name}")


def validate_loaders(records):
    """One reset-only environment; all policies must load before any episode runs."""
    roads = experiment.locomotion_terrain_configs("development")
    if len(roads) != 2:
        raise ValueError("the predeclared protocol requires exactly two development roads")
    args = settings(Path("unused_preflight_output"), records[0])
    with stress_adapter():
        base = experiment.D1LocomotionEnv(
            baseline=args.baseline,
            episode_seconds=DURATION_S,
            terrain=roads[0],
            command_mode="development",
            provider_config=experiment.provider_config(args.source, DELAY_STEPS),
            wheel_leg_control=experiment.wheel_control_config(args),
        )
    env = experiment.D1ObservationHistory(base, 4)
    try:
        env.reset(seed=EPISODE_SEEDS[0])
        for record in records[1:]:
            directory = Path(record["directory"])
            model = load_locomotion_policy(
                directory / "checkpoint.zip", directory / "checkpoint.json", env
            )
            if model.num_timesteps != FINAL_STEPS:
                raise ValueError("deserialized policy is not at the fixed final budget")
            del model
    finally:
        env.close()


def protocol_for(records, study, output):
    cases = [
        {
            "controller": record["controller"],
            "road_index": road,
            "episode_seed": seed,
            "case": f"road{road}_seed{seed}",
        }
        for record in records
        for road in range(2)
        for seed in EPISODE_SEEDS
    ]
    command = [
        sys.executable,
        "scripts/evaluate_d1_unseen_stress.py",
        "--study",
        str(study),
        "--output",
        str(output),
    ]
    return {
        "schema": SCHEMA,
        "controllers": records,
        "cases": cases,
        "case_count": 16,
        "episode_limit_s": DURATION_S,
        "history_length": 4,
        "reproduction_command": shlex.join(command),
        "working_directory": str(experiment.ROOT),
        "scope": "joint extrapolation; not independent-factor attribution or new terrain holdout",
        "selection_rule": "all three h4+DR final 32768-step models; no stress-based selection",
        "replication_unit": "training seed; paired episodes are not independent policy replications",
        "source_kind": "imu_encoder_fusion",
        "controller_parameters": GAINS,
        "measurement_delay_steps": DELAY_STEPS,
        "control_dt_s": 0.01,
        "measurement_delay_s": 0.03,
        "sensor_noise_base": asdict(experiment.NOISE),
        "sensor_noise": asdict(stress_noise()),
        "noise_std_factor": NOISE_FACTOR,
        "scaled_noise_fields": NOISE_STD_FIELDS,
        "noise_scope": "synthetic per-sample standard deviations x2; biases unchanged; not calibration",
        "episode_seeds": EPISODE_SEEDS,
        "measurement_seed_rule": "existing env reset derives sensor RNG seed; actual measurement_seed in each episode JSON, not literally 17/29",
        "randomization": asdict(stress_randomization()),
        "actuator": asdict(ideal_actuator()),
        "friction_scope": "set_domain scales geom_friction[:,0] sliding friction only; torsional/rolling coefficients unchanged; no added motor friction",
        "training_randomization": asdict(TRAINING_RANDOMIZATION),
        "terrain_split": "development",
        "command_split": "development",
        "roads": [asdict(c) for c in experiment.locomotion_terrain_configs("development")],
        "quality_thresholds": experiment.QUALITY,
        "runtime_source": "source.json and source.tar.gz include this adapter and existing evaluator",
        "integrity_scope": "hashes bind local inputs; they do not authenticate provenance or establish performance",
    }


def snapshot_with_adapter(output):
    hashes = experiment.snapshot(output)
    own_name = str(Path(__file__).resolve().relative_to(experiment.ROOT))
    hashes[own_name] = sha256(__file__)
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name in sorted(hashes):
            archive.add(experiment.ROOT / name, arcname=name, recursive=False)
    experiment.write_json(
        output / "source.json",
        {
            "sha256": hashes,
            "archive_sha256": sha256(output / "source.tar.gz"),
        },
    )
    return hashes


def validate_results(results, directory):
    expected = {f"road{road}_seed{seed}" for road in range(2) for seed in EPISODE_SEEDS}
    if len(results) != 4 or {row["case"] for row in results} != expected:
        raise ValueError("evaluator returned missing, duplicate or unexpected cases")
    if read_json(directory / "evaluation.json") != results:
        raise ValueError("evaluation.json differs from returned results")
    for row in results:
        if not (1 <= row["executed_steps"] <= 6000 and 0 < row["duration_s"] <= 60.00000001):
            raise ValueError("case has no executed samples or exceeds the declared duration")
        for suffix in (".csv", ".npz", "_episode.json"):
            path = directory / (row["case"] + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing/empty evaluator artifact: {path}")
        episode = read_json(directory / (row["case"] + "_episode.json"))
        if (
            episode["domain"]["friction_scale"] != FRICTION_SCALE
            or episode["provider"]["sensor_delay_steps"] != DELAY_STEPS
            or episode["provider"]["sensor_noise"] != json_value(asdict(stress_noise()))
            or episode["actuator"] != json_value(asdict(ideal_actuator()))
            or episode["controller_parameters"] != GAINS
        ):
            raise ValueError("actual episode settings differ from the stress protocol")


def run(study, output, *, preflight_only=False):
    study, output = Path(study).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must not exist; failed runs are retained")
    experiment.torch.set_num_threads(1)
    print(json.dumps({"status": "preflight", "study": str(study)}), flush=True)
    records = inspect_checkpoints(study)
    validate_loaders(records)
    check_inputs_unchanged(records)
    protocol = protocol_for(records, study, output)
    if preflight_only:
        print(json.dumps({"status": "ready", "protocol": protocol}), flush=True)
        return protocol
    output.mkdir(parents=True, exist_ok=False)
    try:
        experiment.write_json(output / "protocol.json", protocol)
        hashes = snapshot_with_adapter(output)
        all_results = []
        for record in records:
            check_inputs_unchanged(records)
            directory = output / record["controller"]
            directory.mkdir()
            args = settings(directory, record)
            experiment.write_json(
                directory / "protocol.json",
                {
                    "schema": SCHEMA,
                    "parent_protocol": "../protocol.json",
                    "runtime_source": "../source.json",
                    "arguments": {
                        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    },
                },
            )
            print(
                json.dumps(
                    {
                        "status": "evaluating",
                        "controller": record["controller"],
                        "expected_cases": 4,
                    }
                ),
                flush=True,
            )
            with stress_adapter():
                results = experiment.evaluate(args, directory)
            validate_results(results, directory)
            all_results.extend({"controller": record["controller"], **row} for row in results)
        check_inputs_unchanged(records)
        experiment.verify_source(output, hashes)
        experiment.write_json(output / "evaluation.json", all_results)
        completion = {
            "status": "complete",
            "case_count": len(all_results),
            "completed_episodes": sum(r["completed"] for r in all_results),
            "quality_passes": sum(r["quality_pass"] for r in all_results),
        }
        experiment.write_json(output / "completion.json", completion)
        print(json.dumps(completion), flush=True)
        return all_results
    except BaseException as error:
        experiment.write_json(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "scope": "incomplete run; retain partial evaluator artifacts; do not treat as 16 cases",
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        required=True,
        help="contains h4_dr_seed27000/28000/29000 final training directories",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="load-check all three trusted models, print protocol, create no output",
    )
    args = parser.parse_args(argv)
    try:
        run(args.study, args.output, preflight_only=args.preflight_only)
    except Exception as error:  # noqa: BLE001 -- CLI must report preflight failures too.
        print(
            json.dumps({"status": "failed", "type": type(error).__name__, "message": str(error)}),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
