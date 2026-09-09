"""Fixed joint-stress adapter tests; no training or 60-second evaluation."""

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_d1_unseen_stress.py"


@pytest.fixture(scope="module")
def stress():
    spec = importlib.util.spec_from_file_location("d1_unseen_stress_test_subject", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def episode(stress):
    return {
        "domain": {"friction_scale": 0.6},
        "provider": {"sensor_delay_steps": 3, "sensor_noise": asdict(stress.stress_noise())},
        "actuator": asdict(stress.ideal_actuator()),
        "controller_parameters": stress.GAINS,
    }


@pytest.fixture
def study(stress, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / "src").mkdir()
    sources = {}
    for name in ("scripts/run_d1_locomotion_experiment.py", "pyproject.toml", "src/dummy.py"):
        (repo / name).write_text("fixture source\n")
        sources[name] = stress.sha256(repo / name)
    monkeypatch.setattr(stress.experiment, "ROOT", repo)
    result = tmp_path / "study"
    result.mkdir()
    for seed in stress.TRAINING_SEEDS:
        directory = result / f"h4_dr_seed{seed}"
        directory.mkdir()
        (directory / "checkpoint.zip").write_bytes(f"fixture checkpoint {seed}".encode())
        (directory / "source.tar.gz").write_bytes(b"fixture archive")
        digest = stress.sha256(directory / "checkpoint.zip")
        args = vars(
            stress.experiment.parser().parse_args(
                [
                    "train",
                    "--output",
                    str(directory),
                    "--source",
                    "imu_encoder_fusion",
                    "--history",
                    "4",
                    "--delay-randomization",
                    "--seed",
                    str(seed),
                ]
            )
        )
        args.update(stress.GAINS)
        # The actual study leaves this argument unspecified: its resolved value is one.
        args["leg_feedback_scale"] = None
        args["output"] = str(args["output"])
        write_json(
            directory / "protocol.json",
            {
                "arguments": args,
                "sensor_noise": asdict(stress.experiment.NOISE),
            },
        )
        write_json(
            directory / "checkpoint.json",
            {
                "extra": {"training_seed": seed, "num_timesteps": 32768},
                "history_length": 4,
                "controller_parameters": stress.GAINS,
                "model_sha256": digest,
                "recorded_episode": {"randomization": asdict(stress.TRAINING_RANDOMIZATION)},
            },
        )
        write_json(
            directory / "training.json",
            {
                "num_timesteps": 32768,
                "checkpoint_sha256": digest,
            },
        )
        write_json(directory / "source_consistency.json", {"unchanged": True, "changed": []})
        write_json(
            directory / "source.json",
            {
                "archive_sha256": stress.sha256(directory / "source.tar.gz"),
                "sha256": sources,
            },
        )
    return result


def test_fixed_plan_names_seeds_gains_units_and_noise_meaning(stress, study, tmp_path):
    records = stress.inspect_checkpoints(study)
    protocol = stress.protocol_for(records, study, tmp_path / "new")
    assert [r["controller"] for r in records] == [
        "zero_residual",
        "h4_dr_seed27000",
        "h4_dr_seed28000",
        "h4_dr_seed29000",
    ]
    assert (
        len(protocol["cases"])
        == len(
            {
                (row["controller"], row["road_index"], row["episode_seed"])
                for row in protocol["cases"]
            }
        )
        == 16
    )
    assert protocol["measurement_delay_steps"] * protocol["control_dt_s"] == 0.03
    assert protocol["episode_limit_s"] == 60
    assert protocol["terrain_split"] == protocol["command_split"] == "development"
    assert "not independent-factor" in protocol["scope"]
    assert "not literally 17/29" in protocol["measurement_seed_rule"]
    noise = protocol["sensor_noise"]
    assert noise["gyro_std_rad_s"] == 0.004
    assert noise["accelerometer_std_m_s2"] == 0.06
    assert noise["encoder_position_std_rad"] == 0.001
    assert noise["encoder_velocity_std_rad_s"] == 0.01
    assert noise["gyro_bias_rad_s"] == stress.experiment.NOISE.gyro_bias_rad_s
    for record in records:
        args = stress.settings(tmp_path / "new", record)
        assert args.history == 4 and args.duration == 60 and args.measurement_delay == 3
        assert asdict(stress.experiment.wheel_control_config(args)) == stress.GAINS
        assert bool(args.policy) == (record["training_seed"] is not None)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "failed",
        "budget",
        "seed",
        "history",
        "dr",
        "gains",
        "noise",
        "zip",
        "sidecar_hash",
        "archive",
        "consistency",
        "randomization",
        "source",
    ],
)
def test_preflight_rejects_incomplete_or_wrong_training(stress, study, mutation):
    directory = study / "h4_dr_seed29000"
    if mutation == "missing":
        (directory / "checkpoint.zip").unlink()
    elif mutation == "failed":
        write_json(directory / "failure.json", {})
    elif mutation in ("zip", "archive"):
        name = "checkpoint.zip" if mutation == "zip" else "source.tar.gz"
        (directory / name).write_bytes(b"changed")
    else:
        name = {
            "budget": "training.json",
            "sidecar_hash": "checkpoint.json",
            "consistency": "source_consistency.json",
            "randomization": "checkpoint.json",
            "source": "source.json",
        }.get(mutation, "protocol.json")
        value = json.loads((directory / name).read_text())
        if mutation == "budget":
            value["num_timesteps"] = 16384
        elif mutation == "sidecar_hash":
            value["model_sha256"] = "0" * 64
        elif mutation == "consistency":
            value["unchanged"] = False
        elif mutation == "randomization":
            value["recorded_episode"]["randomization"]["friction_scale"] = [0.6, 0.6]
        elif mutation == "source":
            value["sha256"].pop("src/dummy.py")
        elif mutation == "noise":
            value["sensor_noise"]["gyro_std_rad_s"] *= 2
        else:
            key, new = {
                "seed": ("seed", 28000),
                "history": ("history", 1),
                "dr": ("delay_randomization", False),
                "gains": ("wheel_kp", 2.2),
            }[mutation]
            value["arguments"][key] = new
        write_json(directory / name, value)
    with pytest.raises(ValueError):
        stress.inspect_checkpoints(study)


def test_changed_sealed_input_is_rejected(stress, study):
    records = stress.inspect_checkpoints(study)
    (study / "h4_dr_seed27000" / "checkpoint.zip").write_bytes(b"new")
    with pytest.raises(ValueError, match="predeclared input changed"):
        stress.check_inputs_unchanged(records)


def test_factory_is_bounded_repeatable_and_restores_on_exception(stress, monkeypatch):
    monkeypatch.setattr(stress.experiment, "D1LocomotionEnv", lambda **kwargs: kwargs)
    original = stress.experiment.D1LocomotionEnv
    config = stress.experiment.provider_config("imu_encoder_fusion", 3)
    with pytest.raises(RuntimeError, match="injected"), stress.stress_adapter():
        first = stress.experiment.D1LocomotionEnv(provider_config=config)
        second = stress.experiment.D1LocomotionEnv(provider_config=config)
        assert first == second
        assert first["provider_config"].sensor_noise.gyro_std_rad_s == 0.004
        assert config.sensor_noise.gyro_std_rad_s == 0.002
        assert first["randomization"].friction_scale == (0.6, 0.6)
        assert first["actuator_config"].delay_steps == 0
        assert first["actuator_config"].physics_dt_s == 0.002
        assert first["actuator_config"].gain == 1 and first["actuator_config"].time_constant_s == 0
        raise RuntimeError("injected")
    assert stress.experiment.D1LocomotionEnv is original
    for bad in (
        replace(config, sensor_delay_steps=2),
        replace(config, sensor_noise=stress.stress_noise()),
    ):
        with pytest.raises(ValueError, match="declared"):
            stress.stress_kwargs({"provider_config": bad})
    with pytest.raises(ValueError, match="silently replace"):
        stress.stress_kwargs({"provider_config": config, "randomization": None})


def test_actual_environment_uses_stress_at_reset_and_four_ticks(stress):
    record = {"training_seed": None}
    args = stress.settings(Path("unused"), record)
    with stress.stress_adapter():
        env = stress.experiment.D1LocomotionEnv(
            baseline="wheel_leg",
            episode_seconds=60.0,
            terrain=stress.experiment.locomotion_terrain_configs("development")[0],
            command_mode="development",
            provider_config=stress.experiment.provider_config("imu_encoder_fusion", 3),
            wheel_leg_control=stress.experiment.wheel_control_config(args),
        )
    try:
        nominal_friction = env.plant.model.geom_friction.copy()
        obs, info = env.reset(seed=17)
        actual = info["episode_metadata"]
        assert actual["domain"]["friction_scale"] == 0.6
        np.testing.assert_allclose(
            env.plant.model.geom_friction[:, 0], nominal_friction[:, 0] * 0.6
        )
        np.testing.assert_array_equal(env.plant.model.geom_friction[:, 1:], nominal_friction[:, 1:])
        assert actual["provider"]["sensor_delay_steps"] == 3
        assert actual["provider"]["sensor_noise"]["gyro_std_rad_s"] == 0.004
        assert actual["actuator"]["gain"] == 1
        assert actual["actuator"]["delay_steps"] == actual["actuator"]["time_constant_s"] == 0
        assert actual["controller_parameters"] == stress.GAINS
        assert np.isfinite(obs).all()
        for _ in range(4):
            obs, reward, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
            assert not terminated and not truncated and np.isfinite(reward)
        assert info["metrics"]["measurement_age_s"] == pytest.approx(0.03)
        assert info["metrics"]["time_s"] == pytest.approx(0.04)
    finally:
        env.close()


def test_loader_preflight_closes_environment_and_checks_all_three(stress, monkeypatch):
    resets, loads, closed = [], [], []
    fake = SimpleNamespace(reset=lambda **kw: resets.append(kw), close=lambda: closed.append(True))
    monkeypatch.setattr(stress.experiment, "D1LocomotionEnv", lambda **kw: fake)
    monkeypatch.setattr(stress.experiment, "D1ObservationHistory", lambda base, n: base)

    def loader(model, sidecar, env):
        loads.append((model.parent.name, env))
        return SimpleNamespace(num_timesteps=32768)

    monkeypatch.setattr(stress, "load_locomotion_policy", loader)
    records = [{"training_seed": None}] + [
        {"training_seed": s, "directory": f"h4_dr_seed{s}"} for s in stress.TRAINING_SEEDS
    ]
    stress.validate_loaders(records)
    assert resets == [{"seed": 17}] and len(loads) == 3 and closed == [True]
    monkeypatch.setattr(
        stress, "load_locomotion_policy", lambda *a: SimpleNamespace(num_timesteps=8)
    )
    with pytest.raises(ValueError, match="fixed final budget"):
        stress.validate_loaders(records)
    assert closed == [True, True]


@pytest.fixture
def orchestration(stress, tmp_path, monkeypatch):
    records = [{"controller": "zero_residual", "training_seed": None}] + [
        {
            "controller": f"h4_dr_seed{s}",
            "training_seed": s,
            "directory": str(tmp_path / f"h4_dr_seed{s}"),
            "input_sha256": {},
        }
        for s in stress.TRAINING_SEEDS
    ]
    monkeypatch.setattr(stress, "inspect_checkpoints", lambda study: records)
    monkeypatch.setattr(stress, "validate_loaders", lambda records: None)
    monkeypatch.setattr(stress, "snapshot_with_adapter", lambda directory: {})
    monkeypatch.setattr(stress.experiment, "verify_source", lambda *args: None)
    return tmp_path / "output"


def test_preflight_creates_nothing_and_rejects_existing_output(stress, orchestration):
    protocol = stress.run("unused", orchestration, preflight_only=True)
    assert protocol["case_count"] == 16 and not orchestration.exists()
    orchestration.mkdir()
    with pytest.raises(ValueError, match="must not exist"):
        stress.run("unused", orchestration)


def fake_evaluation(stress, directory):
    rows = []
    for road in range(2):
        for seed in stress.EPISODE_SEEDS:
            name = f"road{road}_seed{seed}"
            rows.append(
                {
                    "case": name,
                    "executed_steps": 3,
                    "duration_s": 0.03,
                    "completed": False,
                    "quality_pass": False,
                    "terminal_reason": "fall_or_body_contact",
                }
            )
            for suffix in (".csv", ".npz"):
                (directory / (name + suffix)).write_bytes(b"fixture artifact")
            write_json(directory / (name + "_episode.json"), episode(stress))
    write_json(directory / "evaluation.json", rows)
    return rows


def test_orchestration_keeps_failures_and_runs_exact_original_four_batches(
    stress, orchestration, monkeypatch
):
    seen = []

    def evaluate(args, directory):
        seen.append((args, directory))
        return fake_evaluation(stress, directory)

    monkeypatch.setattr(stress.experiment, "evaluate", evaluate)
    rows = stress.run("unused", orchestration)
    assert len(seen) == 4 and len(rows) == 16
    assert all(not row["quality_pass"] for row in rows)
    completion = json.loads((orchestration / "completion.json").read_text())
    assert completion == {
        "status": "complete",
        "case_count": 16,
        "completed_episodes": 0,
        "quality_passes": 0,
    }
    assert not (orchestration / "failure.json").exists()


@pytest.mark.parametrize("failure", ["empty", "exception", "wrong_domain"])
def test_silent_partial_and_exception_results_fail_visibly(
    stress, orchestration, monkeypatch, failure
):
    def evaluate(args, directory):
        if failure == "empty":
            return []
        if failure == "exception":
            raise RuntimeError("injected evaluator failure")
        rows = fake_evaluation(stress, directory)
        bad = episode(stress)
        bad["domain"]["friction_scale"] = 1
        write_json(directory / "road0_seed17_episode.json", bad)
        return rows

    monkeypatch.setattr(stress.experiment, "evaluate", evaluate)
    with pytest.raises((RuntimeError, ValueError)):
        stress.run("unused", orchestration)
    assert (orchestration / "failure.json").is_file()
    assert not (orchestration / "completion.json").exists()
    assert not (orchestration / "evaluation.json").exists()


def test_cli_reports_missing_models_before_output_mutation(stress, tmp_path, capsys):
    output = tmp_path / "new"
    assert stress.main(["--study", str(tmp_path), "--output", str(output)]) == 1
    assert not output.exists()
    assert json.loads(capsys.readouterr().err)["status"] == "failed"


def test_snapshot_includes_actual_adapter_without_importing_an_alternate_evaluator(
    stress, tmp_path
):
    import tarfile

    hashes = stress.snapshot_with_adapter(tmp_path)
    name = "scripts/evaluate_d1_unseen_stress.py"
    assert hashes[name] == stress.sha256(SCRIPT)
    assert "scripts/run_d1_locomotion_experiment.py" in hashes
    with tarfile.open(tmp_path / "source.tar.gz") as archive:
        assert archive.extractfile(name).read() == SCRIPT.read_bytes()
    stress.experiment.verify_source(tmp_path, hashes)
