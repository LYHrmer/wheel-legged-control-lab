"""Experiment accounting and fail-closed CLI tests; no PPO training/benchmark."""

import hashlib
import importlib.util
import json
import math
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_d1_locomotion_experiment.py"


@pytest.fixture(scope="module")
def experiment():
    spec = importlib.util.spec_from_file_location("d1_locomotion_experiment_test_subject", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(**updates):
    result = {
        "time_s": 0.01,
        "velocity_error_mps": 0.0,
        "yaw_rate_error_rps": 0.0,
        "height_error_m": 0.0,
        "roll_error_rad": 0.0,
        "pitch_error_rad": 0.0,
        "mechanical_power_w": 0.0,
        "action_mean_square": 0.0,
    }
    result.update(updates)
    return result


def _terminal(reason="time_limit"):
    return {
        "terminal_reason": reason,
        "terrain_exposure": {"nonflat_steps": 0, "nonflat_fraction": 0.0},
    }


def test_summary_matches_hand_calculation_without_relabeling_completion(experiment):
    rows = [
        _row(
            velocity_error_mps=-0.06,
            yaw_rate_error_rps=-0.04,
            height_error_m=0.01,
            roll_error_rad=0.03,
            pitch_error_rad=0.04,
            mechanical_power_w=100,
            action_mean_square=0.25,
        ),
        _row(
            time_s=0.02,
            velocity_error_mps=0.08,
            yaw_rate_error_rps=0.06,
            height_error_m=0.02,
            roll_error_rad=0.06,
            pitch_error_rad=0.08,
            mechanical_power_w=140,
            action_mean_square=1.0,
        ),
    ]
    summary = experiment.summarize(rows, _terminal())
    assert summary["executed_steps"] == 2
    assert summary["duration_s"] == 0.02
    assert summary["velocity_rmse_mps"] == pytest.approx(math.sqrt(0.005))
    assert summary["yaw_rmse_rps"] == pytest.approx(math.sqrt(0.0026))
    assert summary["height_rmse_m"] == pytest.approx(math.sqrt(0.00025))
    assert summary["attitude_rmse_rad"] == pytest.approx(math.sqrt(0.00625))
    assert summary["mean_mechanical_power_w"] == 120
    assert summary["action_rms"] == pytest.approx(math.sqrt(0.625))
    assert summary["completed"] and summary["quality_pass"]
    assert summary["terrain_exposure"]["nonflat_steps"] == 0
    for reason in (None, "map_boundary", "fall_or_body_contact", "ground_query_outside_map"):
        stopped = experiment.summarize(rows, _terminal(reason))
        assert not stopped["completed"] and not stopped["quality_pass"]


@pytest.mark.parametrize(
    "metric,row_field",
    (
        ("velocity_rmse_mps", "velocity_error_mps"),
        ("yaw_rmse_rps", "yaw_rate_error_rps"),
        ("height_rmse_m", "height_error_m"),
        ("attitude_rmse_rad", "roll_error_rad"),
    ),
)
def test_quality_gate_is_inclusive_at_declared_limit_and_fails_above(experiment, metric, row_field):
    limit = experiment.QUALITY[metric]
    assert experiment.QUALITY["height_rmse_m"] == 0.025
    on_limit = experiment.summarize([_row(**{row_field: limit})], _terminal())
    above = experiment.summarize([_row(**{row_field: limit + 1e-8})], _terminal())
    assert on_limit["completed"] and on_limit["quality_pass"]
    assert above["completed"] and not above["quality_pass"]


def test_source_configuration_preserves_source_and_delay_semantics(experiment):
    oracle = experiment.provider_config("oracle", 0)
    impaired = experiment.provider_config("truth_impairment", 2)
    fusion = experiment.provider_config("imu_encoder_fusion", 2)
    assert oracle.kind == "oracle" and oracle.impairments is None
    assert impaired.kind == "truth_impairment" and impaired.impairments.delay_steps == 2
    assert impaired.sensor_noise is None
    assert fusion.sensor_delay_steps == 2 and fusion.impairments is None
    assert fusion.sensor_noise == experiment.NOISE
    assert fusion.initial_position_m == (-3.8, 0, 0.455)
    assert fusion.initial_rpy_rad == (0, 0, 0)
    assert len({c.source_schema for c in (oracle, impaired, fusion)}) == 3
    with pytest.raises(ValueError):
        experiment.provider_config("estimated", 0)


@pytest.mark.parametrize(
    "kind,delay", (("oracle", 2), ("oracle", -1), ("oracle", 1.5), ("oracle", True))
)
def test_source_helper_never_silently_drops_invalid_delay(experiment, kind, delay):
    with pytest.raises((TypeError, ValueError)):
        experiment.provider_config(kind, delay)


def test_four_training_workers_have_distinct_actual_collision_roads(experiment):
    hashes, configs = [], []
    for worker in range(4):
        env = experiment.make_env("wheel_leg", worker, 0.02, "oracle", 2)
        try:
            observation, _ = env.reset(seed=17)
            base = env.unwrapped
            model = base.plant.model
            hashes.append(
                hashlib.sha256(
                    model.hfield_data.tobytes()
                    + model.hfield_size.tobytes()
                    + model.geom_pos.tobytes()
                ).hexdigest()
            )
            configs.append(base.terrain)
            assert env.history_length == 2
            assert observation.shape == env.observation_space.shape
            assert env.source_schema == base.source_schema
        finally:
            env.close()
    assert len(set(hashes)) == 4
    assert tuple(configs) == experiment.locomotion_terrain_configs("train")


def test_fake_vector_benchmark_counts_samples_excludes_warmup_and_closes(
    experiment, monkeypatch, tmp_path
):
    vectors = []

    class Vector:
        def __init__(self, workers):
            self.workers = workers
            self.action_space = SimpleNamespace(shape=(8,))
            self.calls, self.closed, self.seed_value = 0, False, None

        def seed(self, seed):
            self.seed_value = seed

        def reset(self):
            return None

        def step(self, action):
            assert action.shape == (self.workers, 8)
            np.testing.assert_array_equal(action, 0.0)
            self.calls += 1

        def close(self):
            self.closed = True

    def make_vec(args, workers):
        vector = Vector(workers)
        vectors.append(vector)
        return vector

    clock = iter(np.arange(1000) * 0.001)
    monkeypatch.setattr(experiment, "make_vec", make_vec)
    monkeypatch.setattr(experiment, "perf_counter", lambda: float(next(clock)))
    monkeypatch.setattr(
        experiment,
        "process_memory",
        lambda: {
            "process_tree_rss_mb": 100.0,
            "process_tree_uss_mb": 60.0,
            "system_available_mb": 900.0,
        },
    )
    result = experiment.benchmark(SimpleNamespace(seed=77, steps=9), tmp_path)
    assert [row["samples"] for row in result] == [9, 10, 12]
    for row, vector in zip(result, vectors, strict=True):
        calls = math.ceil(9 / vector.workers)
        assert vector.calls == 25 + calls
        assert vector.closed and vector.seed_value == 77
        assert row["wall_s"] == pytest.approx((2 * calls + 1) * 0.001)
        assert row["samples_per_second"] == pytest.approx(row["samples"] / row["wall_s"])
        assert row["vector_step_median_ms"] == pytest.approx(1.0)
        assert row["vector_step_p99_ms"] == pytest.approx(1.0)
        assert row["max_observed_tree_uss_mb"] == 60.0
        assert row["max_observed_tree_rss_mb"] == 100.0
        assert row["min_observed_system_available_mb"] == 900.0
    assert json.loads((tmp_path / "benchmark.json").read_text()) == result


def test_snapshot_records_actual_archive_bytes_and_detects_source_change(
    experiment, monkeypatch, tmp_path
):
    repo, output = tmp_path / "repo", tmp_path / "output"
    relative_files = {
        "scripts/run_d1_locomotion_experiment.py": b"# fixture entry point\n",
        "pyproject.toml": b"[project]\nname='fixture'\n",
        "src/example.py": b"VALUE = 1\n",
        "src/wheel_legged_control/d1/assets/robot.urdf": b"<robot/>\n",
        "src/wheel_legged_control/d1/assets/part.STL": b"fixture mesh bytes",
    }
    for name, contents in relative_files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    output.mkdir()
    monkeypatch.setattr(experiment, "ROOT", repo)
    monkeypatch.setattr(
        experiment, "__file__", str(repo / "scripts/run_d1_locomotion_experiment.py")
    )
    hashes = experiment.snapshot(output)
    assert set(hashes) == set(relative_files)
    manifest = json.loads((output / "source.json").read_text())
    assert manifest["sha256"] == hashes
    assert (
        manifest["archive_sha256"]
        == hashlib.sha256((output / "source.tar.gz").read_bytes()).hexdigest()
    )
    with tarfile.open(output / "source.tar.gz") as archive:
        assert set(archive.getnames()) == set(relative_files)
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
    experiment.verify_source(output, hashes)
    assert json.loads((output / "source_consistency.json").read_text())["unchanged"]
    (repo / "src/example.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="source changed"):
        experiment.verify_source(output, hashes)
    assert json.loads((output / "source_consistency.json").read_text()) == {
        "unchanged": False,
        "changed": ["src/example.py"],
    }


@pytest.mark.parametrize(
    "mode,extra",
    (
        ("train", ["--steps", "1"]),
        ("train", ["--steps", "0"]),
        ("benchmark", ["--duration", "nan"]),
        ("benchmark", ["--duration", "inf"]),
        ("benchmark", ["--duration", "0.015"]),
        ("benchmark", ["--history", "0"]),
        ("benchmark", ["--seed", "-1"]),
        ("train", ["--source", "imu_encoder_fusion", "--measurement-delay", "2"]),
        ("evaluate", ["--source", "imu_encoder_fusion", "--measurement-delay", "-1"]),
        ("evaluate", ["--source", "imu_encoder_fusion", "--delay-randomization"]),
        ("train", ["--delay-randomization"]),
        ("evaluate", ["--measurement-delay", "1"]),
        ("train", ["--split", "holdout"]),
        ("benchmark", ["--split", "flat"]),
        ("train", ["--policy", "POLICY", "--metadata", "META"]),
        ("evaluate", ["--policy", "POLICY"]),
    ),
)
def test_invalid_cli_is_rejected_before_creating_output_or_running_work(
    experiment, monkeypatch, tmp_path, mode, extra
):
    policy, metadata = tmp_path / "checkpoint.zip", tmp_path / "checkpoint.json"
    policy.write_bytes(b"not loaded by this rejection test")
    metadata.write_text("{}")
    replacements = {"POLICY": str(policy), "META": str(metadata)}
    output = tmp_path / "result"
    argv = [str(SCRIPT), mode, "--output", str(output), *[replacements.get(v, v) for v in extra]]
    monkeypatch.setattr(sys, "argv", argv)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CLI started experiment work")

    for name in ("benchmark", "train", "evaluate", "snapshot"):
        monkeypatch.setattr(experiment, name, forbidden)
    with pytest.raises(SystemExit) as stopped:
        experiment.main()
    assert stopped.value.code == 2
    assert not output.exists()


def test_existing_output_is_never_overwritten(experiment, monkeypatch, tmp_path):
    (tmp_path / "keep.txt").write_text("user experiment")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "benchmark", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        experiment.main()
    assert stopped.value.code == 2
    assert (tmp_path / "keep.txt").read_text() == "user experiment"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_protocol_written_before_dispatch_and_source_verified_afterward(
    experiment, monkeypatch, tmp_path
):
    output = tmp_path / "new"
    calls = []
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "benchmark", "--output", str(output), "--steps", "9"]
    )
    monkeypatch.setattr(experiment, "version", lambda name: "fixture-version")

    def snapshot(directory):
        calls.append("snapshot")
        assert (directory / "protocol.json").is_file()
        return {"fixture.py": "fixture-hash"}

    def benchmark(args, directory):
        calls.append("benchmark")
        assert args.steps == 9 and directory == output

    def verify(directory, hashes):
        calls.append("verify")
        assert directory == output and hashes == {"fixture.py": "fixture-hash"}

    monkeypatch.setattr(experiment, "snapshot", snapshot)
    monkeypatch.setattr(experiment, "benchmark", benchmark)
    monkeypatch.setattr(experiment, "verify_source", verify)
    experiment.main()
    assert calls == ["snapshot", "benchmark", "verify"]
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["quality_thresholds"] == experiment.QUALITY
    assert protocol["reward_scale"] == 1.0
    assert protocol["ppo_settings"]["gamma"] == pytest.approx(math.exp(-0.01 / 2.0))
    assert protocol["discount_horizon_s"] == 2.0
    assert "training seed" in protocol["replication_unit"]
    assert "no holdout" in protocol["selection_rule"]
    assert protocol["arguments"]["output"] == str(output)
    assert protocol["arguments"]["duration"] == 60.0
    assert all(value == "fixture-version" for value in protocol["versions"].values())
